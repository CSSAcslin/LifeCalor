import copy
import logging
import shutil
import uuid
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pywt
import h5py
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot, QElapsedTimer, QThread, QSettings
from typing import List
from scipy import signal
from scipy.optimize import curve_fit
from scipy.ndimage import zoom
from DataManager import *
from ArrayCache import ArrayRef

from ResultDisplayWidget import HeartbeatDraw
from diagnostics import AppError, format_exception_details
from calculator import CalculationEngine, CalculationPlan
from calculator.metadata import CalculationMetadataPolicy
from compute import (
    BackendPreference, CapabilityStatus, ComputeRequest, PrecisionPolicy,
    ResourceBudget, compatibility_metadata, execution_metadata, plan_compute,
    resolve_precision,
)
from compute.algorithms import (
    build_cwt_kernel_bank,
    cwt_frequency_trace,
    cwt_quality_spectrum,
    stft_frequency_trace,
)
from compute.algorithms.cwt import cwt_axes
from compute.algorithms.stft_pipeline import run_stft_pipeline
from compute.algorithms.cwt_pipeline import run_cwt_pipeline
from compute.algorithms.preprocess_pipeline import run_preprocess_pipeline
from compute.capabilities import (
    mark_algorithm_verified,
    update_cached_device_capability,
)
from compute.worker import (
    CudaBackendError,
    CudaWorkerClient,
    probe_cuda_capability_isolated,
)
from tasks import CancellationToken, TaskCancelled, TaskContextQueue, task_scoped
from memory import MemoryBudget, array_nbytes, estimate_resident_bytes
from processing.time_axis import scientific_time_axis


@contextmanager
def _array_view(source):
    """Release only mappings opened for this operation."""
    if source is None:
        raise ValueError("没有可处理的科学数组")
    loaded_here = isinstance(source, ArrayRef)
    array = source.load(mmap_mode="r") if loaded_here else np.asanyarray(source)
    try:
        yield array
    finally:
        if loaded_here:
            mapping = getattr(array, "_mmap", None)
            if mapping is not None:
                mapping.close()


def _ensure_working_memory(data, additional_bytes, operation):
    budget_mb = QSettings().value("tool/memory_budget_mb", 4096)
    return MemoryBudget.from_megabytes(budget_mb).ensure(
        additional_bytes,
        estimate_resident_bytes([data, *Data.history, *ProcessedData.history]),
        operation=operation,
    )


def _frame_stack(array):
    if array.ndim not in (2, 3) or any(size == 0 for size in array.shape):
        raise ValueError(f"需要非空 HW 或 THW 数组，实际 shape={array.shape}")
    if array.dtype.kind not in "biufc":
        raise ValueError("仅支持数值数组")
    return array[np.newaxis, ...] if array.ndim == 2 else array


def _analysis_metadata(data):
    metadata = getattr(data, "out_processed", None) or getattr(data, "parameters", None)
    return dict(metadata or {})


def _primary_data_source(data):
    if isinstance(data, ProcessedData):
        values = object.__getattribute__(data, "__dict__")
        source = values.get("_data_processed_storage")
        return source if source is not None else data.data_processed
    if isinstance(data, Data):
        values = object.__getattribute__(data, "__dict__")
        source = values.get("_data_origin_storage")
        return source if source is not None else data.data_origin
    source = getattr(data, "data_processed", None)
    if source is None:
        source = getattr(data, "data_origin", None)
    return source


def get_unfolded_data(data):
    """Return pixel-major [pixels, frames] data without persisting a duplicate."""
    out_processed = getattr(data, "out_processed", {}) or {}
    if "unfolded_data" in out_processed:
        return out_processed["unfolded_data"]
    source = _primary_data_source(data)
    if source is None:
        raise ValueError("无法展开数据：缺少 data_processed/data_origin")
    array = source.load(mmap_mode="r") if isinstance(source, ArrayRef) else source
    if array.ndim != 3:
        raise ValueError(f"无法展开数据：期望3维数组，实际维度 {array.ndim}")
    t, h, w = array.shape
    return array.reshape((t, h * w)).T


def get_mean_time_trace(data, token=None, row_step=32):
    """Compute the spatial mean without unfolding or copying the complete THW array."""
    source = _primary_data_source(data)
    if source is None:
        raise ValueError("无法计算平均信号：缺少 data_processed/data_origin")
    loaded_here = isinstance(source, ArrayRef)
    array = source.load(mmap_mode="r") if loaded_here else np.asanyarray(source)
    try:
        if array.ndim != 3:
            raise ValueError(f"平均信号要求 THW 三维数据，实际 shape={array.shape}")
        accumulator_dtype = np.complex128 if np.iscomplexobj(array) else np.float64
        total = np.zeros(array.shape[0], dtype=accumulator_dtype)
        for row_start in range(0, array.shape[1], max(1, int(row_step))):
            if token is not None:
                token.raise_if_cancelled()
            row_stop = min(array.shape[1], row_start + max(1, int(row_step)))
            total += np.asarray(array[:, row_start:row_stop, :]).sum(
                axis=(1, 2), dtype=accumulator_dtype
            )
        return total / (array.shape[1] * array.shape[2])
    finally:
        if loaded_here:
            mapping = getattr(array, "_mmap", None)
            if mapping is not None:
                mapping.close()

class DataProcessor(QObject):
    """本类包含所有非计算流程的操作（常开线程）"""
    plot_singal = pyqtSignal(np.ndarray,dict)
    plot_series_signal = pyqtSignal(np.ndarray, str)
    processing_error_signal = pyqtSignal(object)
    bad_frames_detected = pyqtSignal(object)
    bad_frames_fixed = pyqtSignal(object)
    bad_frame_failed = pyqtSignal(object)
    def __init__(self):
        super().__init__()
        logging.info("例外数据处理线程已启动")

    @staticmethod
    def process_data(data, max_all, min_all, vmean_array):
        process_show = []
        span = max_all - min_all
        denominator = span if span != 0 else 1.0
        if np.abs(min_all) > np.abs(max_all):
            # n-type 信号中心为黑色，最强值为负
            data_type = 'central negative'
            for every_data in data:
                normalized_data = (every_data - min_all) / denominator
                process_show.append(normalized_data)
            max_mean = np.min(vmean_array)
            phy_max = -min_all
            phy_min = -max_all
        else:
            # p-type 信号中心为白色，最强值为正
            data_type = 'central positive'
            for every_data in data:
                normalized_data = (max_all - every_data) / denominator
                process_show.append(normalized_data)
            max_mean = np.max(vmean_array)
            phy_max = max_all
            phy_min = min_all
        return process_show, data_type, max_mean, phy_max, phy_min

    @pyqtSlot(object)
    def amend_data(self, data):
        """函数修改方法
        输入修改的源数据，导出修改的数据包"""
        data_origin = data
        vmax_array = []
        vmin_array = []
        vmean_array = []
        for data in data_origin:
            vmax_array.append(np.max(data))
            vmin_array.append(np.min(data))
            vmean_array.append(np.mean(data))
        vmax = np.max(vmax_array)
        vmin = np.min(vmin_array)

        images_show, data_type, max_mean, phy_max, phy_min = self.process_data(data_origin, vmax, vmin, vmean_array)

        return {
            'data_origin' : data_origin,
            'image_import': np.stack(images_show, axis=0),
        }

    @pyqtSlot(np.ndarray, float)
    def detect_bad_frames_auto(self, data: np.ndarray, threshold: float = 3.0) -> List[int]:
        """Detect abrupt frame-mean differences and publish the asynchronous result."""
        try:
            values = np.asanyarray(data)
            if values.ndim != 3 or any(size == 0 for size in values.shape):
                raise ValueError("坏帧检测需要非空 THW 数据")
            if values.dtype.kind not in "biuf":
                raise ValueError("坏帧检测需要有限的实数数据")
            if not np.isfinite(threshold) or float(threshold) <= 0:
                raise ValueError("检测阈值必须为有限正数")
            means = np.empty(values.shape[0], dtype=np.float64)
            for index, frame in enumerate(values):
                if not np.all(np.isfinite(frame)):
                    raise ValueError("坏帧检测需要有限的实数数据")
                means[index] = np.mean(frame, dtype=np.float64)
            differences = np.abs(np.diff(means))
            if differences.size == 0:
                bad_frames = []
            else:
                median = np.median(differences)
                mad = 1.4826 * np.median(np.abs(differences - median))
                tolerance = np.finfo(np.float64).eps * max(1.0, float(np.max(np.abs(means)))) * 16
                if mad > tolerance:
                    outliers = np.abs(differences - median) / mad > float(threshold)
                else:
                    outliers = np.abs(differences - median) > tolerance
                bad_indices = set()
                for index in np.flatnonzero(outliers):
                    bad_indices.update((int(index), int(index) + 1))
                bad_frames = sorted(bad_indices)
            self.bad_frames_detected.emit(bad_frames)
            return bad_frames
        except Exception as exc:
            self.bad_frame_failed.emit(AppError(
                "坏帧检测失败", str(exc), stage="坏帧检测", original=exc,
                details=format_exception_details(exc, "坏帧检测"),
            ))
            return None

    @pyqtSlot(object, list, int)
    def fix_bad_frames(self, data, bad_frames: List[int], n_frames: int = 2):
        """Build the complete repair before publishing a derived Data object."""
        try:
            if not isinstance(data, Data):
                raise ValueError("坏帧修复需要原始 Data 数据")
            if isinstance(n_frames, bool) or int(n_frames) != n_frames or n_frames < 1:
                raise ValueError("参考帧范围必须为正整数")
            with _array_view(_primary_data_source(data)) as values:
                if values.ndim != 3 or any(size == 0 for size in values.shape):
                    raise ValueError("坏帧修复需要非空 THW 数据")
                if values.dtype.kind not in "biuf" or not all(np.all(np.isfinite(frame)) for frame in values):
                    raise ValueError("坏帧修复需要有限的实数数据")
                indices = set()
                for index in bad_frames:
                    if isinstance(index, bool) or int(index) != index or not 0 <= int(index) < len(values):
                        raise ValueError(f"坏帧编号超出有效范围: {index}")
                    indices.add(int(index))
                dtype = np.result_type(values.dtype, np.float32)
                _ensure_working_memory(data, int(values.size) * np.dtype(dtype).itemsize * 4, "坏帧修复")
                fixed_data = np.array(values, dtype=dtype, copy=True)
                for index in sorted(indices):
                    first = max(0, index - int(n_frames))
                    last = min(len(values), index + int(n_frames) + 1)
                    neighbors = [frame for frame in range(first, last) if frame not in indices]
                    if not neighbors:
                        raise ValueError(f"第 {index} 帧附近没有有效参考帧，修复未提交")
                    fixed_data[index] = np.mean(values[neighbors], axis=0, dtype=np.float64)
            amended = self.amend_data(fixed_data)
            parameters = dict(data.parameters or {})
            parameters.update(
                bad_frame_source_timestamp=data.timestamp,
                bad_frame_source_serial_number=data.serial_number,
                bad_frame_indices=sorted(indices),
            )
            repaired = data.update_data(**amended, parameters=parameters)
            self.bad_frames_fixed.emit(repaired)
            return repaired
        except Exception as exc:
            self.bad_frame_failed.emit(AppError(
                "坏帧修复失败", str(exc), stage="坏帧修复", original=exc,
                details=format_exception_details(exc, "坏帧修复", data),
            ))
            return None

    @pyqtSlot(np.ndarray, str, object)
    def plot_data_prepare(self, data: np.ndarray, name: str, father_obj):
        """Attach physical time units to actual temporal curve data."""
        try:
            values = np.asarray(data)
            kwargs = {"name": name, "time_unit": "unknown"}
            if values.ndim == 1:
                axis, unit = scientific_time_axis(father_obj, len(values))
                plot_data = np.column_stack((axis, values))
                kwargs["time_unit"] = unit
            elif values.ndim == 2 and values.shape[1] >= 2:
                plot_data = values
                axis, unit = scientific_time_axis(father_obj, len(values))
                source_axis = getattr(father_obj, "time_point", None)
                if source_axis is not None and np.size(source_axis) == len(values) and np.allclose(values[:, 0], axis):
                    kwargs["time_unit"] = unit
            else:
                raise ValueError("曲线数据必须是一维序列或二维 XY 数组")
            self.plot_singal.emit(plot_data, kwargs)
            return plot_data
        except Exception as exc:
            logging.error("数据无法被绘制由于：%s", exc)
            return None

    @staticmethod
    def add_time_from_fps(data: np.ndarray, sampling_rate: float,start_time: float = 0.0) -> np.ndarray:
        """基于fps给一维数组添加时间码"""
        if data.ndim != 1:
            raise ValueError("输入必须是一维数组")
        if sampling_rate <= 0:
            raise ValueError("采样频率必须大于0")

        n = len(data)
        # 计算采样间隔
        sampling_interval = 1.0 / sampling_rate
        # 生成时间戳
        timestamps = np.arange(n) * sampling_interval + start_time
        # 合并为二维数组
        result = np.column_stack((timestamps, data))

        return result

    @staticmethod
    def add_time_simple(data: np.ndarray, time_step:float, start_time: float = 0.0) -> np.ndarray:
        """基于间隔给一维数组添加时间"""
        if data.ndim != 1:
            raise ValueError("输入必须是一维数组")

        n = len(data)
        # 生成序号
        serial_numbers = np.arange(n) * time_step + start_time
        # 合并为二维数组
        result = np.column_stack((serial_numbers, data))

        return result


    @staticmethod
    def aligned_time_axis(data, length: int) -> np.ndarray:
        """Keep the public array-only interface while resolving physical scales."""
        return scientific_time_axis(data, length)[0]

    @pyqtSlot(object, np.ndarray, str, str)
    def get_fast_selection(self, data, mask, method: str, name: str):
        """Compute one ROI statistic per scientific frame with floating output."""
        try:
            source = getattr(data, "image_backup", None)
            if source is None:
                source = _primary_data_source(data)
            methods = {
                "mean": np.mean, "max": np.max, "min": np.min,
                "median": np.median, "quantile_075": lambda values: np.quantile(values, 0.75),
                "std": np.std, "sum": np.sum, "var": np.var,
            }
            if method not in methods:
                raise ValueError(f"不支持的统计方法: {method}")
            with _array_view(source) as values:
                if values.ndim != 3 or any(size == 0 for size in values.shape):
                    raise ValueError("时序统计需要非空 THW 数据")
                mask = np.asarray(mask, dtype=bool)
                if mask.shape != values.shape[1:] or not np.any(mask):
                    raise ValueError("ROI 必须与帧尺寸一致且包含有效像素")
                output = np.empty(values.shape[0], dtype=np.result_type(values.dtype, np.float64))
                for frame_index in range(values.shape[0]):
                    output[frame_index] = methods[method](values[frame_index][mask])
                time_axis, time_unit = scientific_time_axis(data, values.shape[0])
                plot_data = np.column_stack((time_axis, output))
            self.plot_singal.emit(plot_data, {"name": name, "time_unit": time_unit})
            self.plot_series_signal.emit(plot_data, name)
            return plot_data
        except Exception as exc:
            details = format_exception_details(exc, "ROI 时序统计", data)
            self.processing_error_signal.emit(AppError(
                "快速提取失败", str(exc), stage="ROI 时序统计", details=details,
            ))
            return None

    @staticmethod
    def value_distribution_from_frame(frame: np.ndarray, mask: np.ndarray, bins="auto", value_range=None):
        """计算单帧 ROI 内真实数值的频数分布。"""
        frame = np.asarray(frame)
        mask = np.asarray(mask, dtype=bool)
        if frame.ndim != 2:
            raise ValueError(f"值分布统计需要二维当前帧，实际 shape={frame.shape}")
        if mask.shape != frame.shape:
            raise ValueError(f"ROI 蒙版形状 {mask.shape} 与当前帧形状 {frame.shape} 不匹配")

        values = frame[mask]
        if values.size == 0:
            raise ValueError("ROI 内没有可统计的像素")

        value_mode = "raw"
        if np.iscomplexobj(values):
            values = np.abs(values)
            value_mode = "abs_complex"
        values = values[np.isfinite(values)]
        if values.size == 0:
            raise ValueError("ROI 内没有有限数值可统计")

        histogram_range = None
        if value_range is not None:
            if len(value_range) != 2:
                raise ValueError("统计范围必须包含最小值和最大值")
            range_min, range_max = float(value_range[0]), float(value_range[1])
            if not np.isfinite(range_min) or not np.isfinite(range_max):
                raise ValueError("统计范围必须是有限数值")
            if range_min >= range_max:
                raise ValueError("统计范围最小值必须小于最大值")
            histogram_range = (range_min, range_max)

        counts, edges = np.histogram(values, bins=bins, range=histogram_range)
        centers = (edges[:-1] + edges[1:]) / 2
        plot_data = np.column_stack((centers, counts))
        metadata = {
            "pixel_count": int(values.size),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "median": float(np.median(values)),
            "dtype": str(frame.dtype),
            "bins": int(len(counts)),
            "range": histogram_range,
            "value_mode": value_mode,
        }
        return plot_data, metadata

    @staticmethod
    def _distribution_source(data):
        if hasattr(data, "image_backup"):
            return data.image_backup
        if isinstance(data, ProcessedData):
            return data.data_processed
        if isinstance(data, Data):
            return data.data_origin
        source = getattr(data, "data_processed", None)
        if source is None:
            source = getattr(data, "data_origin", None)
        if source is None:
            raise ValueError("无法获取用于值分布统计的数据源")
        return source

    @staticmethod
    def _distribution_frame(data, frame_index: int):
        if getattr(data, "display_source", None) is not None:
            return data.display_source.get_frame(frame_index if getattr(data, "is_temporary", False) else 0)
        source = DataProcessor._distribution_source(data)
        if getattr(source, "ndim", 0) >= 3:
            frame_index = max(0, min(int(frame_index), source.shape[0] - 1))
            return source[frame_index]
        return source

    @pyqtSlot(object, np.ndarray, int, str)
    def get_value_distribution(self, data, mask, frame_index: int, name: str):
        try:
            frame = self._distribution_frame(data, frame_index)
            plot_data, metadata = self.value_distribution_from_frame(frame, mask)
            metadata.update({
                "frame_index": int(frame_index),
                "source_name": getattr(data, "source_name", getattr(data, "name", None)),
            })
            self.plot_singal.emit(plot_data, {
                "name": name,
                "analysis_mode": "hist_precomputed",
                "metadata": metadata,
            })
        except Exception as exc:
            details = format_exception_details(exc, "anchor 值分布统计", data)
            self.processing_error_signal.emit(AppError("值分布统计失败", str(exc), stage="anchor 值分布统计", details=details))


    @pyqtSlot(object, np.ndarray, int, object, object, str)
    def get_roi_value_distribution(self, data, mask, frame_index: int, bins, value_range, name: str):
        try:
            frame = self._distribution_frame(data, frame_index)
            plot_data, metadata = self.value_distribution_from_frame(frame, mask, bins=bins, value_range=value_range)
            metadata.update({
                "frame_index": int(frame_index),
                "source_name": getattr(data, "source_name", getattr(data, "name", None)),
            })
            self.plot_singal.emit(plot_data, {
                "name": name,
                "analysis_mode": "hist_precomputed",
                "metadata": metadata,
            })
        except Exception as exc:
            details = format_exception_details(exc, "选区值分布统计", data)
            self.processing_error_signal.emit(AppError("选区值分布统计失败", str(exc), stage="选区值分布统计", details=details))


class MassDataProcessor(QObject):
    """大型数据（EM-iSCAT）处理的线程解决"""
    processing_progress_signal = pyqtSignal(int, int) # 进度槽
    processed_result = pyqtSignal(object)
    processing_error_signal = pyqtSignal(object)
    processing_cancelled_signal = pyqtSignal()
    calculator_completed = pyqtSignal(object)
    calculator_failed = pyqtSignal(object)
    compute_plan_signal = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        logging.info("大数据处理线程已载入")
        self.abortion = False
        self.cancellation_token = CancellationToken()
        self.task_id = None
        self._task_contexts = TaskContextQueue()

    def set_cancellation_token(self, token):
        self.cancellation_token = token or CancellationToken()

    def set_task_context(self, task_id, token=None):
        self.task_id = str(task_id or "") or None
        if token is not None:
            self.cancellation_token = token

    def enqueue_task_context(self, task_id, token=None):
        return self._task_contexts.enqueue(task_id, token)

    def _activate_task_context(self):
        context = self._task_contexts.next()
        if context is None:
            return True
        self.task_id = context.task_id or None
        self.cancellation_token = context.token
        self.abortion = False
        if context.token.is_cancelled:
            self.processing_cancelled_signal.emit()
            return False
        return True

    def _emit_failure(self, title, stage, exc, data=None):
        details = format_exception_details(exc, stage, data)
        self.processing_error_signal.emit(AppError(
            title,
            str(exc),
            stage=stage,
            severity="error",
            details=details,
            original=exc,
            context={"data": data} if data is not None else {},
            task_id=self.task_id,
        ))
        return False

    def _emit_cancelled(self):
        self.processing_cancelled_signal.emit()
        return False
    @pyqtSlot(object, int, bool, object)
    @task_scoped
    def pre_process(self, data, bg_num=360, unfold=True, compute_options=None):
        """Apply exact median-background normalization with bounded spatial blocks."""
        try:
            logging.info("开始有界预处理...")
            options = dict(compute_options or {})
            values = object.__getattribute__(data, "__dict__")
            storage_name = (
                "_data_origin_storage" if isinstance(data, Data)
                else "_data_processed_storage"
            )
            source = values.get(storage_name)
            if source is None:
                source = data.data_origin if isinstance(data, Data) else data.data_processed
            shape = tuple(int(value) for value in source.shape)
            if len(shape) != 3:
                raise ValueError(f"EM 预处理要求 THW 三维数据，实际 shape={shape}")
            input_dtype = np.dtype(source.dtype)
            backend = BackendPreference(options.get("backend", "auto"))
            precision = PrecisionPolicy(options.get("precision", "compatibility"))
            compute_precision = resolve_precision("em_preprocess", input_dtype, precision)
            compute_itemsize = np.dtype(compute_precision.compute_dtype).itemsize
            output_itemsize = np.dtype(compute_precision.output_dtype).itemsize
            background_count = min(max(1, int(bg_num)), shape[0])
            workspace_per_pixel = (
                shape[0] * (input_dtype.itemsize + compute_itemsize + output_itemsize)
                + background_count * compute_itemsize
            )
            request = ComputeRequest(
                task_id=self.task_id or f"preprocess-{uuid.uuid4().hex}",
                attempt_id=0,
                algorithm="em_preprocess",
                data_id=str(getattr(data, "timestamp", getattr(data, "name", "data"))),
                shape=shape,
                dtype=input_dtype,
                axes="THW",
                source=source,
                parameters={
                    "output_shape": shape,
                    "workspace_bytes_per_spatial_item": workspace_per_pixel,
                    "background_frames": background_count,
                    "max_spatial_items": 4096,
                },
                backend=backend,
                precision=precision,
            )
            cache_dir = Path(
                options.get("cache_directory") or get_array_store().config.cache_dir
            )
            cache_dir.mkdir(parents=True, exist_ok=True)
            budget = ResourceBudget(
                host_limit_bytes=max(
                    256, int(options.get("host_memory_limit_mb", 4096))
                ) * 1024 ** 2,
                disk_free_bytes=int(shutil.disk_usage(cache_dir).free),
                cpu_workers=max(1, int(options.get("cpu_workers", 1))),
            )
            plan = plan_compute(
                request,
                budget,
                capabilities=tuple(options.get("capabilities", ())),
                default_backend=BackendPreference.AUTO,
                allow_cpu_fallback=bool(options.get("allow_cpu_fallback", True)),
                disk_output_threshold_bytes=max(
                    1, int(options.get("cache_threshold_mb", 512))
                ) * 1024 ** 2,
            )
            self.compute_plan_signal.emit(plan)

            def report_progress(current, total, message):
                if self.abortion:
                    self.cancellation_token.cancel()
                self.processing_progress_signal.emit(int(current), int(total))

            result = run_preprocess_pipeline(
                plan,
                background_frames=background_count,
                cache_dir=cache_dir,
                token=self.cancellation_token,
                progress=report_progress,
            )
            parameters = dict(
                data.parameters if isinstance(data, Data) else data.out_processed
            )
            parameters.setdefault(
                "source_format_import", getattr(data, "format_import", "")
            )
            metadata = execution_metadata(
                result.plan,
                execution="bounded_cpu",
                background_frames=background_count,
                background_reduction="exact_median",
                unfolded_data="on_demand_view",
            )
            processed = ProcessedData(
                data.timestamp,
                f"{data.name}@EM_pre",
                "EM_pre_processed",
                time_point=data.time_point,
                data_processed=result.output,
                out_processed={
                    **parameters,
                    "bg_frame": result.background,
                    "compute": metadata,
                },
            )
            self.processed_result.emit(processed)
            return True
        except TaskCancelled:
            return self._emit_cancelled()
        except Exception as exc:
            return self._emit_failure("数据处理失败", "EM_pre_processed", exc, data)
    @pyqtSlot(object,float,int,int,int,int,int,str)
    @task_scoped
    def quality_stft(self,data,target_freq: float,scale_range:int,fps:int, window_size: int, noverlap: int,
                    custom_nfft: int, window_type: str):
        """STFT质量分析"""
        mean_signal = get_mean_time_trace(data, self.cancellation_token)

        # 窗函数的选择和生成
        window = self.get_window(window_type, window_size)

        f, t, Zxx = signal.stft(
            mean_signal,
            fs=fps,
            window=window,
            nperseg=window_size,
            noverlap=noverlap,
            nfft=custom_nfft,
            return_onesided=True,
            scaling='psd'
        )
        # 提取范围内所有对应的索引
        if scale_range > 0:
            low_bound = max(0.0, target_freq - scale_range / 2.0)
            high_bound = min(f[-1], target_freq + scale_range / 2.0)
            target_idx = np.where((f >= low_bound) & (f <= high_bound))[0]
            if len(target_idx) == 0:
                target_idx = [np.argmin(np.abs(f - target_freq))]
        else:
            target_idx = [np.argmin(np.abs(f - target_freq))]
        self.processed_result.emit(ProcessedData(data.timestamp,
                                                             f'{data.name}@stft_q',
                                                             'stft_quality',
                                                             data_processed=Zxx,
                                                            time_point=t,
                                                             out_processed={
                                                                 'window_type': window,
                                                                 'window_size': window_size,
                                                                 'out_length': Zxx.shape[1],
                                                                 'frequencies':f,
                                                                 'time_series':t,
                                                                 'target_freq':target_freq,
                                                                 'scale_range':scale_range,
                                                                 'target_idx':target_idx,
                                                             })
                                   )
        return True
        # except Exception as e:
        #     self.processed_result.emit({'type': "stft_quality", 'error': str(e)})
        #     return False

    @pyqtSlot(object, object, int, int, int, int, int, str, bool, int, int, object)
    @task_scoped
    def python_stft(self, data, target_freq, scale_range: int, fps: int, window_size: int,
                    noverlap: int, custom_nfft: int, window_type: str, is_multipro: bool,
                    batch_size: int, cpu_num: int, compute_options=None):
        """Execute STFT through the bounded CPU/CUDA pipeline."""
        try:
            options = dict(compute_options or {})
            values = object.__getattribute__(data, "__dict__")
            storage_name = (
                "_data_processed_storage" if isinstance(data, ProcessedData)
                else "_data_origin_storage"
            )
            source = values.get(storage_name)
            if source is None:
                source = data.data_processed if isinstance(data, ProcessedData) else data.data_origin
            shape = tuple(int(value) for value in source.shape)
            if len(shape) != 3:
                raise ValueError(f"STFT 期望 THW 三维数据，实际 shape={shape}")
            input_dtype = np.dtype(source.dtype)
            if int(fps) <= 0:
                raise ValueError("STFT 采样帧率必须大于 0")
            if int(window_size) < 1:
                raise ValueError("STFT 窗口大小必须大于 0")
            if not 0 <= int(noverlap) < int(window_size):
                raise ValueError("STFT 窗口重叠必须满足 0 <= noverlap < window_size")

            window = self.get_window(window_type, int(window_size))
            if window is None:
                raise ValueError(f"无法创建 STFT 窗函数: {window_type}")
            nfft = max(int(custom_nfft), int(window_size))
            reference_dtype = input_dtype if input_dtype.kind in "fc" else np.dtype("float32")
            reference = np.zeros(shape[0], dtype=reference_dtype)
            frequencies, time_series, magnitude, target_idx = stft_frequency_trace(
                reference,
                fs=int(fps),
                window=window,
                nperseg=int(window_size),
                noverlap=int(noverlap),
                nfft=nfft,
                target_freq=target_freq,
                scale_range=scale_range,
            )
            output_shape = (int(magnitude.shape[-1]), shape[1], shape[2])

            backend = BackendPreference(options.get("backend", BackendPreference.CPU.value))
            precision_policy = PrecisionPolicy(
                options.get("precision", PrecisionPolicy.COMPATIBILITY.value)
            )
            precision = resolve_precision("stft", input_dtype, precision_policy)
            frequency_bins = nfft if input_dtype.kind == "c" else nfft // 2 + 1
            compute_itemsize = np.dtype(precision.compute_dtype).itemsize
            complex_itemsize = (
                compute_itemsize if np.dtype(precision.compute_dtype).kind == "c"
                else compute_itemsize * 2
            )
            output_itemsize = np.dtype(precision.output_dtype).itemsize
            workspace_per_pixel = (
                shape[0] * compute_itemsize
                + 2 * frequency_bins * output_shape[0] * complex_itemsize
                + output_shape[0] * output_itemsize
            )

            task_id = self.task_id or f"stft-{uuid.uuid4().hex}"
            request = ComputeRequest(
                task_id=task_id,
                attempt_id=0,
                algorithm="stft",
                data_id=str(getattr(data, "timestamp", getattr(data, "name", "data"))),
                shape=shape,
                dtype=input_dtype,
                axes="THW",
                source=source,
                parameters={
                    "output_shape": output_shape,
                    "workspace_bytes_per_spatial_item": workspace_per_pixel,
                    "fps": int(fps),
                    "window_size": int(window_size),
                    "noverlap": int(noverlap),
                    "nfft": nfft,
                    "target_freq": float(target_freq),
                    "scale_range": float(scale_range),
                    "max_spatial_items": 512,
                },
                backend=backend,
                precision=precision_policy,
            )

            preferred_device = str(options.get("preferred_device", "") or "")
            try:
                device_index = int(preferred_device.rsplit(":", 1)[-1]) if preferred_device else 0
            except ValueError:
                device_index = 0
            capabilities = list(options.get("capabilities", ()))
            capability = next(
                (
                    device for device in capabilities
                    if device.kind == "gpu"
                    and device.status is CapabilityStatus.AVAILABLE
                    and "stft" in device.supported_algorithms
                    and (not preferred_device or device.device_id == preferred_device)
                ),
                None,
            )
            if backend is BackendPreference.GPU and capability is None:
                capability = probe_cuda_capability_isolated(device_index, timeout=45.0)
                update_cached_device_capability(capability)
                capabilities = [
                    device for device in capabilities
                    if device.device_id != capability.device_id
                ]
                capabilities.append(capability)
                if capability.status is not CapabilityStatus.AVAILABLE:
                    logging.warning("CUDA STFT 自检未通过: %s", capability.detail)

            host_limit = max(256, int(options.get("host_memory_limit_mb", 4096))) * 1024 ** 2
            device_limit = 0
            if capability is not None and capability.status is CapabilityStatus.AVAILABLE:
                percent = min(90, max(10, int(options.get("gpu_memory_percent", 70))))
                reserve = max(1024 ** 3, int(capability.total_memory_bytes * 0.1))
                available = max(0, capability.free_memory_bytes - reserve)
                device_limit = min(
                    int(capability.total_memory_bytes * percent / 100),
                    available,
                )

            cache_dir = Path(
                options.get("cache_directory") or get_array_store().config.cache_dir
            )
            cache_dir.mkdir(parents=True, exist_ok=True)
            budget = ResourceBudget(
                host_limit_bytes=host_limit,
                device_limit_bytes=device_limit,
                disk_free_bytes=int(shutil.disk_usage(cache_dir).free),
                cpu_workers=max(1, int(options.get("cpu_workers", cpu_num or 1))),
            )
            allow_fallback = bool(options.get("allow_cpu_fallback", True))
            plan = plan_compute(
                request,
                budget,
                capabilities=tuple(capabilities),
                default_backend=BackendPreference.AUTO,
                allow_cpu_fallback=allow_fallback,
                disk_output_threshold_bytes=max(
                    1, int(options.get("cache_threshold_mb", 512))
                ) * 1024 ** 2,
            )
            self.compute_plan_signal.emit(plan)

            params = {
                "fps": int(fps),
                "window": window,
                "window_size": int(window_size),
                "noverlap": int(noverlap),
                "nfft": nfft,
                "target_freq": target_freq,
                "scale_range": scale_range,
            }

            def report_progress(current, total, message):
                if self.abortion:
                    self.cancellation_token.cancel()
                self.processing_progress_signal.emit(int(current), int(total))

            result = run_stft_pipeline(
                plan,
                params,
                cache_dir=cache_dir,
                token=self.cancellation_token,
                progress=report_progress,
                allow_cpu_fallback=allow_fallback,
                device_index=device_index,
            )
            if result.plan.actual_backend == "gpu":
                verified = mark_algorithm_verified(
                    f"nvidia:{device_index}", "stft"
                )
                logging.info(
                    "GPU 算法实际任务验证通过: algorithm=stft device=%s backend=%s",
                    verified.name if verified is not None else f"nvidia:{device_index}",
                    verified.backend if verified is not None else "CuPy/CUDA",
                )
            if result.plan is not plan:
                self.compute_plan_signal.emit(result.plan)
            metadata = execution_metadata(
                result.plan,
                execution=f"bounded_{result.plan.actual_backend}",
                device=capability.name if capability is not None else "CPU",
                fallback_reason=result.fallback_reason or (
                    result.plan.backend_reason
                    if backend is BackendPreference.GPU
                    and result.plan.actual_backend == "cpu"
                    else ""
                ),
                frequency_selection="contract_v1",
                max_gpu_oom_retries=3,
            )
            inherited = {
                key: value for key, value in (getattr(data, "out_processed", {}) or {}).items()
                if key != "unfolded_data"
            }
            inherited.update(getattr(data, "parameters", {}) or {})
            processed = ProcessedData(
                data.timestamp,
                f"{data.name}@r_stft",
                "ROI_stft",
                time_point=result.times,
                data_processed=result.output,
                out_processed={
                    "whole_mean": result.whole_mean,
                    "window_type": window,
                    "window_size": int(window_size),
                    "window_step": int(window_size) - int(noverlap),
                    "target_freq": target_freq,
                    "scale_range": scale_range,
                    "FFT_length": nfft,
                    "frequencies": result.frequencies,
                    "target_idx": result.selected_indices,
                    **inherited,
                    "compute": metadata,
                },
            )
            self.processed_result.emit(processed)
            return True
        except TaskCancelled:
            return self._emit_cancelled()
        except Exception as exc:
            return self._emit_failure("数据处理失败", "ROI_stft", exc, data)
    def get_window(self,window_type, window_size):
        try:
            if window_type == 'gaussian':
                window = signal.get_window((window_type, window_size / 6), window_size, fftbins=False)
            elif window_type == 'general_gaussian':
                window = signal.get_window((window_type, 1.5, window_size / 6), window_size, fftbins=False)
            else:
                window = signal.get_window(window_type, window_size, fftbins=False)
            # # 计算窗口能量
            # win_energy = np.sum(window ** 2)
            #
            # # 对窗口进行能量归一化
            # normalized_window = window / np.sqrt(win_energy)

            return window
        except Exception as e:
            logging.error(f'Window Fault:{e}')

    @pyqtSlot(object, float, int, int, int, str, object)
    @task_scoped
    def quality_cwt(
        self,
        data,
        target_freq: float,
        scale_range: int,
        fps: int,
        totalscales: int,
        wavelet: str = "morl",
        compute_options=None,
    ):
        """
        CWT(连续小波变换)分析信号评估
        参数:
            target_freq: 目标分析频率(Hz)
            EM_fps: 采样频率
            scales: 尺度数组，控制小波变换的频率分辨率
            wavelet: 使用的小波类型(默认为'morl'墨西哥帽小波)
        """
        try:
            options = dict(compute_options or {})
            if int(totalscales) < 2:
                raise ValueError("CWT 质量分析至少需要 2 个尺度")
            mean_signal = get_mean_time_trace(data, self.cancellation_token)
            cparam = 2 * pywt.central_frequency(wavelet) * totalscales
            scales = cparam / np.arange(totalscales, 1, -1)
            self.processing_progress_signal.emit(20, 100)
            total_frames = int(data.timelength)
            requested_backend = BackendPreference(
                options.get("backend", "auto")
            )
            effective_backend = (
                BackendPreference.CPU
                if requested_backend is BackendPreference.AUTO
                else requested_backend
            )
            precision = PrecisionPolicy(
                options.get("precision", "compatibility")
            )
            precision_info = resolve_precision(
                "cwt_quality", mean_signal.dtype, precision
            )
            preferred_device = str(options.get("preferred_device", "") or "")
            try:
                device_index = (
                    int(preferred_device.rsplit(":", 1)[-1])
                    if preferred_device else 0
                )
            except ValueError:
                device_index = 0
            capabilities = list(options.get("capabilities", ()))
            capability = next(
                (
                    device for device in capabilities
                    if device.kind == "gpu"
                    and device.status is CapabilityStatus.AVAILABLE
                    and "cwt_quality" in device.supported_algorithms
                ),
                None,
            )
            if effective_backend is BackendPreference.GPU and capability is None:
                capability = probe_cuda_capability_isolated(
                    device_index, timeout=45.0
                )
                update_cached_device_capability(capability)
                capabilities = [
                    device for device in capabilities
                    if device.device_id != capability.device_id
                ]
                capabilities.append(capability)

            device_limit = 0
            if capability is not None:
                device_limit = max(
                    0,
                    min(
                        int(
                            capability.total_memory_bytes
                            * int(options.get("gpu_memory_percent", 70))
                            / 100
                        ),
                        capability.free_memory_bytes,
                    ),
                )
            request = ComputeRequest(
                task_id=self.task_id or f"cwt-quality-{uuid.uuid4().hex}",
                attempt_id=0,
                algorithm="cwt_quality",
                data_id=str(getattr(data, "timestamp", data.name)),
                shape=mean_signal.shape,
                dtype=mean_signal.dtype,
                axes="T",
                source=mean_signal,
                parameters={
                    "output_shape": (len(scales), total_frames),
                    "workspace_bytes_per_spatial_item": (
                        len(scales)
                        * total_frames
                        * np.dtype(precision_info.output_dtype).itemsize
                    ),
                },
                backend=effective_backend,
                precision=precision,
            )
            plan = plan_compute(
                request,
                ResourceBudget(
                    host_limit_bytes=max(
                        256, int(options.get("host_memory_limit_mb", 4096))
                    ) * 1024 ** 2,
                    device_limit_bytes=device_limit,
                    cpu_workers=1,
                ),
                capabilities=tuple(capabilities),
                allow_cpu_fallback=bool(
                    options.get("allow_cpu_fallback", True)
                ),
            )
            self.compute_plan_signal.emit(plan)
            self.processing_progress_signal.emit(40, 100)
            worker = None
            try:
                if plan.actual_backend == "gpu":
                    worker = CudaWorkerClient(device_index)
                    spectrum, frequencies = worker.execute(
                        "cwt_quality",
                        task_id=request.task_id,
                        attempt_id=request.attempt_id,
                        block_id=0,
                        values=mean_signal,
                        scales=scales,
                        wavelet=wavelet,
                        fps=fps,
                        compute_dtype=precision_info.compute_dtype,
                        output_dtype=precision_info.output_dtype,
                        token=self.cancellation_token,
                    )
                else:
                    spectrum, frequencies = cwt_quality_spectrum(
                        mean_signal,
                        scales,
                        wavelet,
                        fps=fps,
                        compute_dtype=precision_info.compute_dtype,
                        output_dtype=precision_info.output_dtype,
                    )
            except CudaBackendError:
                if not options.get("allow_cpu_fallback", True):
                    raise
                plan = replace(
                    plan,
                    actual_backend="cpu",
                    backend_reason="CWT 质量分析 GPU 失败，已回退 CPU",
                    peak_device_bytes=0,
                )
                self.compute_plan_signal.emit(plan)
                spectrum, frequencies = cwt_quality_spectrum(
                    mean_signal,
                    scales,
                    wavelet,
                    fps=fps,
                    compute_dtype=precision_info.compute_dtype,
                    output_dtype=precision_info.output_dtype,
                )
            finally:
                if worker is not None:
                    worker.close()
            self.processing_progress_signal.emit(70, 100)
            if plan.actual_backend == "gpu":
                mark_algorithm_verified(f"nvidia:{device_index}", "cwt")
            self.processed_result.emit(ProcessedData(
                data.timestamp,
                f"{data.name}@cwt_q",
                "cwt_quality",
                time_point=np.arange(total_frames) / fps,
                data_processed=spectrum,
                out_processed={
                    "frequencies": frequencies,
                    "time_series": np.arange(total_frames) / fps,
                    "target_freq": target_freq,
                    "scale_range": scale_range,
                    "total_scales": totalscales,
                    "wavelet_name": wavelet,
                    "compute": execution_metadata(
                        plan,
                        execution=f"quality_{plan.actual_backend}",
                        requested_quality_backend=requested_backend.value,
                    ),
                },
            ))
            self.processing_progress_signal.emit(100, 100)
            return True
        except Exception as e:
            return self._emit_failure("数据处理失败", "cwt_quality", e, data)

    @pyqtSlot(object, float, int, int, str, float, object)
    @task_scoped
    def python_cwt(
        self,
        data,
        target_freq: float,
        fps: int,
        totalscales: int,
        wavelet: str,
        cwt_scale_range: float,
        compute_options=None,
    ):
        """Execute CWT in bounded spatial blocks with immediate scale reduction."""
        try:
            options = dict(compute_options or {})
            values = object.__getattribute__(data, "__dict__")
            source = values.get("_data_processed_storage")
            if source is None:
                source = data.data_processed
            shape = tuple(int(value) for value in source.shape)
            if len(shape) != 3:
                raise ValueError(f"CWT 要求 THW 三维数据，实际 shape={shape}")
            input_dtype = np.dtype(source.dtype)
            backend = BackendPreference(options.get("backend", "auto"))
            precision = PrecisionPolicy(options.get("precision", "compatibility"))
            precision_info = resolve_precision("cwt", input_dtype, precision)
            complex_dtype = np.dtype(
                "complex128" if np.dtype(precision_info.compute_dtype).itemsize > 8
                else "complex64"
            )
            planned_scales, _ = cwt_axes(
                target_freq=target_freq,
                scale_range=cwt_scale_range,
                total_scales=totalscales,
                wavelet=wavelet,
                fps=fps,
            )
            kernel_bank = build_cwt_kernel_bank(
                planned_scales, wavelet, precision_info.compute_dtype
            )
            max_kernel = max(len(kernel) for kernel in kernel_bank.kernels)
            workspace_per_pixel = (
                4 * (shape[0] + max_kernel) * complex_dtype.itemsize
                + shape[0] * np.dtype(precision_info.output_dtype).itemsize
            )
            request = ComputeRequest(
                task_id=self.task_id or f"cwt-{uuid.uuid4().hex}",
                attempt_id=0,
                algorithm="cwt",
                data_id=str(getattr(data, "timestamp", getattr(data, "name", "data"))),
                shape=shape,
                dtype=input_dtype,
                axes="THW",
                source=source,
                parameters={
                    "output_shape": shape,
                    "workspace_bytes_per_spatial_item": workspace_per_pixel,
                    "target_freq": float(target_freq),
                    "scale_range": float(cwt_scale_range),
                    "total_scales": int(totalscales),
                    "wavelet": str(wavelet),
                    "fps": int(fps),
                    "max_spatial_items": 256,
                },
                backend=backend,
                precision=precision,
            )
            cache_dir = Path(
                options.get("cache_directory") or get_array_store().config.cache_dir
            )
            cache_dir.mkdir(parents=True, exist_ok=True)
            preferred_device = str(options.get("preferred_device", "") or "")
            try:
                device_index = (
                    int(preferred_device.rsplit(":", 1)[-1])
                    if preferred_device else 0
                )
            except ValueError:
                device_index = 0
            capabilities = list(options.get("capabilities", ()))
            capability = next(
                (
                    device for device in capabilities
                    if device.kind == "gpu"
                    and device.status is CapabilityStatus.AVAILABLE
                    and "cwt" in device.supported_algorithms
                    and (
                        not preferred_device
                        or device.device_id == preferred_device
                    )
                ),
                None,
            )
            if backend is BackendPreference.GPU and capability is None:
                capability = probe_cuda_capability_isolated(
                    device_index, timeout=45.0
                )
                update_cached_device_capability(capability)
                capabilities = [
                    device for device in capabilities
                    if device.device_id != capability.device_id
                ]
                capabilities.append(capability)
                if capability.status is not CapabilityStatus.AVAILABLE:
                    logging.warning("CUDA CWT 自检未通过: %s", capability.detail)

            device_limit = 0
            if (
                capability is not None
                and capability.status is CapabilityStatus.AVAILABLE
            ):
                percent = min(
                    90, max(10, int(options.get("gpu_memory_percent", 70)))
                )
                reserve = max(
                    1024 ** 3, int(capability.total_memory_bytes * 0.1)
                )
                available = max(
                    0, capability.free_memory_bytes - reserve
                )
                device_limit = min(
                    int(capability.total_memory_bytes * percent / 100),
                    available,
                )
            budget = ResourceBudget(
                host_limit_bytes=max(
                    256, int(options.get("host_memory_limit_mb", 4096))
                ) * 1024 ** 2,
                device_limit_bytes=device_limit,
                disk_free_bytes=int(shutil.disk_usage(cache_dir).free),
                cpu_workers=max(1, int(options.get("cpu_workers", 1))),
            )
            allow_fallback = bool(options.get("allow_cpu_fallback", True))
            plan = plan_compute(
                request,
                budget,
                capabilities=tuple(capabilities),
                default_backend=BackendPreference.AUTO,
                allow_cpu_fallback=allow_fallback,
                disk_output_threshold_bytes=max(
                    1, int(options.get("cache_threshold_mb", 512))
                ) * 1024 ** 2,
            )
            self.compute_plan_signal.emit(plan)

            params = {
                "target_freq": float(target_freq),
                "scale_range": float(cwt_scale_range),
                "total_scales": int(totalscales),
                "wavelet": str(wavelet),
                "fps": int(fps),
            }

            def report_progress(current, total, message):
                if self.abortion:
                    self.cancellation_token.cancel()
                self.processing_progress_signal.emit(int(current), int(total))

            result = run_cwt_pipeline(
                plan,
                params,
                cache_dir=cache_dir,
                token=self.cancellation_token,
                progress=report_progress,
                allow_cpu_fallback=allow_fallback,
                device_index=device_index,
            )
            if result.plan.actual_backend == "gpu":
                verified = mark_algorithm_verified(
                    f"nvidia:{device_index}", "cwt"
                )
                logging.info(
                    "GPU 算法实际任务验证通过: algorithm=cwt "
                    "device=%s backend=%s wavelet=%s scales=%s",
                    (
                        verified.name
                        if verified is not None
                        else f"nvidia:{device_index}"
                    ),
                    (
                        verified.backend
                        if verified is not None
                        else "CuPy/CUDA"
                    ),
                    wavelet,
                    totalscales,
                )
            if result.plan is not plan:
                self.compute_plan_signal.emit(result.plan)
            inherited = {
                key: value for key, value in (getattr(data, "out_processed", {}) or {}).items()
                if key != "unfolded_data"
            }
            inherited.update(getattr(data, "parameters", {}) or {})
            metadata = execution_metadata(
                result.plan,
                execution=f"bounded_{result.plan.actual_backend}",
                scale_reduction="normalized_mean",
                fallback_reason=result.fallback_reason or (
                    result.plan.backend_reason
                    if backend is BackendPreference.GPU
                    and result.plan.actual_backend == "cpu"
                    else ""
                ),
                max_gpu_oom_retries=3,
            )
            processed = ProcessedData(
                data.timestamp,
                f"{data.name}@cwt",
                "ROI_cwt",
                time_point=np.arange(shape[0]) / float(fps),
                data_processed=result.output,
                out_processed={
                    "whole_mean": result.whole_mean,
                    "total_scales": int(totalscales),
                    "wavelet_name": str(wavelet),
                    "scale_range": float(cwt_scale_range),
                    "target_freq": float(target_freq),
                    "scales": result.scales,
                    "frequencies": result.frequencies,
                    **inherited,
                    "compute": metadata,
                },
            )
            self.processed_result.emit(processed)
            return True
        except TaskCancelled:
            return self._emit_cancelled()
        except Exception as exc:
            return self._emit_failure("数据处理失败", "ROI_cwt", exc, data)
    @pyqtSlot(object)
    @task_scoped
    def accumulate_amplitude(self, data):
        """Return the temporal mean of a scientific THW sequence."""
        try:
            self.cancellation_token.raise_if_cancelled()
            with _array_view(_primary_data_source(data)) as values:
                if values.ndim != 3 or any(size == 0 for size in values.shape):
                    raise ValueError("时间平均振幅图需要非空 THW 序列")
                dtype = np.result_type(values.dtype, np.float64)
                _ensure_working_memory(data, array_nbytes(values.shape[1:], dtype) * 2, "时间平均振幅")
                result = np.zeros(values.shape[1:], dtype=dtype)
                for frame in values:
                    self.cancellation_token.raise_if_cancelled()
                    result += frame / values.shape[0]
            metadata = _analysis_metadata(data)
            metadata.update(scientific_axes="HW", source_axes="HW", reduction="mean_over_time")
            self.cancellation_token.raise_if_cancelled()
            self.processed_result.emit(ProcessedData(
                data.timestamp, f"{data.name}@atam", "Accumulated_time_amplitude_map",
                time_point=np.array([0]), data_processed=result, out_processed=metadata,
            ))
            return True
        except TaskCancelled:
            return self._emit_cancelled()
        except Exception as exc:
            return self._emit_failure("时间平均振幅计算失败", "时间平均振幅", exc, data)

    @staticmethod
    def D2GaussFunction(xy, A, x0, sigmax, y0, sigmay, b):
        """二维高斯函数
        参数:
        coords: 网格坐标 (x, y)
        A: 振幅
        x0, y0: 中心位置
        sigma_x, sigma_y: X/Y方向标准差
        offset: 背景偏移量

        返回:
        二维高斯函数值
        """
        x, y = xy[:, 0], xy[:, 1]
        return A * np.exp(-((x - x0) ** 2 / (2 * sigmax ** 2) + (y - y0) ** 2 / (2 * sigmay ** 2))) + b

    @pyqtSlot(object,int,float,bool)
    @task_scoped
    def twoD_gaussian_fit(self,data:ProcessedData|Data,zm = 2,thr = 2.5,thr_known = False):
        """
        对三维时序数据逐帧进行二维高斯拟合

        参数:
        data: numpy.ndarray, 三维数组 (T, H, W)
        zm 插值系数
        返回:
        results: list of dict, 每帧的拟合参数
        """
        try:
            timer = QElapsedTimer()
            timer.start()

            T, H, W = data.datashape
            self.processing_progress_signal.emit(0, T)

            amplitudes = np.zeros(T)
            centers_x = np.zeros(T)
            centers_y = np.zeros(T)
            mean_signal = np.zeros(T)
            max_signal = np.zeros(T)

            for m in range(T):
                frame = data.data_processed[m]
                self.processing_progress_signal.emit(m, T)
                # 图像插值
                if zm >1:
                    Z = zoom(frame, zm, order=3)
                else:
                    Z = frame

                h_z, w_z = Z.shape
                X, Y = np.meshgrid(np.arange(w_z), np.arange(h_z))
                xy = np.column_stack((X.ravel(), Y.ravel()))
                mean_signal[m] = np.mean(Z)
                max_signal[m] = np.max(Z)
                if thr_known: # 如果知道阈值
                    # 检查是否有超过阈值的点
                    if np.any(Z > thr):
                        max_value = np.max(Z)
                        y0_g, x0_g = np.unravel_index(np.argmax(Z), Z.shape)

                        # 初始参数 [A, x0, sigmax, y0, sigmay, b]
                        x0 = [max_value, x0_g, 1.0, y0_g, 1.0, np.mean(Z)]

                        # 参数边界
                        lb = [0, 0, 0.1, 0, 0.1, 0]
                        ub = [100, w_z, (w_z/2)**2, h_z, (h_z/2)**2, max_value]

                        try:
                            # 二维高斯拟合
                            popt, _ = curve_fit(self.D2GaussFunction,
                                                xy,
                                                Z.ravel(),
                                                p0=x0,
                                                bounds=(lb, ub),
                                                maxfev=5000)

                            amplitudes[m] = popt[0]
                            centers_x[m] = popt[1]
                            centers_y[m] = popt[3]
                        except RuntimeError:
                            amplitudes[m] = np.mean(Z)
                            centers_x[m] = np.nan
                            centers_y[m] = np.nan
                    else:
                        amplitudes[m] = np.mean(Z)
                        centers_x[m] = np.nan
                        centers_y[m] = np.nan
                else:
                    max_value = np.max(Z)
                    y0_g, x0_g = np.unravel_index(np.argmax(Z), Z.shape)

                    # 初始参数 [A, x0, sigmax, y0, sigmay, b]
                    x0 = [max_value, x0_g, 1.0, y0_g, 1.0, np.mean(Z)]

                    # 参数边界
                    lb = [0, 0, 0.1, 0, 0.1, 0]
                    ub = [100, w_z, (w_z / 2) ** 2, h_z, (h_z / 2) ** 2, max_value]
                    mean_signal[m] = np.mean(Z)
                    try:
                        # 二维高斯拟合
                        popt, _ = curve_fit(self.D2GaussFunction,
                                            xy,
                                            Z.ravel(),
                                            p0=x0,
                                            bounds=(lb, ub),
                                            maxfev=5000)

                        amplitudes[m] = popt[0]
                        centers_x[m] = popt[1]
                        centers_y[m] = popt[3]
                    except RuntimeError:
                        amplitudes[m] = np.mean(Z)
                        centers_x[m] = np.nan
                        centers_y[m] = np.nan
            self.processing_progress_signal.emit(T, T)
            self.processed_result.emit(ProcessedData(data.timestamp,
                                             f'{data.name}@scs',
                                             "Single_channel_signal",
                                             time_point=data.time_point,
                                             data_processed=copy.deepcopy(amplitudes),
                                             out_processed={
                                                 'thr_known': thr_known,
                                                 'thr': thr,
                                                 'mean_signal':mean_signal,
                                                 'amplitudes':amplitudes,
                                                 'max_signal':max_signal,**data.out_processed
                                             }))
            return True
        except Exception as e:
            return self._emit_failure("数据处理失败", "Single_channel_signal", e, data)

    @pyqtSlot(object, int, float, bool)
    @task_scoped
    def simple_single_channel(self, data: ProcessedData|Data, zm=2, thr=2.5, thr_known=False):
        """简单的单通道信号处理办法"""
        try:
            timer = QElapsedTimer()
            timer.start()

            T, H, W = data.datashape
            self.processing_progress_signal.emit(0, T)

            amplitudes = np.zeros(T)
            mean_signal = np.zeros(T)
            max_signal = np.zeros(T)
            min_signal = np.zeros(T)

            for m in range(T):
                frame = data.data_processed[m]
                self.processing_progress_signal.emit(m, T)
                # 图像插值
                if zm > 1:
                    Z = zoom(frame, zm, order=3)
                else:
                    Z = frame

                h_z, w_z = Z.shape
                mean_signal[m] = np.mean(Z)
                max_signal[m] = np.max(Z)
                min_signal[m] = np.min(Z)
                if thr_known:  # 如果知道阈值
                    # 检查是否有超过阈值的点
                    if np.any(Z > thr):
                        amplitudes[m] = np.max(Z)
                    else:
                        amplitudes[m] = np.mean(Z)
                else:
                    amplitudes[m] = np.max(Z)
            self.processing_progress_signal.emit(T, T)
            self.processed_result.emit(ProcessedData(data.timestamp,
                                                                 f'{data.name}@scs',
                                                                 "Single_channel_signal",
                                                                 time_point=data.time_point,
                                                                 data_processed=copy.deepcopy(amplitudes),
                                                                 out_processed={
                                                                     'thr_known': thr_known,
                                                                     'thr': thr,
                                                                     'mean_signal': mean_signal,
                                                                     'max_signal': max_signal,
                                                                     'min_signal': min_signal,
                                                                     **{k: data.out_processed.get(k)
                                                                        for k in {'roi_shape','whole_mean'} if k in data.out_processed}
                                                                 }))
            return True
        except Exception as e:
            return self._emit_failure("数据处理失败", "简单 Single_channel_signal", e, data)

    @pyqtSlot(object)
    @task_scoped
    def twoD_fourier_transform(self, data):
        """Transform spatial axes while preserving HW, 1HW and THW shapes."""
        try:
            self.cancellation_token.raise_if_cancelled()
            with _array_view(_primary_data_source(data)) as values:
                frames = _frame_stack(values)
                complex_dtype = np.result_type(values.dtype, np.complex64)
                real_dtype = np.empty((), dtype=complex_dtype).real.dtype
                output_bytes = values.size * (np.dtype(complex_dtype).itemsize + 3 * real_dtype.itemsize)
                _ensure_working_memory(data, output_bytes * 2, "二维傅里叶变换")
                spectrum = np.empty(values.shape, dtype=complex_dtype)
                output_frames = spectrum[np.newaxis, ...] if values.ndim == 2 else spectrum
                self.processing_progress_signal.emit(0, len(frames))
                for index, frame in enumerate(frames):
                    self.cancellation_token.raise_if_cancelled()
                    output_frames[index] = np.fft.fftshift(np.fft.fft2(frame))
                    self.processing_progress_signal.emit(index + 1, len(frames))
                magnitude = np.abs(spectrum)
                phase = np.angle(spectrum)
                logarithmic = np.log1p(magnitude)
                axes = "HW" if values.ndim == 2 else "THW"
                time_point, time_unit = scientific_time_axis(data, len(frames))
            metadata = _analysis_metadata(data)
            metadata.update(
                twoD_FFT=spectrum, magnitude_spectra=magnitude, phase_spectra=phase,
                fft_shifted=True, scientific_axes=axes, source_axes=axes,
                time_axis_kind="physical", time_axis_unit=time_unit, time_unit=time_unit,
            )
            self.cancellation_token.raise_if_cancelled()
            self.processed_result.emit(ProcessedData(
                data.timestamp, f"{data.name}@2DFT", "2D_Fourier_transform",
                time_point=time_point, data_processed=logarithmic, out_processed=metadata,
            ))
            return True
        except TaskCancelled:
            return self._emit_cancelled()
        except Exception as exc:
            return self._emit_failure("二维傅里叶变换失败", "2D_Fourier_transform", exc, data)

    @pyqtSlot(object)
    @task_scoped
    def twoD_inverse_fourier_transform(self, data):
        """Reconstruct real images from complete, centered complex spectra."""
        try:
            metadata = _analysis_metadata(data)
            source = metadata.get("twoD_FFT")
            if source is None:
                source = _primary_data_source(data)
            self.cancellation_token.raise_if_cancelled()
            with _array_view(source) as values:
                frames = _frame_stack(values)
                if not np.iscomplexobj(values):
                    raise ValueError("逆傅里叶变换需要包含幅度和相位的完整复频谱")
                dtype = np.empty((), dtype=values.dtype).real.dtype
                _ensure_working_memory(data, values.size * dtype.itemsize * 2, "二维傅里叶逆变换")
                reconstructed = np.empty(values.shape, dtype=dtype)
                output_frames = reconstructed[np.newaxis, ...] if values.ndim == 2 else reconstructed
                self.processing_progress_signal.emit(0, len(frames))
                for index, frame in enumerate(frames):
                    self.cancellation_token.raise_if_cancelled()
                    output_frames[index] = np.fft.ifft2(np.fft.ifftshift(frame)).real
                    self.processing_progress_signal.emit(index + 1, len(frames))
                axes = "HW" if values.ndim == 2 else "THW"
                time_point, time_unit = scientific_time_axis(data, len(frames))
            metadata.update(
                reconstructed_img=reconstructed, scientific_axes=axes, source_axes=axes,
                time_axis_kind="physical", time_axis_unit=time_unit, time_unit=time_unit,
            )
            self.cancellation_token.raise_if_cancelled()
            self.processed_result.emit(ProcessedData(
                data.timestamp, f"{data.name}@2DIFFT", "2D_Inverse_Fourier_transform",
                time_point=time_point, data_processed=reconstructed, out_processed=metadata,
            ))
            return True
        except TaskCancelled:
            return self._emit_cancelled()
        except Exception as exc:
            return self._emit_failure("二维傅里叶逆变换失败", "2D_Inverse_Fourier_transform", exc, data)

    @pyqtSlot(object, int, int, list, str, str, float)
    @task_scoped
    def heartbeat_movement(self, data, step, base_num, after_series, save_path="", export_mode="video", scale=1):
        """Calculate bounded-budget Farneback flow with atomic result publication."""
        try:
            metadata = _analysis_metadata(data)
            fps = float(metadata.get("fps", 0))
            space_step = float(metadata.get("space_step", 0))
            if not np.isfinite(fps) or fps <= 0 or not np.isfinite(space_step) or space_step <= 0:
                raise ValueError("光流分析需要有限且大于零的 FPS 和空间步长")
            if isinstance(step, bool) or int(step) != step or step < 1:
                raise ValueError("运动矢量采样步长必须为正整数")
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError("运动箭头缩放必须为有限正数")
            self.cancellation_token.raise_if_cancelled()
            with _array_view(_primary_data_source(data)) as values:
                if values.ndim != 3 or any(size == 0 for size in values.shape) or values.dtype.kind not in "biuf":
                    raise ValueError("光流分析需要非空实数灰度 THW 数据")
                series = []
                for index in after_series:
                    if isinstance(index, bool) or int(index) != index or not 0 <= int(index) < len(values):
                        raise ValueError(f"帧编号超出有效范围: {index}")
                    series.append(int(index))
                if not series or any(right <= left for left, right in zip(series, series[1:])):
                    raise ValueError("分析帧列表应为非空、递增且不重复的帧编号")
                if base_num == -1:
                    if len(series) < 2:
                        raise ValueError("相邻帧光流至少需要两个时刻")
                    pairs = list(zip(series[:-1], series[1:]))
                    base_indices = np.asarray(series[:-1], dtype=int)
                    offsets = np.asarray(series[1:]) - series[0]
                    process_name = "连续分析"
                else:
                    if isinstance(base_num, bool) or int(base_num) != base_num or not 0 <= int(base_num) < len(values):
                        raise ValueError("基准帧编号超出有效范围")
                    pairs = [(int(base_num), index) for index in series]
                    base_indices = int(base_num)
                    offsets = np.asarray(series) - int(base_num)
                    process_name = "比较模式"
                count = len(pairs)
                height, width = values.shape[1:]
                pixels = height * width
                output_bytes = count * pixels * np.dtype(np.float32).itemsize * 4
                base_bytes = pixels * values.dtype.itemsize * (count if base_num == -1 else 1)
                _ensure_working_memory(data, 2 * (output_bytes + base_bytes) + pixels * 64, "光流分析")
                flow_stack = np.empty((count, height, width, 2), dtype=np.float32)
                magnitudes = np.empty((count, height, width), dtype=np.float32)
                angles = np.empty((count, height, width), dtype=np.float32)
                max_speed = np.empty(count, dtype=np.float64)
                mean_speed = np.empty(count, dtype=np.float64)
                base_data = (
                    np.empty((count, height, width), dtype=values.dtype)
                    if base_num == -1 else np.array(values[int(base_num)], copy=True)
                )
                cache = {}
                self.processing_progress_signal.emit(0, count)
                for index, (previous, current) in enumerate(pairs):
                    self.cancellation_token.raise_if_cancelled()
                    for frame_index in (previous, current):
                        if frame_index not in cache:
                            cache[frame_index] = self.clahe_and_blur(values[frame_index])
                    flow = cv2.calcOpticalFlowFarneback(
                        cache[previous], cache[current], None,
                        pyr_scale=0.5, levels=3, winsize=15, iterations=3,
                        poly_n=5, poly_sigma=1.2, flags=0,
                    )
                    self.cancellation_token.raise_if_cancelled()
                    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                    delta = abs(current - previous)
                    speed = magnitude * (space_step * fps / delta) if delta else np.zeros_like(magnitude)
                    flow_stack[index] = flow
                    magnitudes[index] = speed
                    angles[index] = angle
                    max_speed[index] = np.max(speed)
                    mean_speed[index] = np.mean(speed, dtype=np.float64)
                    if base_num == -1:
                        base_data[index] = values[previous]
                    keep = {int(base_num), current} if base_num != -1 else {current}
                    cache = {key: image for key, image in cache.items() if key in keep}
                    self.processing_progress_signal.emit(index + 1, count)
                self.cancellation_token.raise_if_cancelled()
            metadata.update(
                process_name=process_name, sampling_step=int(step), base_num=base_indices,
                base_data=base_data, after_series=series, magnitude_list=magnitudes,
                angle_list=angles, max_speed=max_speed, mean_speed=mean_speed,
                time_seconds=offsets / fps, scientific_axes="THWC", source_axes="THWC",
                gray_preprocessing="finite_frame_minmax_uint8_clahe_gaussian",
            )
            metadata.setdefault("space_unit", "px")
            if save_path:
                self.cancellation_token.raise_if_cancelled()
                preview = SimpleNamespace(
                    name=data.name, time_point=offsets,
                    data_processed=flow_stack, out_processed=metadata,
                )
                metadata["saved_video_path"] = HeartbeatDraw.save_video_task(
                    preview, save_path, step, export_mode=export_mode, scale=1 / scale,
                )
            self.cancellation_token.raise_if_cancelled()
            result = ProcessedData(
                data.timestamp, f"{data.name}@heartbeat", "Heartbeat",
                time_point=offsets, data_processed=flow_stack, out_processed=metadata,
            )
            self.processed_result.emit(result)
            return True
        except TaskCancelled:
            return self._emit_cancelled()
        except Exception as exc:
            return self._emit_failure("光流分析失败", "Heartbeat", exc, data)

    def stop(self):
        """请求中止处理"""
        self.abortion = True
        self.cancellation_token.cancel()

    @staticmethod
    def calculate_amp_dur(data, thr, mode='open'):
        """
        计算单峰事件的振幅和持续时间

        参数:
        data: 一维时序数据
        thr: 阈值
        mode: 'open' 或 'close' (默认'open')

        返回:
        amplitudes: 事件振幅列表
        durations: 事件持续时间列表
        """
        amplitudes = []
        durations = []
        i = 0
        n = len(data)

        while i < n:
            if (mode == 'open' and data[i] > thr) or (mode == 'close' and data[i] < thr):
                start = i
                # 寻找事件结束点
                while i < n and ((mode == 'open' and data[i] > thr) or
                                 (mode == 'close' and data[i] < thr)):
                    i += 1
                end = i - 1

                # 计算事件振幅和持续时间
                event_data = data[start:end + 1]
                amplitude = np.mean(event_data)
                duration = (end - start + 1) * 0.65  # 假设采样间隔为0.65

                amplitudes.append(amplitude)
                durations.append(duration)
            else:
                i += 1

        return np.array(amplitudes), np.array(durations)

    @staticmethod
    def clahe_and_blur(img):
        """Convert finite scientific grayscale samples to OpenCV's uint8 input."""
        values = np.asarray(img)
        if values.ndim != 2 or values.dtype.kind not in "biuf" or not np.all(np.isfinite(values)):
            raise ValueError("光流帧必须为有限实数二维灰度数组")
        if values.dtype == np.uint8:
            gray = np.ascontiguousarray(values)
        else:
            low, high = float(np.min(values)), float(np.max(values))
            if high <= low:
                gray = np.zeros(values.shape, dtype=np.uint8)
            else:
                scaled = (values.astype(np.float64) - low) * (255.0 / (high - low))
                gray = np.clip(np.rint(scaled), 0, 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        return cv2.GaussianBlur(clahe.apply(gray), (3, 3), 0)

    # 上升/下降时间常数拟合(未启用)
    def fit_exponential_time(data, thr, mode='up', n=5):
        """
        拟合指数时间常数

        参数:
        data: 一维时序数据
        thr: 阈值
        mode: 'up' (上升) 或 'down' (下降)
        n: 用于拟合的点数 (默认5)

        返回:
        time_constants: 时间常数列表
        mse_values: 均方误差列表
        """
        time_constants = []
        mse_values = []
        x = np.arange(0, n * 0.65, 0.65)  # 时间轴

        for i in range(n, len(data) - n):
            if mode == 'up':
                # 上升沿检测: 当前点超过阈值，前一点低于阈值
                if data[i] > thr and data[i - 1] < thr:
                    # 取前n个点并反转
                    y = data[i - 1:i + n - 1][::-1]
                    # 调整基线
                    y = 2 * data[i - 1] - y
            else:  # mode == 'down'
                # 下降沿检测: 当前点低于阈值，前一点高于阈值
                if data[i] < thr and data[i - 1] > thr:
                    # 取后n个点并反转
                    y = data[i:i + n][::-1]

            # 指数拟合
            try:
                if mode == 'up' or mode == 'down':
                    # 初始参数估计
                    A0 = y[0] - y[-1]
                    tau0 = 1.0
                    b0 = y[-1]
                    p0 = [A0, tau0, b0]

                    # 指数函数模型
                    def exp_model(x, A, tau, b):
                        return A * np.exp(-x / tau) + b

                    # 拟合
                    popt, pcov = curve_fit(exp_model, x, y, p0=p0)

                    # 计算拟合质量
                    y_fit = exp_model(x, *popt)
                    mse = np.mean((y - y_fit) ** 2)

                    time_constants.append(popt[1])
                    mse_values.append(mse)
            except (RuntimeError, ValueError):
                # 拟合失败时跳过
                continue

        return np.array(time_constants), np.array(mse_values)

    @pyqtSlot(object)
    @task_scoped
    def calculation_operation(self, plan):
        """Execute a validated multi-source calculation plan."""
        try:
            if not isinstance(plan, CalculationPlan):
                raise TypeError("计算任务类型无效")
            self.cancellation_token.raise_if_cancelled()
            self.processing_progress_signal.emit(0, 1)
            result_data, validation = CalculationEngine.execute(plan)
            self.cancellation_token.raise_if_cancelled()
            if result_data.ndim == 0:
                result_data = result_data.reshape(1)
            primary = plan.operands[0].source
            metadata = CalculationMetadataPolicy.build(plan, validation, result_data)
            result_name = plan.result_name or f"{getattr(primary, 'name', 'data')}@math"
            processed = ProcessedData(
                getattr(primary, "timestamp", 0.0),
                result_name,
                "Multi_data_math",
                time_point=metadata.time_point,
                data_processed=result_data,
                out_processed=metadata.out_processed,
            )
            processed.parameters.update(metadata.parameters)
            processed._update_history()
            self.processing_progress_signal.emit(1, 1)
            self.processed_result.emit(processed)
            self.calculator_completed.emit(processed)
            return True
        except TaskCancelled:
            return self._emit_cancelled()
        except Exception as exc:
            expression = getattr(plan, "expression", "")
            operands = []
            for item in getattr(plan, "operands", ()):
                source = getattr(item, "source", None)
                operands.append(
                    f"{getattr(item, 'alias', '?')}: "
                    f"name={getattr(source, 'name', '')}, "
                    f"shape={getattr(source, 'datashape', None)}, "
                    f"dtype={getattr(source, 'datatype', None)}, "
                    f"slice={getattr(item, 'slice_text', '') or 'None'}"
                )
            details = format_exception_details(exc, "多数据运算")
            details = f"expression: {expression}\n" + "\n".join(operands) + f"\n{details}"
            self.calculator_failed.emit(AppError(
                "多数据运算失败", str(exc), stage="多数据运算",
                severity="error", details=details, original=exc,
                task_id=self.task_id,
            ))
            return False

    # 4. 主分析流程
    # def analyze_single_peak_data(data, T, x_m, y_m):
    #     """
    #     完整分析流程
    #
    #     参数:
    #     data: 三维时序数据 (T, H, W)
    #     T: 时间轴
    #     x_m, y_m: ROI起始坐标
    #
    #     返回:
    #     所有分析结果
    #     """
    #     # 1. ROI高斯拟合
    #     amplitudes, centers_x, centers_y = fit_gaussian_roi(data, x_m, y_m)
    #
    #     # 绘制振幅变化
    #     plt.figure(figsize=(10, 6))
    #     plt.plot(T, amplitudes)
    #     plt.xlabel('Time')
    #     plt.ylabel('Amplitude')
    #     plt.title('Amplitude over Time')
    #     plt.xlim([0, 15])
    #     plt.ylim([0, 10])
    #     plt.show()
    #
    #     # 2. 单峰振幅-持续时间计算
    #     open_amps, open_durs = calculate_amp_dur(amplitudes, thr=10, mode='open')
    #     close_amps, close_durs = calculate_amp_dur(amplitudes, thr=10, mode='close')
    #
    #     # 3. 时间常数拟合
    #     with warnings.catch_warnings():
    #         warnings.simplefilter("ignore")
    #         tau_up, mse_up = fit_exponential_time(amplitudes, thr=10, mode='up', n=5)
    #         tau_down, mse_down = fit_exponential_time(amplitudes, thr=10, mode='down', n=5)
    #
    #     # 过滤低质量拟合
    #     tau_up = tau_up[mse_up < 0.1]
    #     tau_down = tau_down[mse_down < 0.1]
    #
    #     # 返回所有结果
    #     results = {
    #         'amplitudes': amplitudes,
    #         'centers_x': centers_x,
    #         'centers_y': centers_y,
    #         'open_amps': open_amps,
    #         'open_durs': open_durs,
    #         'close_amps': close_amps,
    #         'close_durs': close_durs,
    #         'tau_up': tau_up,
    #         'tau_down': tau_down
    #     }
    #
    #     return results
