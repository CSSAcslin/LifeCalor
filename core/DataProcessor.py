import copy
import logging
import shutil
import uuid
from pathlib import Path

import cv2
import numpy as np
import pywt
import h5py
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot, QElapsedTimer, QThread
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
from compute.algorithms import cwt_frequency_trace, stft_frequency_trace
from compute.algorithms.stft_pipeline import run_stft_pipeline
from compute.algorithms.cwt_pipeline import run_cwt_pipeline
from compute.algorithms.preprocess_pipeline import run_preprocess_pipeline
from compute.capabilities import (
    mark_algorithm_verified,
    update_cached_device_capability,
)
from compute.worker import probe_cuda_capability_isolated
from tasks import CancellationToken, TaskCancelled, TaskContextQueue, task_scoped


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
    def __init__(self):
        super().__init__()
        logging.info("例外数据处理线程已启动")

    @staticmethod
    def process_data(data, max_all, min_all, vmean_array):
        process_show = []
        if np.abs(min_all) > np.abs(max_all):
            # n-type 信号中心为黑色，最强值为负
            data_type = 'central negative'
            for every_data in data:
                normalized_data = (every_data - min_all) / (max_all - min_all)
                process_show.append(normalized_data)
            max_mean = np.min(vmean_array)
            phy_max = -min_all
            phy_min = -max_all
        else:
            # p-type 信号中心为白色，最强值为正
            data_type = 'central positive'
            for every_data in data:
                normalized_data = (max_all - every_data) / (max_all - min_all)
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

    @pyqtSlot(np.ndarray,float)
    def detect_bad_frames_auto(self, data: np.ndarray, threshold: float = 3.0) -> List[int]:
        """
        自动检测坏帧
        基于帧间差异和均值离群值检测
        """
        # 计算每帧的均值
        frame_means = np.mean(data, axis=(1, 2))

        # 计算帧间差异
        frame_diff = np.abs(np.diff(frame_means))
        median_diff = np.median(frame_diff)
        mad_diff = 1.4826 * np.median(np.abs(frame_diff - median_diff))

        # 找出异常帧
        z_scores = np.abs((frame_diff - median_diff) / mad_diff)
        potential_bad = np.where(z_scores > threshold)[0]

        # 合并相邻坏帧
        bad_frames = []
        for i in potential_bad:
            if not bad_frames or i > bad_frames[-1] + 1:
                bad_frames.extend([i, i + 1])  # 标记差异大的前后两帧
            elif i == bad_frames[-1] + 1:
                bad_frames.append(i + 1)

        return sorted(list(set(bad_frames)))

    @pyqtSlot(object, list, int)
    def fix_bad_frames(self, data: ProcessedData|Data, bad_frames: List[int], n_frames: int = 2) -> np.ndarray:
        """
        修复坏帧 - 使用前后n帧的平均值替换
        """
        aim_data = data.data_origin
        fixed_data = aim_data.copy()
        total_frames = len(aim_data)

        for frame_idx in bad_frames:
            # 计算前后n帧的范围
            start = max(0, frame_idx - n_frames)
            end = min(total_frames, frame_idx + n_frames + 1)

            # 排除坏帧本身
            valid_frames = [i for i in range(start, end)
                            if i != frame_idx and i not in bad_frames]

            if valid_frames:
                # 计算平均值
                fixed_data[frame_idx] = np.mean(aim_data[valid_frames], axis=0)
            else:
                print(f"警告: 无法修复帧 {frame_idx} - 无有效参考帧")

        data.update_data(**self.amend_data(fixed_data))

    @pyqtSlot(np.ndarray,str,object)
    def plot_data_prepare(self, data: np.ndarray,name:str, father_obj: Data| ProcessedData ):
        """完成plot数据的准备操作"""
        try:
            time_point = father_obj.time_point
            father_dict = father_obj.parameters if isinstance(father_obj, Data) else father_obj.out_processed
            if data.ndim == 2:
                self.plot_singal.emit(data,{'name':name})
            else:
                if time_point is None or time_point.shape != data.shape:
                    if hasattr(father_dict, 'fps'):
                        self.plot_singal.emit(self.add_time_from_fps(data,father_dict['fps']),
                                              {'name':name, 'time_unit':'s'})
                    else:
                        self.plot_singal.emit(self.add_time_simple(data,father_dict['time_step']),
                                              {'name':name, 'time_unit':father_dict.get('time_unit',None)})
                else:
                    self.plot_singal.emit(np.column_stack((time_point,data)),
                                          {'name':name, 'time_unit':father_dict.get('time_unit',None)})
        except Exception as e:
            logging.error(f'数据无法被绘制由于：{e}')

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
        """Return an axis matching a temporal result, including restored legacy data."""
        axis = getattr(data, "time_point", None)
        if axis is not None:
            axis = np.asarray(axis).reshape(-1)
            if axis.size == length:
                return axis

        metadata = getattr(data, "out_processed", None) or getattr(data, "parameters", None) or {}
        candidate = metadata.get("time_series")
        if isinstance(candidate, np.ndarray) and candidate.size == length:
            return candidate.reshape(-1)
        try:
            fps = float(metadata.get("fps", 0))
        except (TypeError, ValueError):
            fps = 0.0
        try:
            window_step = float(metadata.get("window_step", 0))
        except (TypeError, ValueError):
            window_step = 0.0
        if fps > 0 and window_step > 0:
            rebuilt = np.arange(length, dtype=np.float64) * window_step / fps
        else:
            try:
                time_step = float(metadata.get("time_step", 0))
            except (TypeError, ValueError):
                time_step = 0.0
            if time_step > 0:
                rebuilt = np.arange(length, dtype=np.float64) * time_step
            elif fps > 0:
                rebuilt = np.arange(length, dtype=np.float64) / fps
            else:
                rebuilt = np.arange(length, dtype=np.float64)
        logging.warning(
            "数据时间轴长度与结果不匹配，已重建: name=%s, axis=%s, result=%s",
            getattr(data, "source_name", getattr(data, "name", "")),
            0 if axis is None else axis.size,
            length,
        )
        return rebuilt

    @pyqtSlot(object, np.ndarray, str, str)
    def get_fast_selection(self,data, mask, method:str, name:str):
        """本函数是获取快速选取数据并发射的函数"""
        try:
            aim_data = data.image_backup
            T, H, W = aim_data.shape
            mask_flat = mask.reshape(-1)

            # 获取蒙版内的索引
            mask_indices = np.where(mask_flat)[0]

            # 重塑数据以便于提取蒙版区域
            data_reshaped = aim_data.reshape(T, -1)

            # 提取蒙版内的数据
            masked_data = data_reshaped[:, mask_indices]

            # 根据不同的统计方法计算结果
            result = np.zeros(T, dtype=aim_data.dtype)

            for t in range(T):
                frame_data = masked_data[t, :]

                if len(frame_data) == 0:
                    result[t] = np.nan
                    continue

                if method == 'mean':
                    result[t] = np.mean(frame_data)
                elif method == 'max':
                    result[t] = np.max(frame_data)
                elif method == 'min':
                    result[t] = np.min(frame_data)
                elif method == 'median':
                    result[t] = np.median(frame_data)
                elif method == 'quantile_075':
                    result[t] = np.quantile(frame_data, 0.75)
                elif method == 'std':
                    result[t] = np.std(frame_data)
                elif method == 'sum':
                    result[t] = np.sum(frame_data)
                elif method == 'var':
                    result[t] = np.var(frame_data)
                else:
                    raise ValueError(f"不支持的统计方法: {method}。"
                                     f"支持的方法: mean, max, min, median, quantile_075, std, sum, var")

            plot_data = np.column_stack((self.aligned_time_axis(data, T), result))
            self.plot_series_signal.emit(plot_data, name)
        except Exception as exc:
            details = format_exception_details(exc, "anchor 快速提取", data)
            self.processing_error_signal.emit(AppError("快速提取失败", str(exc), stage="anchor 快速提取", details=details))

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
                capability = probe_cuda_capability_isolated(device_index, timeout=15.0)
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

    @pyqtSlot(object,float,int,int,int,str)
    @task_scoped
    def quality_cwt(self,data, target_freq: float,scale_range:int, fps: int, totalscales: int, wavelet: str = 'morl'):
        """
        CWT(连续小波变换)分析信号评估
        参数:
            target_freq: 目标分析频率(Hz)
            EM_fps: 采样频率
            scales: 尺度数组，控制小波变换的频率分辨率
            wavelet: 使用的小波类型(默认为'morl'墨西哥帽小波)
        """
        try:
            mean_signal = get_mean_time_trace(data, self.cancellation_token)
            frame_size = data.framesize  # (宽度, 高度)
            cparam = 2 * pywt.central_frequency(wavelet) * totalscales
            scales = cparam/np.arange(totalscales,1,-1)
            # target_freqs = np.linspace(int(target_freq-5), int(target_freq+5), totalscales//4)
            # scales = pywt.frequency2scale(wavelet, target_freqs * 1.0 / EM_fps)
            self.processing_progress_signal.emit(20, 100)
            # 计算参数
            total_frames = int(data.timelength)
            total_pixels = int(data.framesize[0] * data.framesize[1])
            height, width = frame_size
            self.processing_progress_signal.emit(40, 100)
            # 计算平均信号的CWT (用于质量评估)

            coefficients, frequencies = pywt.cwt(mean_signal, scales, wavelet, sampling_period=1.0 / fps)
            self.processing_progress_signal.emit(70, 100)
            # 发送平均信号CWT结果
            self.processed_result.emit(ProcessedData(data.timestamp,
                                                     f'{data.name}@cwt_q',
                                                     'cwt_quality',
                                                     time_point=np.arange(total_frames) / fps,
                                                     data_processed=np.abs(coefficients),
                                                     out_processed={
                                                         'frequencies' : frequencies,
                                                         'time_series' : np.arange(total_frames) / fps,
                                                         'target_freq' : target_freq,
                                                         'scale_range' : scale_range,
                                                         'total_scales' : totalscales,
                                                         'wavelet_name' : wavelet,
                                                     }))
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
            workspace_per_pixel = (
                int(totalscales) * shape[0] * complex_dtype.itemsize
                + shape[0] * (
                    np.dtype(precision_info.compute_dtype).itemsize
                    + np.dtype(precision_info.output_dtype).itemsize
                )
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
            budget = ResourceBudget(
                host_limit_bytes=max(
                    256, int(options.get("host_memory_limit_mb", 4096))
                ) * 1024 ** 2,
                disk_free_bytes=int(shutil.disk_usage(cache_dir).free),
                cpu_workers=max(1, int(options.get("cpu_workers", 1))),
            )
            allow_fallback = bool(options.get("allow_cpu_fallback", True))
            plan = plan_compute(
                request,
                budget,
                capabilities=tuple(options.get("capabilities", ())),
                default_backend=BackendPreference.AUTO,
                allow_cpu_fallback=allow_fallback,
                disk_output_threshold_bytes=max(
                    1, int(options.get("cache_threshold_mb", 512))
                ) * 1024 ** 2,
            )
            self.compute_plan_signal.emit(plan)
            if plan.actual_backend != "cpu":
                raise RuntimeError("CWT CUDA 后端尚未通过科学一致性验证")

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
            )
            inherited = {
                key: value for key, value in (getattr(data, "out_processed", {}) or {}).items()
                if key != "unfolded_data"
            }
            inherited.update(getattr(data, "parameters", {}) or {})
            metadata = execution_metadata(
                result.plan,
                execution="bounded_cpu",
                scale_reduction="normalized_mean",
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
    def accumulate_amplitude(self,data):
        """累计时间振幅图"""
        self.processed_result.emit(ProcessedData(data.timestamp,
                                         f'{data.name}@atam',
                                         "Accumulated_time_amplitude_map",
                                         time_point=data.time_point, # 忘记为什么0.12.1要改这个了，改这个就没法处理单通道了
                                         data_processed=np.mean(data.data_processed if isinstance(data,ProcessedData) else data.data_origin, axis=0),
                                         out_processed={**(data.out_processed if isinstance(data,ProcessedData) else data.parameters),}
                                                             ))
        logging.info("累计时间振幅计算已完成")

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
    def twoD_fourier_transform(self,data,):
        """
            对3D时序视频数据进行2D傅里叶变换

            参数:
            video_data: 形状为 (帧数, 高度, 宽度) 的numpy数组

            返回:
            magnitude_spectra: 傅里叶幅度谱数组
            magnitude_log: 幅度谱的log（默认主参数）
            phase_spectra: 傅里叶相位谱数组
            """
        if isinstance(data, ProcessedData):
            pure_data = data.data_processed
        else:
            pure_data = data.data_origin
            pass
        try:
            if data.timelength == 1:
                frames = 1
                height, width = data.datashape
            else:
                frames, height, width = data.datashape
            tfshift = np.zeros((frames, height, width),dtype=np.complex64)
            magnitude_spectra = np.zeros((frames, height, width))
            magnitude_log = np.zeros((frames, height, width))
            phase_spectra = np.zeros((frames, height, width))
            self.processing_progress_signal.emit(0,frames)
            for i in range(frames):
                # 2D傅里叶变换
                f = np.fft.fft2(pure_data[i])
                fshift = np.fft.fftshift(f)  # 将低频移到中心

                tfshift[i] = fshift.copy()
                # 幅度谱
                magnitude_spectra[i] = np.abs(fshift)

                # 相位谱
                phase_spectra[i] = np.angle(fshift)
                self.processing_progress_signal.emit(i, frames)

                magnitude_log[i] = np.log(magnitude_spectra[i]+1)

            self.processed_result.emit(ProcessedData(data.timestamp,
                                       f'{data.name}@2DFT',
                                       "2D_Fourier_transform",
                                       time_point = data.time_point,
                                       data_processed=np.squeeze(magnitude_log),
                                       out_processed={'twoD_FFT':np.squeeze(tfshift),
                                                      'magnitude_spectra': np.squeeze(magnitude_spectra),
                                                      'phase_spectra': np.squeeze(phase_spectra),**data.out_processed}))
            self.processing_progress_signal.emit(frames, frames)
            return True
        except Exception as e:
            return self._emit_failure("数据处理失败", "2D_Fourier_transform", e, data)

    @pyqtSlot(object)
    @task_scoped
    def twoD_inverse_fourier_transform(self, data):
        """
        对2D傅里叶变换后的频域数据进行逆变换，恢复3D时序视频数据（空间域）

        参数:
        data: ProcessedData对象 或 原始数据。必须包含复数频域数据。

        返回:
        reconstructed_data: 逆傅里叶变换后恢复的空间域/图像数组
        """
        # 1. 尝试获取包含幅度和相位的"复数频域数据"
        if isinstance(data, ProcessedData):
            # 优先从 out_processed 中提取正变换时保存的完整复数数据 'twoD_FFT'
            if 'twoD_FFT' in data.out_processed:
                pure_data = data.out_processed['twoD_FFT']
            else:
                pure_data = data.data_processed
        else:
            pure_data = data.data_origin

        # 确保 pure_data 是复数类型。如果丢失了相位信息（纯实数），逆变换将是不准确的。
        if not np.iscomplexobj(pure_data):
            print("警告: 输入数据不包含复数信息(缺失相位)，恢复的图像可能只有边缘或失真。")

        try:
            # 2. 形状与帧数解析
            # 由于之前存入 out_processed 时可能被 squeeze 过，这里做一个稳健的维度处理
            pure_data = np.atleast_3d(pure_data) if pure_data.ndim == 2 else pure_data

            if data.timelength == 1 or pure_data.shape[0] == 1:
                frames = 1
                height, width = pure_data.shape[-2], pure_data.shape[-1]
            else:
                frames, height, width = pure_data.shape

            # 预分配内存保存恢复的图像数据
            # 正常逆变换后的图像应该是实数，所以预分配为 float
            reconstructed_data = np.zeros((frames, height, width), dtype=np.float32)

            self.processing_progress_signal.emit(0, frames)

            # 3. 逐帧进行逆变换
            for i in range(frames):
                # 如果只有1帧且被强行升维，pure_data[i]的提取需要注意，但上面的 atleast_3d 和 正常3D数据都能兼容
                frame_data = pure_data[i] if pure_data.ndim == 3 else pure_data

                # 第一步：逆中心化 (将低频从中心移回左上角)
                f_ishift = np.fft.ifftshift(frame_data)

                # 第二步：2D逆傅里叶变换
                img_back = np.fft.ifft2(f_ishift)

                # 第三步：取实部 (由于计算精度问题，ifft2的结果通常带有极微小的虚部，取实部或取绝对值来消除)
                # 注：如果你的原图包含负数，用 np.real()；如果原图全是正数值亮度，用 np.abs() 也可以。这里使用标准 np.real()
                reconstructed_data[i] = np.real(img_back)

                self.processing_progress_signal.emit(i + 1, frames)  # 进度+1

            # 4. 封装并发送处理完成信号
            # 继承上一步的属性，修改标识
            self.processed_result.emit(ProcessedData(
                data.timestamp,
                f'{data.name}@2DIFFT',
                "2D_Inverse_Fourier_transform",
                time_point=data.time_point,
                data_processed=np.squeeze(reconstructed_data),  # 默认主参数为恢复的图像
                out_processed={
                    'reconstructed_img': np.squeeze(reconstructed_data),
                    **data.out_processed  # 保留之前的处理记录
                }
            ))

            self.processing_progress_signal.emit(frames, frames)
            return True

        except Exception as e:
            return self._emit_failure("数据处理失败", "2D_Inverse_Fourier_transform", e, data)

    @pyqtSlot(object, int, int ,list, str, str, float)
    @task_scoped
    def heartbeat_movement(self, data, step, base_num, after_series, save_path = "", export_mode = 'video', scale = 1):
        """心肌细胞运动分析，使用
        稠密光流 (Farneback算法)"""
        # try:
        timer = QElapsedTimer()
        timer.start()
        if isinstance(data, ProcessedData):
            aim_data = data.data_processed
            out_processed = data.out_processed
        else:
            aim_data = data.data_origin
            out_processed = data.parameters

        if base_num != -1:
            # 固定基准帧模式
            num = len(after_series)
            pairs = [(base_num, current) for current in after_series]
            # 时间点：相对于基准帧的时间
            time_point = np.array(after_series) - base_num
            logging.info(f"模式：固定基准帧 {base_num}, 比较 {num} 帧")
            process_name = "比较模式"
        else:
            # 相邻连续帧模式
            num = len(after_series) - 1
            pairs = list(zip(after_series[:-1], after_series[1:]))
            # 时间点：相对于序列第一帧的时间
            time_point = np.array(after_series[1:]) - after_series[0]
            logging.info(f"模式：相邻连续帧, 比较 {num} 组")
            process_name = "连续分析"
        self.processing_progress_signal.emit(0, num + 2)
        h, w = aim_data[0].shape
        # 建议直接初始化由 numpy 堆叠的大数组，比 list 推导式稍微快一点点且内存连续
        flow_stack = np.zeros((num, h, w, 2), dtype=np.float32)
        magnitude_stack = np.zeros((num, h, w), dtype=np.float32)
        angle_stack = np.zeros((num, h, w), dtype=np.float32)
        max_speed = np.zeros(num)
        mean_speed = np.zeros(num)

        self.processing_progress_signal.emit(1,num+2)

        # 辅助函数：获取预处理后的图像 (带简单缓存),避免重复预处理
        img_cache = {}

        def get_processed_img(idx):
            if idx not in img_cache:
                # 这里的 clahe_and_blur 应该包含你之前的灰度拉伸+CLAHE+高斯模糊
                img_cache[idx] = self.clahe_and_blur(aim_data[idx])
            return img_cache[idx]

        # 如果是固定基准帧，先缓存基准帧，避免循环里反复查
        if base_num != -1:
            base_img_fixed = get_processed_img(base_num)
            base_num_series = base_num
        else:
            base_img_list = np.zeros((num, h, w), dtype=np.float32)
            base_num_series = np.zeros(num, dtype=int)

        for i, (idx_prev, idx_curr) in enumerate(pairs):
            # 获取图像
            if base_num != -1:
                prev_gray = base_img_fixed
            else:
                prev_gray = get_processed_img(idx_prev)
                base_img_list[i] = aim_data[idx_prev]
                base_num_series[i] = idx_prev

            next_gray = get_processed_img(idx_curr)

            # 光流计算
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, next_gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2,
                flags=0)  # 目前参数都定死了

            # 坐标转换与速度计算
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

            # 真实物理速度计算
            # 注意：这里的时间间隔是 (idx_curr - idx_prev)，在相邻模式下通常是1，但在固定模式下会变大
            delta_frame = idx_curr - idx_prev
            if delta_frame == 0: delta_frame = 1  # 防止除以0，虽然逻辑上不应该出现

            factor = out_processed['space_step'] * out_processed['fps'] / delta_frame
            magnitude_real = mag * factor

            # 存入结果
            flow_stack[i] = flow
            magnitude_stack[i] = magnitude_real
            angle_stack[i] = ang
            max_speed[i] = np.max(magnitude_real)
            mean_speed[i] = np.mean(magnitude_real)

            # 内存管理：如果是相邻帧模式，prev_gray (idx_prev) 在下一轮就不会再用了，可以从缓存删除以省内存
            if base_num == -1 and idx_prev in img_cache:
                del img_cache[idx_prev]

            # 发送进度
            self.processing_progress_signal.emit(i + 2, num + 2)

        # 5. 打包结果
        # 注意：base_data 在相邻模式下没有单一值，现在存了数组，主要是后续绘图用
        result_base_data = aim_data[base_num] if base_num != -1 else base_img_list

        result = ProcessedData(data.timestamp,
                                           f'{data.name}@heartbeat',
                                           "Heartbeat",
                                           time_point = time_point,
                                           data_processed=flow_stack,
                                           out_processed={'process_name':process_name,
                                                          'sampling_step':step,
                                                          "base_num": base_num_series,
                                                          'base_data': result_base_data,
                                                          'after_series': after_series,
                                                          'magnitude_list': magnitude_stack,
                                                          'angle_list': angle_stack,
                                                          'max_speed': max_speed ,
                                                          'mean_speed': mean_speed,
                                                          **out_processed})

        # 内部保存
        if save_path and isinstance(save_path, str) and len(save_path) > 0:
            logging.info("计算完成，开始后台生成视频...")
            # 发送一个信号告诉UI正在保存（可选，让进度条显示"Saving..."）
            self.processing_progress_signal.emit(num + 1, num + 2)

            # try:
                # 直接调用独立的保存函数
                # 因为在 Agg 后端下运行，且数据是独立的，不会冲突
            video_file = HeartbeatDraw.save_video_task(result, save_path, step, export_mode = export_mode, scale = 1/scale)
            logging.info(f"视频已保存至: {video_file}")

            # 你可以将视频路径放入 out_processed 以便 UI 知道
            result.out_processed['saved_video_path'] = video_file

            # except Exception as e:
            #     logging.error(f"保存视频失败: {e}")

        self.processing_progress_signal.emit(num + 2, num + 2)
        self.processed_result.emit(result)
        return True

        # except Exception as e:
        #     self.processed_result.emit({'type': "heartbeat", 'error': str(e)})
        #     return False

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
        """限制对比度自适应直方图均衡化+模糊处理"""
        # CLAHE (限制对比度自适应直方图均衡化)
        # 对于明场细胞图像，这是增强纹理的神器
        # clipLimit: 阈值，越大对比度越强，但也可能放大噪声，建议 2.0-4.0
        # tileGridSize: 局部窗口大小，(8,8)是标准值
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        img_clahe = clahe.apply(img)

        # 轻微的高斯模糊
        # 因为你的图像较小(300x200)，且我们做过增强，噪声也会变明显
        # 用 (3,3) 的小核既能去噪又不至于损失太多细节
        img_blur = cv2.GaussianBlur(img_clahe, (3, 3), 0)
        return img_blur

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
