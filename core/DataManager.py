import logging
import os
import time
import copy
import weakref
import shutil
import tempfile
from pathlib import Path

import numpy as np
import tifffile as tiff
import sif_parser
import cv2
from collections import deque
from typing import ClassVar, Optional, List, Dict, Any, Union
from PyQt5.QtCore import QObject, pyqtSignal, QThread, pyqtSlot
from dataclasses import dataclass, field, fields
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.ndimage.interpolation import zoom
from ArrayCache import (
    ArrayCacheConfig,
    ArrayStore,
    ArrayRef,
    cache_large_arrays_in_mapping,
    resolve_array,
    should_cache_array,
)
from display.source import DisplaySourceFactory
from display.renderer import FrameRenderParams, FrameRenderer
from tasks.model import CancellationToken, TaskCancelled
from diagnostics import AppError, format_exception_details
_ARRAY_CACHE_CONFIG = ArrayCacheConfig(cache_dir=Path.cwd() / ".lifecalor_cache", threshold_bytes=512 * 1024 * 1024)
_ARRAY_CACHE_PROGRESS_CALLBACK = None


def configure_array_cache(config: ArrayCacheConfig) -> None:
    global _ARRAY_CACHE_CONFIG
    _ARRAY_CACHE_CONFIG = config
    _ARRAY_CACHE_CONFIG.ensure_dir()


def set_array_cache_progress_callback(callback) -> None:
    global _ARRAY_CACHE_PROGRESS_CALLBACK
    _ARRAY_CACHE_PROGRESS_CALLBACK = callback


def get_array_store(progress_callback=None) -> ArrayStore:
    return ArrayStore(_ARRAY_CACHE_CONFIG, progress_callback=progress_callback or _ARRAY_CACHE_PROGRESS_CALLBACK)


def collect_array_refs(value) -> list[ArrayRef]:
    refs = []
    if isinstance(value, ArrayRef):
        return [value]
    if isinstance(value, dict):
        for item in value.values():
            refs.extend(collect_array_refs(item))
        return refs
    if isinstance(value, (list, tuple, set, deque)):
        for item in value:
            refs.extend(collect_array_refs(item))
        return refs
    if hasattr(value, "__dict__"):
        for item in vars(value).values():
            refs.extend(collect_array_refs(item))
    return refs


def clear_array_cache(active_refs=None) -> int:
    store = get_array_store()
    if active_refs is None:
        return store.clear_all()
    return store.cleanup_orphans(active_refs)


def video_fps_for_duration(num_frames, duration) -> int:
    try:
        duration_value = float(duration)
    except (TypeError, ValueError):
        duration_value = 1.0
    if duration_value <= 0:
        duration_value = 1.0
    return max(1, int(round(max(1, int(num_frames)) / duration_value)))


def prepare_video_frames(data):
    frames = np.asarray(data)
    if frames.ndim == 2:
        return frames[np.newaxis, ...], False
    if frames.ndim == 3:
        return frames, False
    if frames.ndim == 4 and frames.shape[-1] in (3, 4):
        if frames.shape[-1] == 4:
            frames = frames[..., :3]
        return frames, True
    raise ValueError(f"当前数据不是可导出的视频图像序列，shape={frames.shape}")


def prepare_still_frame(frame):
    image = np.asarray(frame)
    if image.ndim == 2:
        return image, 'minisblack'
    if image.ndim == 3 and image.shape[-1] == 1:
        return image[..., 0], 'minisblack'
    if image.ndim == 3 and image.shape[-1] == 4:
        return image, 'rgb'
    if image.ndim == 3 and image.shape[-1] == 3:
        return image, 'rgb'
    raise ValueError(f"当前帧不是可导出的图像，shape={image.shape}")


def image_mode_for_pillow(frame):
    image, photometric = prepare_still_frame(frame)
    if photometric == 'minisblack':
        return image, 'L'
    if image.ndim == 3 and image.shape[-1] == 4:
        return image, 'RGBA'
    return image, 'RGB'


def _is_aligned_temporal_array(candidate, source):
    if candidate is None or source is None:
        return False
    candidate_array = np.asarray(candidate)
    source_array = np.asarray(source)
    if source_array.ndim < 3 or candidate_array.ndim < 3:
        return False
    return candidate_array.shape[0] == source_array.shape[0]


def export_array_from_data(data, format_type, is_temporal):
    if isinstance(data, np.ndarray):
        return data.copy()

    source = getattr(data, "image_backup", None)
    display = getattr(data, "image_data", None)

    if is_temporal:
        if _is_aligned_temporal_array(display, source):
            return display
        if source is not None:
            return source

    if display is not None:
        return display
    if source is not None:
        return source
    raise ValueError("当前数据对象不包含可导出的图像数组")


def render_frame_for_export(frame, render_params=None):
    if not render_params or not render_params.get("use_colormap", False):
        return np.asarray(frame)
    params = FrameRenderParams(
        use_colormap=True,
        colormap=render_params.get("colormap", "jet"),
        auto_range=bool(render_params.get("auto_range", True)),
        min_value=render_params.get("min_value"),
        max_value=render_params.get("max_value"),
    )
    return FrameRenderer.render(np.asarray(frame), params).image


def export_as_temporal_images(result, is_temporal):
    image = np.asarray(result)
    return bool(is_temporal and image.ndim in (3, 4))

class CacheLoadCancelled(Exception):
    """Raised when a cache restore task is cancelled by the user."""


def cache_array_or_keep_memory(store: ArrayStore, array: np.ndarray, owner_id: str, field_name: str):
    try:
        return store.put_array(array, owner_id, field_name)
    except Exception:
        logging.error("缓存写入失败，保留内存数组: field=%s shape=%s dtype=%s", field_name, getattr(array, "shape", None), getattr(array, "dtype", None))
        return array


def force_cache_arrays(target, include_image_import: bool = False, cancellation_token=None, progress_callback=None) -> list[ArrayRef]:
    """Persist a history object's array payloads to npy refs, regardless of threshold."""
    store = get_array_store(progress_callback=progress_callback)
    refs: list[ArrayRef] = []
    owner_id = f"{target.__class__.__name__}_{getattr(target, 'serial_number', 'unknown')}_{getattr(target, 'timestamp', '')}"

    def cache_storage(storage_attr: str, public_attr: str, field_name: str):
        storage = object.__getattribute__(target, "__dict__").get(storage_attr)
        if isinstance(storage, ArrayRef):
            refs.append(storage)
            return
        if isinstance(storage, np.ndarray):
            cancellation_token.raise_if_cancelled() if cancellation_token is not None else None
            ref = store.put_array(storage, owner_id, field_name, cancellation_token=cancellation_token)
            if isinstance(ref, ArrayRef):
                object.__setattr__(target, storage_attr, ref)
                object.__setattr__(target, public_attr, ref)
                refs.append(ref)

    if isinstance(target, Data):
        cache_storage("_data_origin_storage", "data_origin", "data_origin")
        if include_image_import:
            cache_storage("_image_import_storage", "image_import", "image_import")
    elif isinstance(target, ProcessedData):
        cache_storage("_data_processed_storage", "data_processed", "data_processed")
        for key, value in list((target.out_processed or {}).items()):
            if isinstance(value, ArrayRef):
                refs.append(value)
            elif isinstance(value, np.ndarray):
                cancellation_token.raise_if_cancelled() if cancellation_token is not None else None
                ref = store.put_array(value, owner_id, f"out_processed_{key}", cancellation_token=cancellation_token)
                if isinstance(ref, ArrayRef):
                    target.out_processed[key] = ref
                    refs.append(ref)
    else:
        raise TypeError(f"不支持强制缓存的数据类型: {type(target).__name__}")
    return refs


def materialize_cached_arrays(target, progress_callback=None, cancel_check=None, cancellation_token=None):
    """Load ArrayRef-backed arrays into memory, suitable for worker-thread execution."""
    store = get_array_store(progress_callback=progress_callback)
    cancellation_token = cancellation_token or CancellationToken()

    def check_cancelled():
        if cancel_check is not None and cancel_check():
            cancellation_token.cancel()
        try:
            cancellation_token.raise_if_cancelled()
        except TaskCancelled as exc:
            raise CacheLoadCancelled(str(exc)) from exc

    if isinstance(target, Data):
        check_cancelled()
        storage = object.__getattribute__(target, "__dict__").get("_data_origin_storage")
        if isinstance(storage, ArrayRef):
            loaded = store.load_ref(storage, mmap_mode=None, cancellation_token=cancellation_token)
            check_cancelled()
            object.__setattr__(target, "_data_origin_storage", loaded)
            object.__setattr__(target, "data_origin", loaded)
        check_cancelled()
        image_storage = object.__getattribute__(target, "__dict__").get("_image_import_storage")
        if isinstance(image_storage, ArrayRef):
            loaded = store.load_ref(image_storage, mmap_mode=None)
            check_cancelled()
            object.__setattr__(target, "_image_import_storage", loaded)
            object.__setattr__(target, "image_import", loaded)
    elif isinstance(target, ProcessedData):
        check_cancelled()
        storage = object.__getattribute__(target, "__dict__").get("_data_processed_storage")
        if isinstance(storage, ArrayRef):
            loaded = store.load_ref(storage, mmap_mode=None, cancellation_token=cancellation_token)
            check_cancelled()
            object.__setattr__(target, "_data_processed_storage", loaded)
            object.__setattr__(target, "data_processed", loaded)
        for key, value in list((target.out_processed or {}).items()):
            check_cancelled()
            if isinstance(value, ArrayRef):
                target.out_processed[key] = store.load_ref(value, mmap_mode=None, cancellation_token=cancellation_token)
    check_cancelled()
    return target


class ArrayLoadWorker(QObject):
    progress_signal = pyqtSignal(object, object, str)
    finished_signal = pyqtSignal(object)
    error_signal = pyqtSignal(str)
    cancelled_signal = pyqtSignal()

    def __init__(self, target, cancellation_token=None):
        super().__init__()
        self.target = target
        self.cancellation_token = cancellation_token or CancellationToken()
        self._cancel_requested = False

    def cancel(self):
        self._cancel_requested = True
        self.cancellation_token.cancel()

    def is_cancel_requested(self):
        return self._cancel_requested

    def run(self):
        try:
            materialize_cached_arrays(
                self.target,
                progress_callback=self.progress_signal.emit,
                cancel_check=self.is_cancel_requested,
                cancellation_token=self.cancellation_token,
            )
            if self._cancel_requested:
                self.cancelled_signal.emit()
            else:
                self.finished_signal.emit(self.target)
        except CacheLoadCancelled:
            logging.info("缓存读取已取消")
            self.cancelled_signal.emit()
        except Exception as exc:
            logging.exception("缓存读取线程失败")
            self.error_signal.emit(str(exc))


class DataManager(QObject):
    # save_request_back = pyqtSignal(dict)
    # read_request_back = pyqtSignal(dict)
    # remove_request_back = pyqtSignal(dict)
    # amend_request_back = pyqtSignal(dict)
    data_progress_signal = pyqtSignal(int, int)
    processed_result = pyqtSignal(object)
    export_finished = pyqtSignal(object)
    export_failed = pyqtSignal(str)
    export_cancelled = pyqtSignal()
    processing_error_signal = pyqtSignal(object)
    processing_cancelled_signal = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.cancellation_token = CancellationToken()
        logging.info("图像数据管理线程已启动")

    def set_cancellation_token(self, token):
        self.cancellation_token = token or CancellationToken()

    def cancel(self):
        self.cancellation_token.cancel()

    def _check_cancelled(self):
        self.cancellation_token.raise_if_cancelled()

    @pyqtSlot(object, str, str, str, bool, dict)
    def export_data(self, data, output_dir, prefix, format_type='tif', is_temporal=True, arg_dict=None):
        format_type = format_type.lower()
        arg_dict = arg_dict or {}
        os.makedirs(output_dir, exist_ok=True)
        staging_dir = tempfile.mkdtemp(prefix=".lifecalor_export_", dir=output_dir)
        try:
            self._check_cancelled()
            duration = arg_dict.get('duration', 60)
            cmap = arg_dict.get('cmap', 'jet')
            max_bound = arg_dict.get('max_bound', 255)
            min_bound = arg_dict.get('min_bound', 0)
            title = arg_dict.get('title', '')
            colorbar_label = arg_dict.get('colorbar_label', '')
            result = export_array_from_data(data, format_type, is_temporal)
            render_params = arg_dict.get('render_params')
            if format_type == 'tif':
                created = self.export_as_tif(result, staging_dir, prefix, is_temporal, render_params)
            elif format_type == 'avi':
                created = self.export_as_avi(result, staging_dir, prefix, duration, render_params)
            elif format_type == 'png':
                created = self.export_as_png(result, staging_dir, prefix, is_temporal, render_params)
            elif format_type == 'gif':
                created = self.export_as_gif(result, staging_dir, prefix, duration, render_params)
            elif format_type == 'plt':
                created = self.export_as_plt(data.image_backup, staging_dir, prefix, is_temporal, cmap, max_bound, min_bound, title, colorbar_label)
            else:
                raise ValueError(f"不支持格式: {format_type}。请使用 'tif', 'avi', 'png' 或 'gif'")
            self._check_cancelled()
            final_files = []
            for staged_file in created:
                destination = os.path.join(output_dir, os.path.basename(staged_file))
                os.replace(staged_file, destination)
                final_files.append(destination)
            shutil.rmtree(staging_dir, ignore_errors=True)
            self.export_finished.emit(final_files)
            return final_files
        except TaskCancelled:
            shutil.rmtree(staging_dir, ignore_errors=True)
            logging.warning("导出任务已取消: %s", prefix)
            self.export_cancelled.emit()
            return None
        except Exception as exc:
            shutil.rmtree(staging_dir, ignore_errors=True)
            logging.exception("导出任务失败: format=%s prefix=%s", format_type, prefix, extra={"lifecalor_user_reported": True})
            self.export_failed.emit(str(exc))
            return None

    def _normalize_data(self, data):
        """统一归一化处理，支持彩色/灰度数据"""
        if data.dtype == np.uint8:
            return data.copy()

        # 计算全局最小最大值（避免逐帧计算不一致）
        data_min = data.min()
        data_max = data.max()

        # 处理全零数据
        if data_max - data_min < 1e-6:
            return np.zeros_like(data, dtype=np.uint8)

        normalized = ((data - data_min) / (data_max - data_min) * 255).astype(np.uint8)
        return normalized

    def export_as_tif(self, result, output_dir, prefix, is_temporal=True, render_params=None):
        """支持彩色TIFF导出"""
        created_files = []
        if export_as_temporal_images(result, is_temporal):
            num_frames = result.shape[0]
            num_digits = len(str(num_frames))
            self.data_progress_signal.emit(0, num_frames)

            for frame_idx in range(num_frames):
                self._check_cancelled()
                frame_name = f"{prefix}-{frame_idx:0{num_digits}d}.tif"
                output_path = os.path.join(output_dir, frame_name)

                frame = render_frame_for_export(result[frame_idx], render_params)
                frame, photometric = prepare_still_frame(frame)
                tiff.imwrite(output_path, frame, photometric=photometric)

                created_files.append(output_path)
                self.data_progress_signal.emit(frame_idx + 1, num_frames)
        else:
            num_frames = 1
            frame_name = f"{prefix}.tif"
            output_path = os.path.join(output_dir, frame_name)
            frame = render_frame_for_export(result, render_params)
            frame, photometric = prepare_still_frame(frame)
            tiff.imwrite(output_path, frame, photometric=photometric)
            created_files.append(output_path)
            self.data_progress_signal.emit(num_frames + 1, num_frames)

        logging.info(f'导出TIFF完成: {output_dir}, 共{num_frames}帧')
        return created_files

    def export_as_avi(self, result, output_dir, prefix, duration=60, render_params=None):
        """逐帧渲染并导出视频，避免创建完整伪彩数组。"""
        os.makedirs(output_dir, exist_ok=True)
        frames, source_is_color = prepare_video_frames(result)
        num_frames = frames.shape[0]
        self.data_progress_signal.emit(0, num_frames)
        output_path = os.path.join(output_dir, f"{prefix}.avi")
        fps = video_fps_for_duration(num_frames, duration)
        first = render_frame_for_export(frames[0], render_params)
        if render_params and render_params.get("use_colormap", False):
            first = first[..., :3]
        elif not source_is_color:
            data_min = np.nanmin(np.abs(frames) if np.iscomplexobj(frames) else frames)
            data_max = np.nanmax(np.abs(frames) if np.iscomplexobj(frames) else frames)
            gray_params = FrameRenderParams(auto_range=False, min_value=float(data_min), max_value=float(data_max))
            first = FrameRenderer.render(first, gray_params).image
        is_color = first.ndim == 3 and first.shape[-1] >= 3
        height, width = first.shape[:2]
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MJPG'), fps, (width, height), isColor=is_color)
        if not out.isOpened():
            raise OSError(f"无法创建AVI文件: {output_path}")
        try:
            for frame_idx in range(num_frames):
                self._check_cancelled()
                frame = render_frame_for_export(frames[frame_idx], render_params)
                if render_params and render_params.get("use_colormap", False):
                    frame = frame[..., :3]
                elif not source_is_color:
                    frame = FrameRenderer.render(frame, gray_params).image
                if frame.ndim == 3:
                    frame = frame[..., :3][..., ::-1]
                out.write(np.ascontiguousarray(frame))
                self.data_progress_signal.emit(frame_idx + 1, num_frames)
        finally:
            out.release()
        logging.info('导出AVI完成: %s, 共%s帧', output_path, num_frames)
        return [output_path]

    def export_as_png(self, result, output_dir, prefix, is_temporal=True, render_params=None):
        """支持彩色PNG导出"""
        created_files = []
        # 归一化处理
        normalized = result if render_params and render_params.get('use_colormap', False) else self._normalize_data(result)

        if export_as_temporal_images(result, is_temporal):
            num_frames = result.shape[0]
            num_digits = len(str(num_frames))
            self.data_progress_signal.emit(0, num_frames)
            for frame_idx in range(num_frames):
                self._check_cancelled()
                frame_name = f"{prefix}-{frame_idx:0{num_digits}d}.png"
                output_path = os.path.join(output_dir, frame_name)

                frame = render_frame_for_export(normalized[frame_idx], render_params)
                frame, mode = image_mode_for_pillow(frame)
                img = Image.fromarray(frame, mode)

                img.save(output_path)
                created_files.append(output_path)
                self.data_progress_signal.emit(frame_idx + 1, num_frames)

        else:
            num_frames = 1
            frame_name = f"{prefix}.png"
            output_path = os.path.join(output_dir, frame_name)
            frame = render_frame_for_export(normalized, render_params)
            frame, mode = image_mode_for_pillow(frame)
            img = Image.fromarray(frame, mode)
            img.save(output_path)
            created_files.append(output_path)
            self.data_progress_signal.emit(num_frames + 1, num_frames)

        logging.info(f'导出PNG完成: {output_dir}, 共{num_frames}帧')
        return created_files

    def export_as_gif(self, result, output_dir, prefix, duration=60, render_params=None):
        """彩色GIF导出"""
        normalized = result if render_params and render_params.get('use_colormap', False) else self._normalize_data(result)
        normalized, is_color = prepare_video_frames(normalized)
        num_frames = normalized.shape[0]
        self.data_progress_signal.emit(0, num_frames)
        images = []
        palette_img = None

        for frame_idx in range(num_frames):
            self._check_cancelled()
            frame = render_frame_for_export(normalized[frame_idx], render_params)
            if frame.ndim == 3 and frame.shape[-1] == 4:
                frame = frame[..., :3]
            is_color = frame.ndim == 3

            if is_color:
                img = Image.fromarray(frame, 'RGB')
                if palette_img is None:
                    palette_img = img.convert('P', palette=Image.ADAPTIVE, colors=256)
                    images.append(palette_img)
                else:
                    images.append(img.quantize(palette=palette_img))
            else:
                images.append(Image.fromarray(frame, 'L'))

        output_path = os.path.join(output_dir, f"{prefix}.gif")
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=max(1, int(duration or 1)),
            loop=0,
            optimize=True
        )
        self.data_progress_signal.emit(num_frames, num_frames)
        logging.info(f'导出GIF完成: {output_path}, 共{num_frames}帧')
        return [output_path]

    def export_as_plt(self, result, output_dir, prefix, is_temporal=True, cmap='viridis',
                      max_bound=255, min_bound=0, title='', colorbar_label=''):
        """使用matplotlib的彩色导出"""
        mpl.use('Agg')

        # 默认配置
        config = {
            'title': title,
            'xlabel': '',
            'ylabel': '',
            'cmap': cmap,
            'dpi': 300,
            'figsize': (8, 6),
            'colorbar_label': colorbar_label,
            'vmin': min_bound,
            'vmax': max_bound,
            'aspect': 'equal'
        }
        # config = {**default_config, **heatmap_config} # 新合并写法，多学学

        created_files = []

        def create_heatmap_frame(data, frame_idx=None):
            """创建单帧热图"""
            # 创建图形
            fig, ax = plt.subplots(figsize=config['figsize'])

            # 绘制热图
            im = ax.imshow(data, cmap=config['cmap'],
                           vmin=config['vmin'], vmax=config['vmax'],
                           aspect=config['aspect'])

            # 设置标题和标签
            if config['title']:
                if frame_idx is not None:
                    ax.set_title(f"{config['title']} - Frame {frame_idx}")
                else:
                    ax.set_title(config['title'])
            #
            # if config['xlabel']:
            #     ax.set_xlabel(config['xlabel'])
            # if config['ylabel']:
            #     ax.set_ylabel(config['ylabel'])

            # 添加colorbar
            cbar = fig.colorbar(im, ax=ax)
            if config['colorbar_label']:
                cbar.set_label(config['colorbar_label'])

            # 设置紧凑布局
            plt.tight_layout()

            # 生成文件名
            if frame_idx is not None:
                num_digits = len(str(result.shape[0]))
                frame_name = f"{prefix}-{frame_idx:0{num_digits}d}_heatmap.tif"
            else:
                frame_name = f"{prefix}_heatmap.tif"

            output_path = os.path.join(output_dir, frame_name)

            # 保存图像
            plt.savefig(output_path, dpi=config['dpi'], bbox_inches='tight',
                        pad_inches=0.2, format='tiff')
            plt.close(fig)  # 关闭图形释放内存

            return output_path

        # 处理时间序列数据
        if is_temporal and result.ndim == 3:
            num_frames = result.shape[0]
            self.data_progress_signal.emit(0, num_frames)

            for frame_idx in range(num_frames):
                frame_data = result[frame_idx]

                # 确保数据是2D的
                if frame_data.ndim > 2:
                    frame_data = frame_data.squeeze()

                output_path = create_heatmap_frame(frame_data, frame_idx)
                created_files.append(output_path)
                self.data_progress_signal.emit(frame_idx + 1, num_frames)

        # 处理单帧数据
        else:
            num_frames = 1
            self.data_progress_signal.emit(0, 1)

            # 处理多维数据
            if result.ndim > 2:
                # 如果是时间序列但不需要分帧，取第一帧或平均值
                if result.ndim == 3 and not is_temporal:
                    frame_data = result[0] if result.shape[0] > 1 else result.mean(axis=0)
                else:
                    frame_data = result.squeeze()
            else:
                frame_data = result

            output_path = create_heatmap_frame(frame_data)
            created_files.append(output_path)
            self.data_progress_signal.emit(1, 1)

        self.data_progress_signal.emit(num_frames + 1, num_frames)
        logging.info(f'导出热图TIFF完成: {output_dir}, 共{len(created_files)}帧')
        return created_files

    @pyqtSlot(object, np.ndarray, float, bool, bool, float)
    def ROI_processed(self, data, mask, multiply_factor, is_crop=False, is_zoom=False, zoom_factor=1.0):
        """Run ROI processing with the shared cancellation and diagnostic contract."""
        try:
            return self._roi_processed_impl(data, mask, multiply_factor, is_crop, is_zoom, zoom_factor)
        except TaskCancelled:
            self.processing_cancelled_signal.emit()
            return False
        except Exception as exc:
            logging.exception("ROI 处理失败")
            self.processing_error_signal.emit(AppError(
                "ROI 处理失败", str(exc), stage="ROI 处理", severity="error",
                original=exc, details=format_exception_details(exc, "ROI 处理", data),
                context={"data": data},
            ))
            return False

    def _roi_processed_impl(self, data, mask, multiply_factor, is_crop=False, is_zoom=False, zoom_factor=1.0):
        """Apply an ROI transform; called only by ROI_processed."""
        self._check_cancelled()
        data_roi = []
        if isinstance(data, Data):
            aim_data = data.data_origin.copy()
            out_processed = data.parameters
        elif isinstance(data, ProcessedData):
            aim_data = data.data_processed.copy()
            out_processed = data.out_processed
        else:
            raise ValueError("不可能错误，数据类型有误")
        total_frames = data.timelength
        self.data_progress_signal.emit(0, total_frames)
        if is_crop:
            true_coords = np.argwhere(mask)
            # 计算边界框
            min_y, min_x = true_coords.min(axis=0)
            max_y, max_x = true_coords.max(axis=0)
            # 初始化裁剪后的数组
            processed_frame = np.zeros(((max_y - min_y + 1), (max_x - min_x + 1)))
            mask = mask[min_y:max_y + 1, min_x:max_x + 1]
            if len(aim_data.shape) == 2:
                processed_frame = aim_data[min_y:max_y + 1, min_x:max_x + 1]
                processed_frame = np.where(
                    mask,
                    processed_frame * multiply_factor,
                    processed_frame
                )
                if is_zoom:
                    processed_frame = zoom(processed_frame, zoom_factor, order=3)
                data_roi.append(processed_frame)
            else: # shape=3
                aim_data = aim_data[:,min_y:max_y + 1, min_x:max_x + 1]
                for i, frame in enumerate(aim_data):
                    self._check_cancelled()
                    processed_frame = np.where(
                        mask,
                        frame * multiply_factor,
                        frame
                    )
                    if is_zoom:
                        processed_frame = zoom(processed_frame, zoom_factor, order=3)
                    data_roi.append(processed_frame)
                    self.data_progress_signal.emit(i, total_frames)
        else:
            processed_frame = np.zeros(data.framesize)
            if len(aim_data.shape) == 2:
                processed_frame = np.where(
                    mask,
                    aim_data * multiply_factor,
                    aim_data
                )
                if is_zoom:
                    processed_frame = zoom(processed_frame, zoom_factor, order=3)
                data_roi.append(processed_frame)
            else:
                for i, frame in enumerate(aim_data):
                    self._check_cancelled()
                    processed_frame = np.where(
                        mask,
                        frame * multiply_factor,
                        frame
                    )
                    if is_zoom:
                        processed_frame = zoom(processed_frame, zoom_factor, order=3)
                    data_roi.append(processed_frame)
                    self.data_progress_signal.emit(i, total_frames)

        data_processed = np.squeeze(np.array(data_roi))
        out_processed = {k: v for k, v in out_processed.items() if k != 'unfolded_data'}
        self.processed_result.emit(ProcessedData(data.timestamp,
                                                 f'{data.name}@ROIed',
                                                 'Roi_applied',
                                                 time_point=data.time_point,
                                                 data_processed=data_processed,
                                                 out_processed={**out_processed,
                                                                "frame_size":f'{mask.shape[0]}×{mask.shape[1]}',
                                                                'ori_shape':data.datashape,
                                                                },
                                                 ROI_applied = True,
                                                 ROI_mask= mask,
                                                 ))
        self.data_progress_signal.emit(total_frames + 1, total_frames)



def _array_layout(shape, metadata=None, time_point=None):
    shape = tuple(int(value) for value in shape)
    ndim = len(shape)
    metadata = metadata or {}
    axes = str(metadata.get("scientific_axes") or metadata.get("source_axes") or "").upper()
    if len(axes) != ndim:
        axes = ""
    if "T" in axes:
        timelength = shape[axes.index("T")]
    elif ndim == 3:
        timelength = shape[0]
    else:
        timelength = 1
    h_axis = axes.find("H") if "H" in axes else axes.find("Y")
    w_axis = axes.find("W") if "W" in axes else axes.find("X")
    if h_axis >= 0 and w_axis >= 0:
        framesize = (shape[h_axis], shape[w_axis])
    elif ndim >= 2:
        framesize = (shape[-2], shape[-1])
    elif ndim == 1:
        framesize = (shape[0], 1)
    else:
        framesize = (1, 1)
    return timelength, framesize
@dataclass
class Data:
    """
    数据导入类型
    :cvar data_origin 原始导入数据（经过初步处理）
    :cvar time_point 时间点（已匹配时间尺度）
    :cvar format_import 导入格式
    :cvar image_import 原生成像数据
    :cvar parameters 其他参数
    :cvar  name 数据命名
    :cvar  out_processed 在0.11.9版本加入，解决自由调用时总报错的问题，同时减少代码冗余，于是加入了一个空的占位用
    :cvar  timestamp 时间戳（用于识别匹配数据流）
    :cvar  ROI_applied 是否应用ROI蒙版
    :cvar  history 历史保存（3组）
    :cvar  serial_number 序号
    内含参数还有：self.datashape；
        self.timelength；
        self.framesize；
        self.dtype；
        self.datamax；
        self.datamin；
    """
    data_origin: np.ndarray
    time_point: np.ndarray
    format_import: str
    image_import: np.ndarray
    parameters: dict = field(default_factory=dict)
    name: str = None
    out_processed : dict = field(init=False,default_factory=dict)
    timestamp: float = field(init=False, default_factory=time.time)
    ROI_applied: bool = field(init=False, default=False)
    history: ClassVar[deque] = deque(maxlen=10)
    serial_number: int = field(init=False)
    _counter: int = field(init=False, repr=False, default=0)
    _amend_counter: int = field(init=False, default=0)
    _data_origin_storage: object = field(init=False, repr=False, default=None)
    _image_import_storage: object = field(init=False, repr=False, default=None)

    def __getattribute__(self, name):
        if name in {"data_origin", "image_import"}:
            storage_name = f"_{name}_storage"
            values = object.__getattribute__(self, "__dict__")
            if storage_name in values and values[storage_name] is not None:
                return resolve_array(values[storage_name])
        return object.__getattribute__(self, name)

    def __setattr__(self, name, value):
        if name in {"data_origin", "image_import"}:
            object.__setattr__(self, f"_{name}_storage", value)
        object.__setattr__(self, name, value)
    def __post_init__(self):
        Data._counter += 1
        self.serial_number = Data._counter  # 生成序号
        self._recalculate()

        if self.name is None:
            self.name = f"{self.format_import}_{self.serial_number}"
        else:
            self.name = f"{self.name}_{self.serial_number}"
        Data.history.append(self._history_snapshot())  # 实例存储

    def _recalculate(self):
        array = self.data_origin
        self.datashape = array.shape
        self.timelength, self.framesize = _array_layout(
            self.datashape, self.parameters, self.time_point
        )
        self.datatype = array.dtype
        self.datamax = array.max()
        self.datamin = array.min()
        self.ndim = array.ndim

    def get_data_mean(self):
        return self.data_origin.mean()

    def get_data_std(self):
        return self.data_origin.std()

    def get_data_median(self):
        return np.median(self.data_origin)

    def update_params(self, **kwargs):
        """
        更新参数并根据物理量变化自动调整相关数据（如时间轴）
        :param kwargs:包含 fps, time_step, time_unit, space_step, space_unit 等
        :return: 实际发生变更的参数键列表
        """
        updated_keys = []

        # 1. 处理 FPS 变更导致的时间轴缩放
        if 'fps' in kwargs:
            new_fps = kwargs['fps']
            old_fps = self.parameters.get('fps')
            # 只有当新旧FPS都有效且不相等时才计算
            if new_fps is not None and old_fps is not None and old_fps != 0 and new_fps != old_fps:
                if self.time_point is not None:
                    # 公式：新时间 = 旧时间 * (旧FPS / 新FPS)
                    self.time_point = self.time_point * (old_fps / new_fps)
            self.parameters['fps'] = new_fps
            updated_keys.append('fps')

        # 2. 处理 Time Step 变更导致的时间轴缩放
        if 'time_step' in kwargs:
            new_step = kwargs['time_step']
            old_step = self.parameters.get('time_step')
            if new_step is not None and old_step is not None and old_step != 0 and new_step != old_step:
                if self.time_point is not None:
                    # 公式：新时间 = 旧时间 * (新步长 / 旧步长)
                    # 注意：原代码逻辑为 self.data.time_point / old * new
                    self.time_point = self.time_point * (new_step / old_step)
            self.parameters['time_step'] = new_step
            updated_keys.append('time_step')

        # 3. 处理其他通用参数 (time_unit, space_step, space_unit 等)
        for key, value in kwargs.items():
            if key not in ['fps', 'time_step']:  # 已经处理过的跳过
                if self.parameters.get(key) != value:
                    self.parameters[key] = value
                    updated_keys.append(key)

        # 更新历史记录
        if updated_keys:
            self._update_history()

        return updated_keys

    def update_data(self, **kwargs):
        """数据更新（目前仅仅坏点修复需要）"""
        Data._amend_counter += 1
        if 'data_origin' in kwargs:
            return self._create_new_instance(**kwargs)

        # 更新其他字段
        for key, value in kwargs.items():
            setattr(self, key, value)

        # 更新历史记录
        self._update_history()
        logging.info("生成了新的数据，请注意查看")
        return self

    def apply_ROI(self, mask: np.ndarray):
        """设置 ROI 蒙版"""
        # 验证蒙版形状
        if mask is None:
            raise ValueError("无效蒙版")
        if mask.shape != self.datashape:
            raise ValueError(f"蒙版形状 {mask.shape} 与图像形状 {self.datashape} 不匹配")

        self.ROI_mask = mask

        # 根据蒙版类型应用不同的处理
        if self.ROI_mask.dtype == bool:
            # 布尔蒙版：将非 ROI 区域置零
            if self.ndim == 3:
                self.ROI_mask = copy.deepcopy(self.data_processed)
                for every_data in self.ROI_mask:
                    every_data[~mask] = self.datamin
            elif self.ndim == 2:
                self.ROI_mask = copy.deepcopy(self.data_processed)
                self.ROI_mask[~mask] = self.datamin
            else:
                raise ValueError("该数据无法应用ROI蒙版")
        else:
            # 数值蒙版：应用乘法操作
            if self.ndim >= 2:
                self.data_processed_ROI = self.data_processed * self.ROI_mask
            else:
                raise ValueError("该数据无法应用ROI蒙版")

        self.ROI_applied = True

    def _create_new_instance(self, **kwargs) -> 'Data':
        """创建新实例（当data_origin变更时）"""
        # 获取当前所有字段值
        current_values = {f.name: getattr(self, f.name) for f in fields(self) if f.init}

        # 应用更新
        current_values.update(kwargs)

        # 创建新实例（会分配新序列号）
        new_instance = Data(
            data_origin=current_values['data_origin'],
            time_point=current_values['time_point'],
            format_import=current_values['format_import'],
            image_import=current_values['image_import'],
            parameters=current_values.get('parameters'),
            name=f"{current_values.get('name')}@"
        )
        return new_instance

    def trim_time(self, start_idx: int, end_idx: int):
        """
        沿时间轴裁剪数据 (原地修改)
        :param start_idx: 起始帧索引 (包含)
        :param end_idx: 结束帧索引 (不包含，即 Python 切片逻辑 [start:end])
        """
        if self.ndim != 3:
            raise ValueError("当前数据不支持时间裁剪")

        # 1. 边界检查
        total = self.timelength
        if start_idx < 0: start_idx = 0
        if end_idx > total: end_idx = total
        if start_idx >= end_idx:
            raise ValueError("无效的裁剪区间")

        # 2. 裁剪原始数据 (image_backup)
        # 注意：这里我们使用 copy() 确保释放掉不需要的内存，否则 numpy 可能会持有原大数组的视图
        self.data_origin = self.data_origin[start_idx:end_idx].copy()

        # 4. 裁剪时间点数据 (如果有)
        if self.time_point is not None and len(self.time_point) == total:
            self.time_point = self.time_point[start_idx:end_idx] - self.time_point[start_idx]

        # 5. 更新元数据
        self.datashape = self.data_origin.shape
        self.timelength = self.datashape[0]

        # 6. (可选) 重新计算最大最小值，防止裁剪后亮度范围变化导致显示不准
        self.datamin = self.data_origin.min()
        self.datamax = self.data_origin.max()
        self.name = f'{self.name}@trimmed'
        self.format_import = f'{self.format_import}@trimmed'

        logging.info(f"Data数据已裁剪: 帧数变为 {self.timelength}")
        self._update_history()

    def __setitem__(self, key, value):
        """字典式赋值支持"""
        valid_keys = [f.name for f in fields(self)]
        if key not in valid_keys:
            raise KeyError(f"Invalid field: {key}. Valid fields: {valid_keys}")

        setattr(self, key, value)

        # 特殊字段处理
        # 特殊处理：如果更新data_origin，创建新实例
        if key == 'data_origin':
            return self._create_new_instance(data_origin=value)

        setattr(self, key, value)

        # 更新历史记录中的当前实例
        self._update_history()

    def __repr__(self):
        return (
            f"Data<{self.name}: '{self.timestamp}'>' "
            f"({self.format_import}) | "
            f"Shape: {self.datashape} | "
            f"Range: [{self.datamin:.2f}, {self.datamax:.2f}] | "
            f"Time: {self.time_point[0] if self.time_point.size > 0 else 'N/A'}>"
        )

    @classmethod
    def find_history(cls, timestamp: float) -> Optional['ProcessedData']:
        """根据时间戳查找历史记录中的特定数据"""
        # 使用生成器表达式高效查找
        try:
            return next(
                (data for data in cls.history if abs(data.timestamp - timestamp) < 1e-6),
                None
            )
        except Exception as e:
            print(f"查找历史记录时出错: {e}")
            return None

    @classmethod
    def get_history_by_serial(cls, serial_number: int) -> Optional['Data']:
        """根据序列号获取历史记录并调整位置"""
        for i, record in enumerate(cls.history):
            if record.serial_number == serial_number:
                # 移除并重新添加以调整位置
                cls.history.remove(record)
                cls.history.append(record)
                return copy.deepcopy(record)
        return None

    # @classmethod
    # def get_history_by_timestamp(cls, timestamp: float) -> Optional['Data']:
    #     """根据时间戳获取历史记录并调整位置"""
    #     for i, record in enumerate(cls.history):
    #         if abs(record.timestamp - timestamp) < 1e-6:  # 浮点数精度处理
    #             # 移除并重新添加以调整位置
    #             cls.history.remove(record)
    #             cls.history.append(record)
    #             return copy.deepcopy(record)
    #     return None

    @classmethod
    def get_history_list(cls) -> list:
        """获取当前历史记录列表（按从旧到新排序）"""
        history_list = list(cls.history)
        history_list.reverse()
        return history_list

    @classmethod
    def get_history_serial_numbers(cls) -> list:
        """获取历史记录的序列号列表（按从旧到新排序）"""
        return [record.serial_number for record in cls.history]

    @classmethod
    def get_history_timestamps(cls) -> List[float]:
        """获取历史记录的时间戳列表（按从旧到新排序）"""
        return [record.timestamp for record in cls.history]

    @classmethod
    def get_history_summary(cls) -> str:
        """获取历史记录的摘要信息"""
        summary = []
        for record in cls.history:
            summary.append(
                f"#{record.serial_number}: {record.name} "
                f"({time.strftime('%H:%M:%S', time.localtime(record.timestamp))})"
            )
        return " | ".join(summary)

    def _history_snapshot(self):
        """Create a history snapshot that caches large primary arrays by reference."""
        snapshot = copy.copy(self)
        store = get_array_store()
        owner_id = str(self.timestamp)
        data_origin = self.data_origin
        image_import = self.image_import
        if should_cache_array(data_origin, store.config):
            cached_origin = cache_array_or_keep_memory(store, data_origin, owner_id, "data_origin")
            object.__setattr__(snapshot, "_data_origin_storage", cached_origin)
            if isinstance(cached_origin, ArrayRef):
                object.__setattr__(snapshot, "data_origin", None)
        else:
            object.__setattr__(snapshot, "_data_origin_storage", data_origin)
        # image_import is display-oriented preview data; keep it in memory to avoid UI stalls from disk I/O.
        object.__setattr__(snapshot, "_image_import_storage", image_import)
        return snapshot
    def _update_history(self):
        """更新历史记录中的当前实例"""
        # 查找历史记录中的当前实例
        for i, record in enumerate(Data.history):
            if record.serial_number == self.serial_number:
                # 更新历史记录中的实例
                Data.history[i] = self._history_snapshot()
                break
        return None

    @classmethod
    def clear_history(cls, remove_cache: bool = True):
        """清空所有历史记录，并按需删除历史引用的缓存文件。"""
        if remove_cache:
            store = get_array_store()
            for ref in collect_array_refs(cls.history):
                store.delete_ref(ref)
        cls.history.clear()


@dataclass
class ProcessedData:
    """
    经过处理的数据
    :cvar timestamp_inherited 处理前数据的时间戳: float
    :cvar name 命名（需要更新）: str
    :cvar type_processed 处理类型（最后） : str
    :cvar time_point 时间点: np.ndarray
    :cvar data_processed 处理出来的数据（此处存放尤指具有时空尺度的核心数据）: np.ndarray = None
    :cvar out_processed 其他处理出来的数据（比如拟合得到的参数，二维序列等等）: dict = None
    :cvar parameters 在0.11.9版本加入，解决自由调用时总报错的问题，同时减少代码冗余，于是加入了一个空的占位用
    :cvar timestamp 新数据时间戳
    :cvar ROI_applied 是否应用ROI蒙版
    :cvar history 历史，无限保留，考虑和绘图挂钩: ClassVar[Dict[str, 'ProcessedData']] = {}
    :return 包含name type datashape datamin-max 这四项信息的str
    """
    timestamp_inherited: float
    name: str
    type_processed: str
    time_point: np.ndarray = None
    data_processed: np.ndarray = None
    out_processed: dict = field(default_factory=dict)
    parameters: dict = field(init=False, default_factory=dict)
    timestamp: float = field(init=False, default_factory=time.time)
    ROI_applied: bool = False
    ROI_mask: np.ndarray = None
    serial_number: int = field(init=False)
    _counter: int = field(init=False, repr=False, default=0)
    history: ClassVar[deque] = deque(maxlen=30)
    _data_processed_storage: object = field(init=False, repr=False, default=None)

    def __getattribute__(self, name):
        if name == "data_processed":
            values = object.__getattribute__(self, "__dict__")
            if "_data_processed_storage" in values and values["_data_processed_storage"] is not None:
                return resolve_array(values["_data_processed_storage"])
        return object.__getattribute__(self, name)

    def __setattr__(self, name, value):
        if name == "data_processed":
            object.__setattr__(self, "_data_processed_storage", value)
        object.__setattr__(self, name, value)

    def out_processed_array(self, key: str):
        return resolve_array(self.out_processed[key])
    def __post_init__(self):
        Data._counter += 1
        self.serial_number = Data._counter  # 生成序号

        if self.data_processed is not None:
            array = self.data_processed
            self.datashape = array.shape
            self.timelength, self.framesize = _array_layout(
                self.datashape, self.out_processed, self.time_point
            )
            self.datamin = array.min()
            self.datamax = array.max()
            self.datatype = array.dtype
            self.datamean = array.mean()
            self.ndim = array.ndim

        # 加序列号
        self.name = f"{self.name}-{self.serial_number}"
        # 添加到历史记录
        ProcessedData.history.append(self._history_snapshot())

    def update_params(self, **kwargs):
        """
        更新 out_processed 中的参数
        """
        updated_keys = []

        # 1. 处理 FPS
        if 'fps' in kwargs:
            new_fps = kwargs['fps']
            old_fps = self.out_processed.get('fps')
            if new_fps is not None and old_fps is not None and old_fps != 0 and new_fps != old_fps:
                if self.time_point is not None:
                    self.time_point = self.time_point * (old_fps / new_fps)
            self.out_processed['fps'] = new_fps
            updated_keys.append('fps')

        # 2. 处理 Time Step
        if 'time_step' in kwargs:
            new_step = kwargs['time_step']
            old_step = self.out_processed.get('time_step')
            if new_step is not None and old_step is not None and old_step != 0 and new_step != old_step:
                if self.time_point is not None:
                    self.time_point = self.time_point * (new_step / old_step)
            self.out_processed['time_step'] = new_step
            updated_keys.append('time_step')

        # 3. 其他参数
        for key, value in kwargs.items():
            if key not in ['fps', 'time_step']:
                if self.out_processed.get(key) != value:
                    self.out_processed[key] = value
                    updated_keys.append(key)

        # 更新历史
        if updated_keys:
            self._update_history()

        return updated_keys

    def apply_ROI(self, mask: np.ndarray):
        """设置 ROI 蒙版"""
        # 验证蒙版形状
        if mask is None:
            raise ValueError("无效蒙版")
        if mask.shape != self.datashape:
            raise ValueError(f"蒙版形状 {mask.shape} 与图像形状 {self.datashape} 不匹配")

        self.ROI_mask = mask

        # 根据蒙版类型应用不同的处理
        if self.ROI_mask.dtype == bool:
            # 布尔蒙版：将非 ROI 区域置零
            if self.ndim == 3:
                self.ROI_mask = copy.deepcopy(self.data_processed)
                for every_data in self.ROI_mask:
                    every_data[~mask] = self.datamin
            elif self.ndim == 2:
                self.ROI_mask = copy.deepcopy(self.data_processed)
                self.ROI_mask[~mask] = self.datamin
            else:
                raise ValueError("该数据无法应用ROI蒙版")
        else:
            # 数值蒙版：应用乘法操作
            if self.ndim >= 2:
                self.data_processed = self.data_processed * self.ROI_mask
            else:
                raise ValueError("该数据无法应用ROI蒙版")

        self.ROI_applied = True

    @classmethod
    def find_history(cls, timestamp: float) -> Optional['ProcessedData']:
        """
        根据时间戳查找历史记录中的特定数据
        :param timestamp: float
        """
        # 使用生成器表达式高效查找
        try:
            return next(
                (data for data in cls.history if abs(data.timestamp - timestamp) < 1e-6),
                None
            )
        except Exception as e:
            print(f"查找历史记录时出错: {e}")
            return None

    # @classmethod
    # def remove_from_history(cls, name: str):
    #     """从历史记录中删除指定名称的处理数据"""
    #     if name in cls.history:
    #         del cls.history[name]
    #
    @classmethod
    def clear_history(cls, remove_cache: bool = True):
        """清空所有历史记录，并按需删除历史引用的缓存文件。"""
        if remove_cache:
            store = get_array_store()
            for ref in collect_array_refs(cls.history):
                store.delete_ref(ref)
        cls.history.clear()

    @classmethod
    def get_history_list(cls) -> list:
        """获取当前历史记录列表（按从新到旧排序）"""
        history_list = list(cls.history)
        history_list.reverse()
        return history_list

    def _history_snapshot(self):
        """Create a history snapshot that caches large result arrays by reference."""
        snapshot = copy.copy(self)
        store = get_array_store()
        owner_id = str(self.timestamp)
        data_processed = self.data_processed
        if should_cache_array(data_processed, store.config):
            cached_processed = cache_array_or_keep_memory(store, data_processed, owner_id, "data_processed")
            object.__setattr__(snapshot, "_data_processed_storage", cached_processed)
            if isinstance(cached_processed, ArrayRef):
                object.__setattr__(snapshot, "data_processed", None)
        else:
            object.__setattr__(snapshot, "_data_processed_storage", data_processed)
        try:
            snapshot.out_processed = cache_large_arrays_in_mapping(self.out_processed, store, owner_id=owner_id)
        except Exception:
            logging.exception("out_processed 缓存写入失败，保留内存数据: %s", self.name)
            snapshot.out_processed = self.out_processed
        return snapshot
    def _update_history(self):
        """更新历史记录中的当前实例"""
        # 查找历史记录中的当前实例
        for i, record in enumerate(ProcessedData.history):
            if record.serial_number == self.serial_number:
                # 更新历史记录中的实例
                ProcessedData.history[i] = self._history_snapshot()
                break
        return None

    def trim_time(self, start_idx: int, end_idx: int):
        """
        沿时间轴裁剪数据 (原地修改)
        :param start_idx: 起始帧索引 (包含)
        :param end_idx: 结束帧索引 (不包含，即 Python 切片逻辑 [start:end])
        """
        if self.ndim != 3:
            raise ValueError("当前数据不支持时间裁剪")

        # 1. 边界检查
        total = self.timelength
        if start_idx < 0: start_idx = 0
        if end_idx > total: end_idx = total
        if start_idx >= end_idx:
            raise ValueError("无效的裁剪区间")

        # 2. 裁剪原始数据 (image_backup)
        self.data_processed = self.data_processed[start_idx:end_idx].copy()

        # 4. 裁剪时间点数据 (如果有)
        if self.time_point is not None and len(self.time_point) == total:
            self.time_point = self.time_point[start_idx:end_idx] - self.time_point[start_idx]

        # 如果有多个存储结果，都需要裁剪
        if hasattr(self, 'out_processed') and isinstance(self.out_processed, dict):
            for key, val in self.out_processed.items():
                if isinstance(val, np.ndarray) and len(val) > end_idx and len(val) == self.timelength:
                    self.out_processed[key] = val[start_idx:end_idx]

        # 5. 更新元数据
        self.datashape = self.data_processed.shape
        self.timelength = self.datashape[0]

        # 6. (可选) 重新计算最大最小值，防止裁剪后亮度范围变化导致显示不准
        self.datamin = self.data_processed.min()
        self.datamax = self.data_processed.max()
        self.name = f'{self.name}@trimmed'
        self.type_processed = f'{self.type_processed}@trimmed'

        logging.info(f"Processed数据已裁剪: 帧数变为 {self.timelength}")

    def upgrade_processed(self, key: str) -> Optional['ProcessedData']:
        """
        从 out_processed 字典中提取指定键的数据，创建一个新的 ProcessedData 实例。
        新实例会自动继承时间戳和时间点，并加入历史记录。

        :param key: out_processed 中的键名
        :return: 新创建的 ProcessedData 实例，如果 key 不存在或数据无效则返回 None
        """
        # 1. 检查 out_processed 是否存在以及键是否存在
        if self.out_processed is None or key not in self.out_processed:
            logging.warning(f"提取失败: 键 '{key}' 不在 {self.name} 的输出结果中。")
            return None

        # 2. 获取目标数据
        target_data = self.out_processed[key]

        # 3. 数据类型检查与转换
        if not isinstance(target_data, np.ndarray):
            # 尝试转换列表为数组，如果是其他类型则报错
            if isinstance(target_data, list):
                target_data = np.array(target_data)
            else:
                logging.warning(f"提取失败: '{key}' 的数据类型为 {type(target_data)}，需要 np.ndarray。")
                return None

        # 4. 准备新实例的参数
        # 命名规则: 原名-键名
        new_name = f"{self.name}-{key}"
        # 类型标记
        new_type = f"extracted_{key}"

        # 继承时间点 (深拷贝防止后续修改影响原数据)
        if target_data.ndim == 3:
            new_time_point = self.time_point.copy() if self.time_point is not None else None
        else:
            new_time_point = None

        try:
            # 5. 实例化新对象
            # 注意：实例化后会自动调用 __post_init__，
            # 从而自动计算 shape, min, max, framesize 并添加到 history 中
            new_instance = ProcessedData(
                timestamp_inherited=self.timestamp_inherited,
                name=new_name,
                type_processed=new_type,
                time_point=new_time_point,
                data_processed=target_data.copy(),  # 深拷贝数据，保证独立性
                out_processed={**{k:self.out_processed.get(k) for k in self.out_processed if k in ['fps', 'time_step','time_unit','duration','space_step','space_unit']}}  # 新实例只继承基础参数
            )

            logging.info(f"已从 {self.name} 提取 '{key}' 生成新数据实例: {new_name}")
            return new_instance

        except Exception as e:
            logging.error(f"创建衍生数据实例时发生错误: {e}")
            return None

    def __repr__(self):
        return (
            f"ProcessedData<{self.name} | "
            f"Type: {self.type_processed} | "
            f"Shape: {self.datashape} | "
            f"Range: [{self.datamin:.2f}, {self.datamax:.2f}]>"
        )


@dataclass
class ImagingData:
    """
    图像显示类型
    :cvar timestamp_inherited:
    :cvar image_backup:
    :cvar image
    """
    timestamp_inherited: float
    image_backup: np.ndarray = None  # 原始数据
    image_data: np.ndarray = None  # 归一，放宽，整数化后的数据 （最终显示）
    image_type: str = None
    colormode: str = None  # 色彩模式，目前在sub类中实现
    canvas_num: int = field(default=0)
    fps: int = field(default=None)
    is_temporary: bool = field(init=False, default=False)
    time_point: np.ndarray = None
    parent_data = None # 弱引用实现的直接父类调用0.12加入
    timestamp: float = field(init=False, default_factory=time.time)

    def __post_init__(self):
        self.imageshape = self.image_backup.shape
        self.ndim = self.image_backup.ndim
        if getattr(self, "display_source", None) is not None:
            self.totalframes = self.display_source.frame_count
            self.framesize = self.display_source.frame_shape
            self.is_temporary = self.totalframes > 1
        else:
            self.totalframes = self.imageshape[0] if self.ndim == 3 else 1
            self.framesize = (self.imageshape[1], self.imageshape[2]) if self.ndim == 3 else (self.imageshape[0], self.imageshape[1])
            self.is_temporary = self.ndim == 3
        preview = self.display_source.get_frame(0) if getattr(self, "display_source", None) is not None else self.image_backup
        preview_for_stats = np.abs(preview) if np.iscomplexobj(preview) else preview
        finite_preview = preview_for_stats[np.isfinite(preview_for_stats)]
        if finite_preview.size:
            self.imagemin = finite_preview.min()
            self.imagemax = finite_preview.max()
        else:
            self.imagemin = 0
            self.imagemax = 0
        self.datatype = self.image_backup.dtype
        self.image_data = self.to_uint8(preview)
        self.ROI_mask = None
        self.ROI_applied = False

    @classmethod
    def create_image(cls, data_obj: Union['Data', 'ProcessedData'], *arg: str) -> 'ImagingData':
        """初始化ImagingData"""

        instance = cls.__new__(cls)
        instance.timestamp = time.time()
        instance.parent_data = weakref.ref(data_obj)
        # 设置图像数据
        if isinstance(data_obj, Data):
            instance.display_source = DisplaySourceFactory.from_data(data_obj)
            instance.image_backup = instance.display_source.array
            instance.timestamp_inherited = data_obj.timestamp
            instance.source_type = "Data"
            instance.source_name = data_obj.name
            instance.source_format = data_obj.format_import
            # instance.fps = getattr(data_obj, 'parameters', {}).get('fps', 10) # 优雅
            instance.fps = (getattr(data_obj, 'parameters') or {}).get('fps')
            instance.image_type = 'from_data'
        elif isinstance(data_obj, ProcessedData):
            if arg:
                instance.display_source = DisplaySourceFactory.from_data(data_obj, key=arg[0])
                instance.image_backup = instance.display_source.array
            else:
                instance.display_source = DisplaySourceFactory.from_data(data_obj)
                instance.image_backup = instance.display_source.array
            instance.timestamp_inherited = data_obj.timestamp
            instance.source_type = "ProcessedData"
            instance.source_name = data_obj.name
            instance.source_format = data_obj.type_processed
            instance.fps = (getattr(data_obj, 'out_processed') or {}).get('fps')
            if data_obj.ROI_applied:
                instance.image_type = 'from_ROIed'
            else:
                instance.image_type = 'from_processed'

        instance.time_point = data_obj.time_point.copy() if data_obj.time_point is not None else None
        # 调用后初始化
        instance.__post_init__()
        return instance

    def apply_ROI(self, mask: np.ndarray):
        """设置 ROI 蒙版"""
        # 验证蒙版形状
        if mask is None:
            raise ValueError("无效蒙版")
        if mask.shape != self.imageshape:
            raise ValueError(f"蒙版形状 {mask.shape} 与图像形状 {self.imageshape} 不匹配")

        self.ROI_mask = mask

        # 根据蒙版类型应用不同的处理
        if self.ROI_mask.dtype == bool:
            # 布尔蒙版：将非 ROI 区域置零
            if self.ndim == 3:
                self.ROI_mask = copy.deepcopy(self.image_backup)
                for every_data in self.ROI_mask:
                    every_data[~mask] = self.imagemin
            if self.ndim == 2:
                self.ROI_mask = copy.deepcopy(self.image_backup)
                self.ROI_mask[~mask] = self.imagemin
            else:
                raise ValueError("该数据无法应用ROI蒙版")
        else:
            # 数值蒙版：应用乘法操作
            if self.ndim >= 2:
                self.image_data = self.image_backup * self.ROI_mask
            else:
                raise ValueError("该数据无法应用ROI蒙版")

        self.ROI_applied = True

    def to_uint8(self, data=None):
        """归一化和数字类型调整"""
        if data is None:
            data = self.image_backup
        if data.dtype == np.uint8 and np.max(data) == 255:
            return data.copy()

        # 计算数组的最小值和最大值
        min_val = np.min(data)
        max_val = np.max(data)

        # if self.source_format == "ROI_stft" or self.source_format == "ROI_cwt":
        #     return ((data - np.min(data))/(np.max(data)- np.min(data))*255).astype(np.uint8)
        result = ((data - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        # 通用线性变换公式
        # 使用64位浮点保证精度，避免中间步骤溢出
        # scaled = (data.astype(np.float64) - min_val) * (255.0 / (max_val - min_val))
        #
        # # 四舍五入并确保在[0,255]范围内
        # result = np.clip(np.round(scaled), 0, 255).astype(np.uint8)
        return result

    def update_params(self, **kwargs):
        """
        更新显示相关的参数
        """
        updated_keys = []

        # 更新 FPS
        if 'fps' in kwargs and kwargs['fps'] != self.fps:
            self.fps = kwargs['fps']
            updated_keys.append('fps')

        # 同步时间点 (通常计算好后传入)
        if 'time_point' in kwargs:
            # 直接替换，不进行计算，因为计算应该在源数据层(Data/ProcessedData)完成
            self.time_point = kwargs['time_point']
            # time_point 不计入 updated_keys 以避免日志冗余，或者根据需要添加

        return updated_keys

    def trim_time(self, start_idx: int, end_idx: int):
        """
        沿时间轴裁剪数据 (原地修改)
        :param start_idx: 起始帧索引 (包含)
        :param end_idx: 结束帧索引 (不包含，即 Python 切片逻辑 [start:end])
        """
        if not self.is_temporary or self.ndim != 3:
            raise ValueError("当前数据不支持时间裁剪")

        # 1. 边界检查
        total = self.totalframes
        if start_idx < 0: start_idx = 0
        if end_idx > total: end_idx = total
        if start_idx >= end_idx:
            raise ValueError("无效的裁剪区间")

        # 2. 裁剪原始数据 (image_backup)
        # 注意：这里我们使用 copy() 确保释放掉不需要的内存，否则 numpy 可能会持有原大数组的视图
        self.image_backup = self.image_backup[start_idx:end_idx].copy()

        # 3. 裁剪显示数据 (image_data)
        self.image_data = self.image_data[start_idx:end_idx].copy()

        # 4. 裁剪时间点数据 (如果有)
        if self.time_point is not None and len(self.time_point) == total:
            self.time_point = self.time_point[start_idx:end_idx] - self.time_point[start_idx]

        # 5. 更新元数据
        self.imageshape = self.image_data.shape
        self.totalframes = self.imageshape[0]

        # 6. (可选) 重新计算最大最小值，防止裁剪后亮度范围变化导致显示不准
        self.imagemin = self.image_backup.min()
        self.imagemax = self.image_backup.max()
        self.image_type = f'{self.image_type}@trimmed'

        logging.info(f"数据已裁剪: 帧数变为 {self.totalframes}")

    def __repr__(self):
        return (
            f"ImageData<Source: {self.source_type}:{self.source_name} "
            f"| Shape: {self.imageshape} | "
            f"Range: [{self.imagemin:.2f}, {self.imagemax:.2f}]>"
        )


class ColorMapManager:
    """伪彩色映射管理器"""

    def __init__(self):
        self.colormaps = {
            "jet": self.jet_colormap,
            "hot": self.hot_colormap,
            # "Cool": self.cool_colormap,
            # "Spring": self.spring_colormap,
            # "Summer": self.summer_colormap,
            # "Autumn": self.autumn_colormap,
            # "Winter": self.winter_colormap,
            # "Bone": self.bone_colormap,
            # "Copper": self.copper_colormap,
            # "Greys": self.greys_colormap,
            # "Viridis": self.viridis_colormap,
            # "Plasma": self.plasma_colormap,
            # "Inferno": self.inferno_colormap,
            # "Magma": self.magma_colormap,
            # "Cividis": self.cividis_colormap,
            # "Rainbow": self.rainbow_colormap,
            # "Turbo": self.turbo_colormap
        }

        # 创建Matplotlib兼容的colormap
        self.matplotlib_cmaps = {
            "jet": cm.jet,
            "hot": cm.hot,
            "cool": cm.cool,
            "spring": cm.spring,
            "summer": cm.summer,
            "autumn": cm.autumn,
            "winter": cm.winter,
            "bone": cm.bone,
            "copper": cm.copper,
            "gray": cm.gray,
            "viridis": cm.viridis,
            "plasma": cm.plasma,
            "inferno": cm.inferno,
            "magma": cm.magma,
            "cividis": cm.cividis,
            "turbo": cm.turbo,
            'CMRmap': cm.CMRmap,
            'gnuplot2': cm.gnuplot2,
            "Rainbow*": self.create_rainbow_cmap(),
        }

    def get_colormap_names(self):
        """获取所有可用的colormap名称"""
        return list(self.matplotlib_cmaps.keys())  # 暂时用Matplotlib

    def apply_colormap(self, image_data, colormap_name, min_val=None, max_val=None):
        """应用伪彩色映射到图像数据"""
        if colormap_name not in self.matplotlib_cmaps:
            colormap_name = "jet"  # 默认使用jet

        cmap = self.matplotlib_cmaps[colormap_name]

        # 归一化数据
        if min_val is None:
            min_val = np.min(image_data)
        if max_val is None:
            max_val = np.max(image_data)

        # 避免除以零
        if min_val == max_val:
            normalized = np.zeros_like(image_data)
        else:
            normalized = (image_data - min_val) / (max_val - min_val)
            normalized = np.clip(normalized, 0, 1)

        # 应用colormap
        colored = (cmap(normalized) * 255).astype(np.uint8)
        return colored

    def create_rainbow_cmap(self):
        """创建自定义彩虹colormap"""
        cdict = {
            'red': [(0.0, 1.0, 1.0),
                    (0.15, 0.0, 0.0),
                    (0.3, 0.0, 0.0),
                    (0.45, 0.0, 0.0),
                    (0.6, 1.0, 1.0),
                    (0.75, 1.0, 1.0),
                    (1.0, 1.0, 1.0)],
            'green': [(0.0, 0.0, 0.0),
                      (0.15, 0.0, 0.0),
                      (0.3, 1.0, 1.0),
                      (0.45, 1.0, 1.0),
                      (0.6, 1.0, 1.0),
                      (0.75, 0.0, 0.0),
                      (1.0, 0.0, 0.0)],
            'blue': [(0.0, 0.0, 0.0),
                     (0.15, 1.0, 1.0),
                     (0.3, 1.0, 1.0),
                     (0.45, 0.0, 0.0),
                     (0.6, 0.0, 0.0),
                     (0.75, 0.0, 0.0),
                     (1.0, 1.0, 1.0)]
        }
        return LinearSegmentedColormap('Rainbow', cdict)

    # 以下是各种colormap的实现（保留作为参考）
    def jet_colormap(self, value):
        """Jet colormap实现"""
        if value < 0.125:
            r = 0
            g = 0
            b = 0.5 + 4 * value
        elif value < 0.375:
            r = 0
            g = 4 * (value - 0.125)
            b = 1
        elif value < 0.625:
            r = 4 * (value - 0.375)
            g = 1
            b = 1 - 4 * (value - 0.375)
        elif value < 0.875:
            r = 1
            g = 1 - 4 * (value - 0.625)
            b = 0
        else:
            r = max(1 - 4 * (value - 0.875), 0)
            g = 0
            b = 0
        return (int(r * 255), int(g * 255), int(b * 255))

    def hot_colormap(self, value):
        """Hot colormap实现"""
        r = min(3 * value, 1.0)
        g = min(3 * value - 1, 1.0) if value > 1 / 3 else 0
        b = min(3 * value - 2, 1.0) if value > 2 / 3 else 0
        return (int(r * 255), int(g * 255), int(b * 255))

    # 其他colormap实现类似，这里省略以节省空间...
    # 实际使用中我们使用matplotlib的实现


class PublicEasyMethod:
    """各种简单的算法v.11.10加入"""
    @staticmethod
    def quick_mask(framesize,**kwargs):
        """快速选取简易ROI"""
        h, w = framesize
        y, x = kwargs.get('center',(h//2,w//2))
        shape = kwargs.get('shape','circle')
        size = kwargs.get('size',5)

        if shape == 'square':
            y_min = max(0, y - (size - 1) // 2)
            y_max = min(h, y_min + size)
            x_min = max(0, x - (size - 1) // 2)
            x_max = min(w, x_min + size)
            mask = np.zeros((h, w), dtype=bool)
            mask[y_min:y_max, x_min:x_max] = True
        elif shape == 'circle':  # circle
            yy, xx = np.ogrid[:h, :w]
            mask = (yy - y) ** 2 + (xx - x) ** 2 <= (size-1) ** 2
        elif shape == 'custom':  # 留给绘制roi
            pass
        return mask
