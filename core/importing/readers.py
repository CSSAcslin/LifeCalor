from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tifffile

from dataio import copy_npy_to_memory
from .model import ImportProbe, ImportRequest, ImportedPayload, Importer


def _emit(progress, current, total, message):
    if progress is not None:
        progress(int(current), int(total), message)


def _time_axis(shape, axes, options):
    frame_count = int(shape[axes.index("T")]) if "T" in axes else 1
    if options.get("time_basis") == "fps":
        fps = float(options.get("fps", 0))
        if fps <= 0:
            raise ValueError("使用 FPS 时间基准时，FPS 必须大于 0")
        return np.arange(frame_count, dtype=np.float64) / fps
    step = float(options.get("time_step", 1.0))
    return np.arange(frame_count, dtype=np.float64) * step


class NpyImporter(Importer):
    format_id = "npy"
    extensions = (".npy",)

    def probe(self, request):
        source = np.load(request.path, mmap_mode="r", allow_pickle=False)
        if source.ndim not in (2, 3):
            raise ValueError(f"NPY 数据必须是二维或三维数组，当前 shape={source.shape}")
        axes = "YX" if source.ndim == 2 else "TYX"
        return ImportProbe(self.format_id, tuple(source.shape), str(source.dtype), axes)

    def read(self, request, progress=None, token=None):
        probe = self.probe(request)
        array = copy_npy_to_memory(request.path, progress=progress, token=token, message="正在读取 NPY 数据")
        params = dict(request.options)
        params.update({
            "file_path": str(request.path), "source_shape": tuple(array.shape),
            "source_dtype": str(array.dtype), "source_axes": probe.axes,
            "display_axes": probe.axes, "external_npy": True,
        })
        return ImportedPayload(array, array, _time_axis(array.shape, probe.axes, params), "npy", params, request.path.name)


class TiffImporter(Importer):
    format_id = "tiff"
    extensions = (".tif", ".tiff")

    @staticmethod
    def _axes(series, shape, photometric):
        axes = (getattr(series, "axes", "") or "").upper()
        if len(shape) == 2:
            return "YX"
        is_color = photometric == "RGB" and shape[-1] in (3, 4)
        if len(shape) == 3 and is_color:
            return "YXC"
        if len(shape) == 3:
            return "TYX"
        if len(shape) == 4 and shape[-1] in (3, 4):
            return "TYXC"
        if len(axes) == len(shape) and "Y" in axes and "X" in axes:
            axes = axes.replace("S", "C")
            return axes
        return "?" * len(shape)

    def probe(self, request):
        with tifffile.TiffFile(request.path) as tf:
            series = tf.series[0]
            page = series.pages[0]
            photo = getattr(getattr(page, "photometric", None), "name", str(getattr(page, "photometric", ""))).upper()
            axes = self._axes(series, tuple(series.shape), photo)
            if axes == "TYXC":
                color_mode = "color_stack"
            elif axes.endswith("C") or photo == "RGB":
                color_mode = "rgb"
            elif photo == "PALETTE":
                color_mode = "palette"
            else:
                color_mode = "grayscale"
            description = getattr(page, "description", None)
            metadata = {"photometric": photo, "page_count": len(tf.pages)}
            if description:
                try:
                    metadata["description"] = json.loads(description)
                except (TypeError, ValueError, json.JSONDecodeError):
                    metadata["description"] = description[:500]
            return ImportProbe(self.format_id, tuple(series.shape), str(series.dtype), axes, color_mode, metadata)

    @staticmethod
    def _rgb_uint8(array):
        rgb = np.asarray(array)[..., :3]
        if rgb.dtype == np.uint8:
            return rgb.copy()
        values = rgb.astype(np.float32)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return np.zeros(rgb.shape, dtype=np.uint8)
        low, high = float(finite.min()), float(finite.max())
        if high <= low:
            return np.zeros(rgb.shape, dtype=np.uint8)
        return np.clip((values - low) * 255.0 / (high - low), 0, 255).astype(np.uint8)

    @staticmethod
    def _luminance(rgb):
        values = np.asarray(rgb)[..., :3].astype(np.float32)
        return values[..., 0] * 0.2126 + values[..., 1] * 0.7152 + values[..., 2] * 0.0722

    @staticmethod
    def _palette_rgb(indices, colormap):
        colors = np.asarray(colormap)
        if colors.shape[0] < 3:
            raise ValueError("TIFF Palette 调色板格式无效")
        colors = (colors[:3].T / 257.0).clip(0, 255).astype(np.uint8)
        return colors[np.asarray(indices, dtype=np.int64)]

    def read(self, request, progress=None, token=None):
        probe = self.probe(request)
        if probe.axes == "TYXC":
            raise ValueError("暂不支持 TYXC 彩色时序 TIFF；请拆分为单帧 RGB 或灰度 TYX 后导入")
        total = max(1, int(np.prod(probe.shape)) * np.dtype(probe.dtype).itemsize)
        _emit(progress, 0, total, "正在读取 TIFF 数据")
        if token is not None:
            token.raise_if_cancelled()
        with tifffile.TiffFile(request.path) as tf:
            series = tf.series[0]
            array = series.asarray()
            colormap = getattr(series.pages[0], "colormap", None)
        if token is not None:
            token.raise_if_cancelled()

        color_policy = request.options.get("color_policy", "preserve")
        display = array
        scientific = array
        warning = None
        if probe.color_mode == "palette" and colormap is not None:
            display = self._palette_rgb(array, colormap) if color_policy == "preserve" else array
        elif probe.color_mode == "rgb":
            scientific = self._luminance(array)
            display = self._rgb_uint8(array) if color_policy == "preserve" else scientific
            warning = "RGB TIFF 不包含可恢复的原始标量；分析数据已按亮度转换"

        scientific_axes = probe.axes.replace("C", "")
        if np.asarray(display).ndim == 3 and np.asarray(display).shape[-1] in (3, 4) and "T" not in probe.axes:
            display_axes = "YXC"
        else:
            display_axes = scientific_axes
        params = dict(request.options)
        params.update({
            "file_path": str(request.path), "source_shape": tuple(array.shape),
            "source_dtype": str(array.dtype), "source_axes": probe.axes,
            "scientific_axes": scientific_axes, "display_axes": display_axes,
            "photometric": probe.metadata.get("photometric"), "color_mode": probe.color_mode,
            "scientific_values_reconstructed": probe.color_mode not in {"rgb"},
        })
        if warning:
            params["import_warning"] = warning
        _emit(progress, total, total, "TIFF 数据读取完成")
        return ImportedPayload(
            np.asarray(scientific), np.asarray(display),
            _time_axis(scientific.shape, scientific_axes, params), "tiff", params, request.path.name,
        )
