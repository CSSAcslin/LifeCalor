from __future__ import annotations

import json
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import tifffile
import cv2
import h5py
import sif_parser

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
        axes = (getattr(series, "axes", "") or "").upper().replace("S", "C").replace("Q", "T").replace("I", "T")
        if len(axes) == len(shape) and "Y" in axes and "X" in axes:
            return axes
        if len(shape) == 2:
            return "YX"
        is_color = photometric == "RGB" and shape[-1] in (3, 4)
        if len(shape) == 3 and is_color:
            return "YXC"
        if len(shape) == 3:
            return "TYX"
        if len(shape) == 4 and shape[-1] in (3, 4):
            return "TYXC"
        return "?" * len(shape)

    @staticmethod
    def _ome_physical_metadata(ome_xml):
        if not ome_xml:
            return {}
        try:
            root = ElementTree.fromstring(ome_xml)
            pixels = next(
                (element for element in root.iter() if element.tag.rsplit("}", 1)[-1] == "Pixels"),
                None,
            )
        except ElementTree.ParseError:
            return {}
        if pixels is None:
            return {}
        source_keys = (
            "PhysicalSizeX", "PhysicalSizeXUnit", "PhysicalSizeY", "PhysicalSizeYUnit",
            "PhysicalSizeZ", "PhysicalSizeZUnit", "TimeIncrement", "TimeIncrementUnit",
        )
        values = {key: pixels.attrib[key] for key in source_keys if key in pixels.attrib}
        return {"is_ome": True, "ome_physical_metadata": values}
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
            metadata.update(self._ome_physical_metadata(getattr(tf, "ome_metadata", None)))
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
        elif probe.color_mode == "color_stack":
            scientific = self._luminance(array)
            display = self._rgb_uint8(array) if color_policy == "preserve" else scientific
            warning = "彩色时序 TIFF 的分析数据已按亮度转换，原始颜色仅用于显示"
        elif probe.color_mode == "rgb":
            scientific = self._luminance(array)
            display = self._rgb_uint8(array) if color_policy == "preserve" else scientific
            warning = "RGB TIFF 不包含可恢复的原始标量；分析数据已按亮度转换"

        scientific_axes = probe.axes.replace("C", "")
        display_array = np.asarray(display)
        scientific_array = np.asarray(scientific)
        if display_array.ndim == scientific_array.ndim + 1 and display_array.shape[-1] in (3, 4):
            display_axes = f"{scientific_axes}C"
        elif display_array.ndim >= 3 and display_array.shape[-1] in (3, 4) and "C" in probe.axes:
            display_axes = probe.axes
        else:
            display_axes = scientific_axes
        params = dict(request.options)
        params.update({
            "file_path": str(request.path), "source_shape": tuple(array.shape),
            "source_dtype": str(array.dtype), "source_axes": probe.axes,
            "scientific_axes": scientific_axes, "display_axes": display_axes,
            "photometric": probe.metadata.get("photometric"), "color_mode": probe.color_mode,
            "is_ome": bool(probe.metadata.get("is_ome")),
            "ome_physical_metadata": dict(probe.metadata.get("ome_physical_metadata", {})),
            "scientific_values_reconstructed": probe.color_mode not in {"rgb"},
        })
        if warning:
            params["import_warning"] = warning
        _emit(progress, total, total, "TIFF 数据读取完成")
        return ImportedPayload(
            np.asarray(scientific), np.asarray(display),
            _time_axis(scientific.shape, scientific_axes, params), "tiff", params, request.path.name,
        )

def list_hdf5_datasets(path):
    datasets = []
    with h5py.File(path, "r") as handle:
        def visit(name, value):
            if isinstance(value, h5py.Dataset):
                datasets.append((f"/{name}", tuple(value.shape), str(value.dtype)))
        handle.visititems(visit)
    return datasets


class AviImporter(Importer):
    format_id = "avi"
    extensions = (".avi", ".mp4", ".mov", ".mkv")

    def probe(self, request):
        cap = cv2.VideoCapture(str(request.path))
        if not cap.isOpened():
            raise IOError(f"无法打开视频文件: {request.path}")
        try:
            frames = max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            height = max(0, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            width = max(0, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        finally:
            cap.release()
        if not frames or not height or not width:
            raise ValueError("视频缺少有效的帧数或尺寸信息")
        return ImportProbe(self.format_id, (frames, height, width), "uint8", "TYX", "grayscale", {"source_fps": fps})

    def read(self, request, progress=None, token=None):
        probe = self.probe(request)
        cap = cv2.VideoCapture(str(request.path))
        frames = []
        try:
            for index in range(probe.shape[0]):
                if token is not None:
                    token.raise_if_cancelled()
                ok, frame = cap.read()
                if not ok:
                    break
                if frame.ndim == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frames.append(frame)
                _emit(progress, index + 1, probe.shape[0], "正在读取视频")
        finally:
            cap.release()
        if not frames:
            raise ValueError("视频中没有读取到有效帧")
        array = np.stack(frames, axis=0)
        params = dict(request.options)
        source_fps = float(probe.metadata.get("source_fps") or 0)
        fps = float(params.get("fps") or source_fps or 0)
        params.update({
            "file_path": str(request.path), "fps": fps or None,
            "source_fps": source_fps or None, "source_shape": tuple(array.shape),
            "source_dtype": str(array.dtype), "source_axes": "TYX",
            "scientific_axes": "TYX", "display_axes": "TYX",
        })
        return ImportedPayload(array, array, _time_axis(array.shape, "TYX", params), "video", params, request.path.name)


class SifImporter(Importer):
    format_id = "sif"
    extensions = (".sif",)

    @staticmethod
    def _read(path):
        result = sif_parser.np_open(str(path))
        array = np.asarray(result[0][0])
        if array.ndim not in (2, 3):
            raise ValueError(f"SIF 数据必须是二维或三维，当前 shape={array.shape}")
        return array

    def probe(self, request):
        array = self._read(request.path)
        axes = "YX" if array.ndim == 2 else "TYX"
        return ImportProbe(self.format_id, tuple(array.shape), str(array.dtype), axes)

    def read(self, request, progress=None, token=None):
        if token is not None:
            token.raise_if_cancelled()
        array = self._read(request.path)
        axes = "YX" if array.ndim == 2 else "TYX"
        params = dict(request.options)
        params.update({
            "file_path": str(request.path), "source_shape": tuple(array.shape),
            "source_dtype": str(array.dtype), "source_axes": axes,
            "scientific_axes": axes, "display_axes": axes,
        })
        _emit(progress, 1, 1, "SIF 数据读取完成")
        return ImportedPayload(array, array, _time_axis(array.shape, axes, params), "sif", params, request.path.name)


class Hdf5Importer(Importer):
    format_id = "hdf5"
    extensions = (".h5", ".hdf5", ".hdf")

    def probe(self, request):
        dataset_path = request.options.get("dataset_path")
        if not dataset_path:
            raise ValueError("请先选择 HDF5 数据集")
        with h5py.File(request.path, "r") as handle:
            if dataset_path not in handle or not isinstance(handle[dataset_path], h5py.Dataset):
                raise ValueError(f"HDF5 数据集不存在: {dataset_path}")
            dataset = handle[dataset_path]
            if dataset.dtype.kind in {"O", "S", "U", "V"}:
                raise ValueError(f"暂不支持非数值 HDF5 dtype: {dataset.dtype}")
            shape = tuple(dataset.shape)
            axes_attr = dataset.attrs.get("axes", "")
            if isinstance(axes_attr, bytes):
                axes_attr = axes_attr.decode("utf-8", errors="replace")
            axes = str(request.options.get("axes") or axes_attr or "").upper()
        if len(axes) != len(shape):
            axes = "YX" if len(shape) == 2 else "TYX" if len(shape) == 3 else "?" * len(shape)
        return ImportProbe(self.format_id, shape, str(dataset.dtype), axes, metadata={"dataset_path": dataset_path})

    def read(self, request, progress=None, token=None):
        probe = self.probe(request)
        total = max(1, int(np.prod(probe.shape)))
        with h5py.File(request.path, "r") as handle:
            dataset = handle[probe.metadata["dataset_path"]]
            if dataset.ndim == 0:
                array = np.asarray(dataset[()])
                _emit(progress, 1, 1, "HDF5 数据读取完成")
            else:
                array = np.empty(dataset.shape, dtype=dataset.dtype)
                block = max(1, min(dataset.shape[0], 64))
                completed = 0
                for start in range(0, dataset.shape[0], block):
                    if token is not None:
                        token.raise_if_cancelled()
                    end = min(dataset.shape[0], start + block)
                    array[start:end] = dataset[start:end]
                    completed += int(np.prod(dataset[start:end].shape))
                    _emit(progress, completed, total, "正在读取 HDF5 数据集")
        params = dict(request.options)
        params.update({
            "file_path": str(request.path), "dataset_path": probe.metadata["dataset_path"],
            "source_shape": tuple(array.shape), "source_dtype": str(array.dtype),
            "source_axes": probe.axes, "scientific_axes": probe.axes, "display_axes": probe.axes,
        })
        return ImportedPayload(array, array, _time_axis(array.shape, probe.axes, params), "hdf5", params, f"{request.path.name}:{probe.metadata['dataset_path']}")