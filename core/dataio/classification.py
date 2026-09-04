from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np



class DataCategory(str, Enum):
    SCALAR = "标量"
    VECTOR = "向量"
    LINEAR = "线性"
    IMAGE = "图片"
    VIDEO = "视频"
    MATRIX_3D = "三维矩阵"
    HIGH_DIMENSIONAL = "高维数据"
    STRUCTURED = "结构数据"
    UNKNOWN = "未知"


@dataclass(frozen=True)
class DataDescriptor:
    category: DataCategory
    shape: tuple[int, ...] = ()
    dtype: str = ""
    axes: tuple[str, ...] = ()
    reason: str = ""
    inferred: bool = True

    @property
    def label(self) -> str:
        return self.category.value


def _shape_dtype(value: Any) -> tuple[tuple[int, ...], str]:
    if isinstance(value, np.ndarray) or value.__class__.__name__ == "ArrayRef":
        return tuple(int(v) for v in value.shape), str(value.dtype)
    if np.isscalar(value) or value is None:
        return (), type(value).__name__
    if isinstance(value, (list, tuple)):
        return (len(value),), type(value).__name__
    shape = tuple(getattr(value, "datashape", ()) or getattr(value, "shape", ()) or ())
    dtype = getattr(value, "datatype", None) or getattr(value, "dtype", None)
    return tuple(int(v) for v in shape), str(dtype or "")


def _metadata(source: Any) -> dict:
    result = {}
    for name in ("parameters", "out_processed"):
        value = getattr(source, name, None)
        if isinstance(value, dict):
            result.update(value)
    return result


def _axes(source: Any, metadata: dict, ndim: int) -> tuple[str, ...]:
    value = None
    for key in ("scientific_axes", "source_axes", "display_axes", "axes", "axis_order"):
        value = metadata.get(key)
        if value is not None:
            break
    if value is None:
        value = getattr(source, "axes", None)
    if isinstance(value, str):
        value = tuple(part.strip().upper() for part in value.replace(",", " ").split() if part.strip())
        if len(value) == 1 and len(value[0]) == ndim:
            value = tuple(value[0])
    elif isinstance(value, (list, tuple)):
        value = tuple(str(part).strip().upper() for part in value)
    else:
        value = ()
    aliases = {"Y": "H", "X": "W", "TIME": "T", "HEIGHT": "H", "WIDTH": "W"}
    normalized = tuple(aliases.get(part, part) for part in value)
    return normalized if len(normalized) == ndim else ()


def _time_length(source: Any) -> int | None:
    value = getattr(source, "time_point", None)
    if value is None:
        return None
    try:
        return int(value.shape[0]) if hasattr(value, "shape") else len(value)
    except (TypeError, IndexError):
        return None


def describe_value(value: Any = None, *, shape=None, dtype=None, axes=(), time_length=None, semantic_hint="") -> DataDescriptor:
    if shape is None:
        shape, inferred_dtype = _shape_dtype(value)
        dtype = dtype or inferred_dtype
    shape = tuple(int(v) for v in (shape or ()))
    dtype = str(dtype or "")
    axes = tuple(str(v).upper() for v in (axes or ()))
    hint = str(semantic_hint or "").lower()
    ndim = len(shape)
    if isinstance(value, dict):
        return DataDescriptor(DataCategory.STRUCTURED, shape, dtype, axes, "映射参数")
    if ndim == 0:
        return DataDescriptor(DataCategory.SCALAR, shape, dtype, axes, "单一数值")
    if ndim == 1:
        if time_length == shape[0] or any(word in hint for word in ("line", "curve", "plot", "time")):
            return DataDescriptor(DataCategory.LINEAR, shape, dtype, axes, "一维连续序列")
        return DataDescriptor(DataCategory.VECTOR, shape, dtype, axes, "一维数据")
    if ndim == 2:
        if 1 in shape:
            return DataDescriptor(DataCategory.LINEAR, shape, dtype, axes, "单行或单列序列")
        return DataDescriptor(DataCategory.IMAGE, shape, dtype, axes, "二维矩阵")
    if ndim == 3:
        if axes.count("T") == 1:
            return DataDescriptor(DataCategory.VIDEO, shape, dtype, axes, "轴元数据包含 T", inferred=False)
        matching = [index for index, size in enumerate(shape) if time_length and size == time_length]
        if len(matching) == 1:
            return DataDescriptor(DataCategory.VIDEO, shape, dtype, axes, "唯一维度匹配时间轴")
        return DataDescriptor(DataCategory.MATRIX_3D, shape, dtype, axes, "未确认时间轴")
    return DataDescriptor(DataCategory.HIGH_DIMENSIONAL, shape, dtype, axes, f"{ndim} 维数据")


def describe_source(source: Any, payload_key: str | None = None) -> DataDescriptor:
    metadata = _metadata(source)
    if payload_key is not None:
        value = metadata.get(payload_key)
        shape, dtype = _shape_dtype(value)
        axes_value = metadata.get(f"{payload_key}_axes", ())
        axes = tuple(axes_value) if isinstance(axes_value, (list, tuple)) else ()
        hint = metadata.get(f"{payload_key}_type", payload_key)
        return describe_value(value, shape=shape, dtype=dtype, axes=axes, time_length=_time_length(source), semantic_hint=hint)
    shape, dtype = _shape_dtype(source)
    axes = _axes(source, metadata, len(shape))
    hint = metadata.get("analysis_mode", getattr(source, "type_processed", ""))
    return describe_value(shape=shape, dtype=dtype, axes=axes, time_length=_time_length(source), semantic_hint=hint)