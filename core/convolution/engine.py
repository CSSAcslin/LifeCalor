from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ArrayCache import ArrayRef
from DataManager import ProcessedData
from calculator.engine import CalculationEngine
from calculator.metadata import CalculationMetadataPolicy, EXCLUDED_KEYS
from calculator.model import OperandSpec
from compute.algorithms.convolution_pipeline import (
    convolution_halo, convolve_region, run_spatiotemporal_convolution,
)
from tasks.model import CancellationToken


@dataclass(frozen=True)
class ConvolutionPlan:
    source: object
    kernel: np.ndarray
    axes: str
    payload_key: str | None = None
    normalize: bool = False
    boundary: str = "reflect"
    cval: float = 0.0
    origin: tuple[int, ...] = ()
    memory_limit_bytes: int = 512 * 1024 * 1024
    resident_bytes: int = 0
    disk_threshold_bytes: int = 512 * 1024 * 1024
    cache_dir: str | None = None


@dataclass(frozen=True)
class ConvolutionLayout:
    shape: tuple[int, ...]
    dtype: np.dtype
    core_shape: tuple[int, ...]
    output_to_disk: bool
    working_bytes: int


def source_storage(plan):
    source = plan.source
    if plan.payload_key is not None:
        return source.out_processed[plan.payload_key]
    values = getattr(source, "__dict__", {})
    for key in ("_data_processed_storage", "_data_origin_storage"):
        if values.get(key) is not None:
            return values[key]
    return CalculationEngine.source_array(OperandSpec("A", source, plan.payload_key))


def effective_kernel(plan):
    weights = np.asarray(plan.kernel)
    if weights.ndim != len(plan.axes) or not weights.size or weights.dtype.kind not in "biufc":
        raise ValueError("卷积核必须是与 HW/THW 数据维数一致的数值数组")
    weights = weights.astype(np.result_type(weights.dtype, np.float64), copy=True)
    if not np.all(np.isfinite(weights)):
        raise ValueError("卷积核权重必须是有限数值")
    if plan.normalize:
        total = weights.sum()
        if total == 0:
            raise ValueError("零和卷积核无法按权重和归一化，请关闭归一化")
        if not np.isfinite(total):
            raise ValueError("卷积核权重和溢出，无法归一化")
        weights /= total
    return weights


def _working_bytes(core_shape, before, after, weights, dtype):
    padded = tuple(size + left + right for size, left, right in zip(core_shape, before, after))
    arrays_bytes = int(np.prod(padded, dtype=object)) * dtype.itemsize * 4
    # ndimage creates an offset row for every distinct border configuration.
    # Dense long kernels can make this table dominate the bounded input tile.
    border_configurations = int(np.prod(
        [min(length, kernel_size) for length, kernel_size in zip(padded, weights.shape)],
        dtype=object,
    ))
    offsets_bytes = int(np.count_nonzero(weights)) * border_configurations * np.dtype(np.intp).itemsize
    if dtype.kind == "c":
        offsets_bytes *= 2
    return arrays_bytes + weights.nbytes + offsets_bytes


def validate_convolution(plan):
    info = CalculationEngine.source_info(OperandSpec("A", plan.source, plan.payload_key))
    if plan.axes not in ("HW", "THW") or len(info.shape) != len(plan.axes):
        raise ValueError("请选择二维 HW 或三维 THW 数值数据")
    if not all(info.shape) or info.dtype.kind not in "biufc":
        raise ValueError("时空卷积需要非空数值数据")
    metadata = CalculationMetadataPolicy.source_mapping(plan.source)
    source_axes = metadata.get(f"{plan.payload_key}_axes") if plan.payload_key else (
        metadata.get("scientific_axes") or metadata.get("source_axes")
    )
    if source_axes:
        source_axes = "".join(source_axes).replace("Y", "H").replace("X", "W").upper()
        if source_axes != plan.axes:
            raise ValueError(f"源数据轴为 {source_axes}，请先用数据计算器转换为 {plan.axes}")
    weights = effective_kernel(plan)
    before, after = convolution_halo(weights.shape, plan.origin or 0)
    if plan.boundary not in ("reflect", "mirror", "nearest", "wrap", "constant"):
        raise ValueError("不支持的卷积边界模式")
    if not np.isfinite(plan.cval):
        raise ValueError("常值边界值必须有限")
    dtype = np.dtype(np.result_type(info.dtype, weights.dtype, np.float32))
    output_bytes = int(np.prod(info.shape, dtype=object)) * dtype.itemsize
    limit = int(plan.memory_limit_bytes) - int(plan.resident_bytes)
    to_disk = output_bytes >= plan.disk_threshold_bytes or output_bytes > limit // 3
    available = limit - (0 if to_disk else output_bytes)
    core = [min(size, 64) for size in info.shape]

    working = _working_bytes(core, before, after, weights, dtype)
    while working > available and max(core) > 1:
        axis = int(np.argmax(core))
        core[axis] = max(1, core[axis] // 2)
        working = _working_bytes(core, before, after, weights, dtype)
    if working > available:
        raise MemoryError("卷积核和现有数据超过工作内存预算，请缩小卷积核或提高预算")
    return ConvolutionLayout(info.shape, dtype, tuple(core), to_disk, working)


def _kernel_metadata(weights):
    values = np.asarray(weights)
    if np.iscomplexobj(values):
        return {"real": values.real.tolist(), "imag": values.imag.tolist()}
    return values.tolist()


def _metadata(plan, layout, task_id):
    inherited = CalculationMetadataPolicy.source_mapping(plan.source)
    metadata = {}
    for key, value in inherited.items():
        if key in EXCLUDED_KEYS or key in ("result_fields", "active_result_field"):
            continue
        safe = CalculationMetadataPolicy.safe_value(value)
        if safe is not None:
            metadata[key] = safe
    metadata.update(scientific_axes=plan.axes, source_axes=plan.axes, display_axes=plan.axes)
    metadata["convolution"] = {
        "kernel": _kernel_metadata(plan.kernel),
        "effective_kernel": _kernel_metadata(effective_kernel(plan)),
        "normalize": plan.normalize,
        "boundary": plan.boundary,
        "cval": plan.cval,
        "origin": tuple(plan.origin or (0,) * len(layout.shape)),
        "source_name": getattr(plan.source, "name", ""),
        "source_timestamp": getattr(plan.source, "timestamp", None),
        "source_payload_key": plan.payload_key,
    }
    metadata["compute"] = {
        "algorithm": "spatiotemporal_convolution", "task_id": task_id,
        "requested_backend": "cpu", "actual_backend": "cpu",
        "compute_dtype": layout.dtype.name, "output_dtype": layout.dtype.name,
        "chunk_shape": layout.core_shape, "output_to_disk": layout.output_to_disk,
        "working_bytes": layout.working_bytes,
    }
    return metadata


def execute_convolution(plan, *, token=None, task_id="convolution", progress=None):
    token = token or CancellationToken()
    token.raise_if_cancelled()
    layout = validate_convolution(plan)
    metadata = _metadata(plan, layout, task_id)
    result = None
    try:
        result = run_spatiotemporal_convolution(
            source_storage(plan), effective_kernel(plan), core_shape=layout.core_shape,
            boundary=plan.boundary, cval=plan.cval, origin=plan.origin or 0,
            cache_dir=plan.cache_dir, output_to_disk=layout.output_to_disk,
            task_id=task_id, token=token, progress=progress,
        )
        token.raise_if_cancelled()
        points = CalculationMetadataPolicy._time_axis(plan.source, layout.shape, metadata)
        processed = ProcessedData(
            getattr(plan.source, "timestamp", 0.0),
            f"{getattr(plan.source, 'name', 'data')}@conv", "spatiotemporal_convolution",
            time_point=points, data_processed=result, out_processed=metadata,
        )
        return processed
    except Exception:
        if isinstance(result, ArrayRef):
            Path(result.path).unlink(missing_ok=True)
        raise


def preview_convolution(plan, frame, row, column, size=48):
    """Read one bounded spatial ROI plus its full temporal/spatial halo."""
    layout = validate_convolution(plan)
    weights = effective_kernel(plan)
    start = (row, column) if plan.axes == "HW" else (frame, row, column)
    extent = (size, size) if plan.axes == "HW" else (1, size, size)
    if any(index < 0 or index >= length for index, length in zip(start, layout.shape)):
        raise ValueError("预览起点超出数据范围")
    core = tuple(slice(index, min(length, index + count))
                 for index, count, length in zip(start, extent, layout.shape))
    before, after = convolution_halo(weights.shape, plan.origin or 0)
    core_shape = tuple(part.stop - part.start for part in core)
    if _working_bytes(core_shape, before, after, weights, layout.dtype) > 64 * 1024 * 1024:
        raise MemoryError("预览所需邻域超过 64 MB，请缩小卷积核")
    storage = source_storage(plan)
    loaded = isinstance(storage, ArrayRef)
    array = storage.load(mmap_mode="r") if loaded else np.asanyarray(storage)
    try:
        original = np.array(array[core], copy=True)
        result = convolve_region(array, weights, core, boundary=plan.boundary,
                                 cval=plan.cval, origin=plan.origin or 0)
        return original.reshape(original.shape[-2:]), result.reshape(result.shape[-2:])
    finally:
        if loaded and getattr(array, "_mmap", None) is not None:
            array._mmap.close()
