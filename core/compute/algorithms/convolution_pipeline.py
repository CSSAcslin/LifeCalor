from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy import ndimage

from ArrayCache import ArrayRef
from tasks.model import CancellationToken

from compute.blocking import iter_block_plans
from compute.executor import _close_loaded_mapping, _source_array
from compute.io import ComputeNpySink


def _normalize_origin(origin, ndim):
    if np.isscalar(origin):
        origin = (int(origin),) * ndim
    origin = tuple(int(value) for value in origin)
    if len(origin) != ndim:
        raise ValueError(f"origin 必须包含 {ndim} 个整数")
    return origin


def convolution_halo(kernel_shape, origin=0):
    kernel_shape = tuple(int(value) for value in kernel_shape)
    if not kernel_shape or any(value <= 0 for value in kernel_shape):
        raise ValueError(f"无效卷积核尺寸: {kernel_shape}")
    origin = _normalize_origin(origin, len(kernel_shape))
    before, after = [], []
    for size, shift in zip(kernel_shape, origin):
        center = size // 2 + shift
        if center < 0 or center >= size:
            raise ValueError(
                f"origin={origin} 超出 kernel_shape={kernel_shape} 的有效范围"
            )
        before.append(size - 1 - center)
        after.append(center)
    return tuple(before), tuple(after)


def convolve_region(array, kernel, core_slices, *, boundary="reflect", cval=0.0, origin=0):
    """Compute a bounded region using neighbors from the global source.

    Global coordinate mapping also handles wrap boundaries and kernels whose
    halo is longer than the image, without wrapping or reflecting a local tile.
    """
    before, after = convolution_halo(kernel.shape, origin)
    indices = []
    valid_axes = []
    for part, length, left, right in zip(core_slices, array.shape, before, after):
        positions = np.arange(part.start - left, part.stop + right)
        valid_axes.append((positions >= 0) & (positions < length))
        if boundary == "wrap":
            mapped = positions % length
        elif boundary == "reflect":
            folded = positions % (2 * length)
            mapped = np.minimum(folded, 2 * length - 1 - folded)
        elif boundary == "mirror":
            period = max(1, 2 * length - 2)
            folded = positions % period
            mapped = np.minimum(folded, period - folded)
        elif boundary in ("nearest", "constant"):
            mapped = np.clip(positions, 0, length - 1)
        else:
            raise ValueError(f"不支持的卷积边界模式: {boundary}")
        indices.append(mapped)
    dtype = np.result_type(array.dtype, kernel.dtype, np.float32)
    padded = np.asarray(array[np.ix_(*indices)], dtype=dtype)
    if boundary == "constant":
        for axis, valid in enumerate(valid_axes):
            invalid_index = [slice(None)] * array.ndim
            invalid_index[axis] = ~valid
            padded[tuple(invalid_index)] = cval
    convolved = ndimage.convolve(padded, np.asarray(kernel, dtype=dtype),
                                mode="constant", cval=0.0, origin=origin, output=dtype)
    local_core = tuple(slice(left, left + part.stop - part.start)
                       for part, left in zip(core_slices, before))
    return convolved[local_core].copy()


def run_spatiotemporal_convolution(
    source,
    kernel,
    *,
    core_shape,
    boundary="reflect",
    cval=0.0,
    origin=0,
    cache_dir=None,
    output_to_disk=False,
    task_id="convolution",
    attempt_id=0,
    token=None,
    progress=None,
):
    """Convolve HW or THW data using overlap tiles with global-only padding."""
    token = token or CancellationToken()
    array, loaded_here = _source_array(source)
    weights = np.asarray(kernel)
    sink = None
    try:
        if array.ndim not in (2, 3):
            raise ValueError(f"时空卷积仅支持 HW/THW，实际 shape={array.shape}")
        if weights.ndim != array.ndim:
            raise ValueError(
                f"数据与卷积核维数必须一致: data={array.ndim}, kernel={weights.ndim}"
            )
        origin = _normalize_origin(origin, array.ndim)
        halo_before, halo_after = convolution_halo(weights.shape, origin)
        core_shape = tuple(int(value) for value in core_shape)
        if len(core_shape) != array.ndim or any(size <= 0 for size in core_shape):
            raise ValueError("core_shape 需要为每个轴提供正整数")
        plans = iter_block_plans(
            array.shape,
            core_shape=core_shape,
            halo_before=halo_before,
            halo_after=halo_after,
        )
        output_dtype = np.result_type(array.dtype, weights.dtype, np.float32)
        if output_to_disk:
            if cache_dir is None:
                raise ValueError("落盘卷积结果需要可写缓存目录")
            sink = ComputeNpySink(
                Path(cache_dir), task_id, int(attempt_id), "data_processed",
                array.shape, output_dtype, token=token,
            )
            destination = None
        else:
            destination = np.empty(array.shape, dtype=output_dtype)

        total = math.prod((size + step - 1) // step for size, step in zip(array.shape, core_shape))
        if progress is not None:
            progress(0, total, "正在执行分块时空卷积")
        for completed, plan in enumerate(plans, 1):
            token.raise_if_cancelled()
            result = convolve_region(array, weights, plan.output_slices,
                                     boundary=boundary, cval=cval, origin=origin)
            if sink is not None:
                sink.write_block(plan.output_slices, result)
            else:
                destination[plan.output_slices] = result
            if progress is not None:
                progress(completed, total, "正在执行分块时空卷积")

        token.raise_if_cancelled()
        return sink.commit() if sink is not None else destination
    except Exception:
        if sink is not None:
            sink.abort()
        raise
    finally:
        _close_loaded_mapping(array, loaded_here)
