from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage

from ArrayCache import ArrayRef
from tasks.model import CancellationToken

from compute.blocking import iter_block_plans
from compute.executor import _close_loaded_mapping, _source_array
from compute.io import ComputeNpySink


_PAD_MODES = {
    "reflect": "symmetric",
    "mirror": "reflect",
    "nearest": "edge",
    "wrap": "wrap",
    "constant": "constant",
}


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


def _pad_global_edges(block, pad_width, boundary, cval):
    try:
        mode = _PAD_MODES[str(boundary)]
    except KeyError as exc:
        raise ValueError(f"不支持的卷积边界模式: {boundary}") from exc
    if not any(left or right for left, right in pad_width):
        return np.asarray(block)
    if mode == "constant":
        return np.pad(block, pad_width, mode=mode, constant_values=cval)
    return np.pad(block, pad_width, mode=mode)


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
        plans = tuple(iter_block_plans(
            array.shape,
            core_shape=core_shape,
            halo_before=halo_before,
            halo_after=halo_after,
        ))
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

        total = len(plans)
        if progress is not None:
            progress(0, total, "正在执行分块时空卷积")
        for completed, plan in enumerate(plans, 1):
            token.raise_if_cancelled()
            read_block = np.asarray(array[plan.read_slices], dtype=output_dtype)
            padded = _pad_global_edges(
                read_block, plan.pad_width, boundary, cval
            )
            convolved = ndimage.convolve(
                padded,
                np.asarray(weights, dtype=output_dtype),
                mode="constant",
                cval=0.0,
                origin=origin,
                output=output_dtype,
            )
            result = convolved[plan.local_core_slices]
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
