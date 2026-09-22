from __future__ import annotations

from pathlib import Path

import numpy as np

from ArrayCache import ArrayRef
from tasks.model import CancellationToken

from .blocking import iter_block_plans
from .io import ComputeNpySink


def _source_array(source):
    if isinstance(source, ArrayRef):
        return source.load(mmap_mode="r"), True
    array = np.asanyarray(source)
    return array, False


def _close_loaded_mapping(array, loaded_here):
    if not loaded_here:
        return
    mapping = getattr(array, "_mmap", None)
    if mapping is not None:
        mapping.close()


def split_spatial_tile(block, output_index):
    """Split a THW spatial tile while preserving its global output coordinates."""
    height, width = block.shape[1:]
    if height <= 1 and width <= 1:
        return ()
    if height >= width and height > 1:
        first = height // 2
        row_slice = output_index[1]
        middle = row_slice.start + first
        return (
            (
                block[:, :first, :],
                (output_index[0], slice(row_slice.start, middle), output_index[2]),
            ),
            (
                block[:, first:, :],
                (output_index[0], slice(middle, row_slice.stop), output_index[2]),
            ),
        )
    first = width // 2
    column_slice = output_index[2]
    middle = column_slice.start + first
    return (
        (
            block[:, :, :first],
            (output_index[0], output_index[1], slice(column_slice.start, middle)),
        ),
        (
            block[:, :, first:],
            (output_index[0], output_index[1], slice(middle, column_slice.stop)),
        ),
    )


def _chunk_ranges(plan):
    shape = plan.request.shape
    chunk_shape = plan.chunk_shape
    if len(shape) == 3:
        for block in iter_block_plans(shape, core_shape=chunk_shape):
            if len(plan.output_shape) == 3:
                output_index = block.output_slices
            elif len(plan.output_shape) == 2:
                output_index = block.output_slices[1:]
            else:
                raise ValueError(
                    f"THW 输入无法映射到输出 shape={plan.output_shape}"
                )
            yield block.read_slices, output_index
        return

    for block in iter_block_plans(shape, core_shape=chunk_shape):
        yield block.read_slices, block.output_slices

def run_bounded_cpu(
    plan,
    compute_block,
    *,
    cache_dir=None,
    field_name="data_processed",
    token: CancellationToken | None = None,
    progress=None,
):
    """Execute a planned CPU transform without materializing the complete input copy.

    ``compute_block`` receives ``(input_block, input_index, output_index)`` and must
    return exactly the output block described by ``output_index``.
    """
    if plan.actual_backend != "cpu":
        raise ValueError("run_bounded_cpu 只接受 CPU 执行计划")
    token = token or CancellationToken()
    source, loaded_here = _source_array(plan.request.source)
    if tuple(source.shape) != tuple(plan.request.shape):
        _close_loaded_mapping(source, loaded_here)
        raise ValueError(
            f"输入引用尺寸已变化: expected={plan.request.shape}, actual={source.shape}"
        )

    sink = None
    destination = None
    try:
        if plan.output_to_disk:
            if cache_dir is None:
                raise ValueError("流式结果需要可写缓存目录")
            sink = ComputeNpySink(
                Path(cache_dir),
                plan.request.task_id,
                plan.request.attempt_id,
                field_name,
                plan.output_shape,
                plan.precision.output_dtype,
                token=token,
            )
        else:
            destination = np.empty(plan.output_shape, dtype=plan.precision.output_dtype)

        total = plan.chunk_count
        if progress is not None:
            progress(0, total * 10, "正在执行 CPU 分块计算")
        for completed, (input_index, output_index) in enumerate(_chunk_ranges(plan), 1):
            token.raise_if_cancelled()
            result = np.asarray(
                compute_block(source[input_index], input_index, output_index)
            )
            expected = tuple(plan.output_shape[index] if isinstance(part, slice) and part == slice(None)
                             else (part.stop - part.start)
                             for index, part in enumerate(output_index))
            if result.shape != expected:
                raise ValueError(
                    f"算法返回块尺寸不匹配: expected={expected}, actual={result.shape}"
                )
            if sink is not None:
                sink.write_block(output_index, result)
            else:
                destination[output_index] = result
            if progress is not None:
                progress(completed * 9, total * 10, "正在执行 CPU 分块计算")

        token.raise_if_cancelled()
        result = sink.commit() if sink is not None else destination
        if progress is not None:
            progress(total * 10, total * 10, "CPU 分块计算完成")
        return result
    except Exception:
        if sink is not None:
            sink.abort()
        raise
    finally:
        _close_loaded_mapping(source, loaded_here)
