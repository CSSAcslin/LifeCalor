from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tasks.model import CancellationToken

from compute.executor import _chunk_ranges, _close_loaded_mapping, _source_array
from compute.io import ComputeNpySink


@dataclass(frozen=True)
class PreprocessRunResult:
    output: object
    background: np.ndarray
    plan: object


def _safe_background(values):
    background = np.median(values, axis=0)
    safe = background.copy()
    zero_mask = np.abs(safe) < 1e-10
    if np.any(zero_mask):
        non_zero = np.abs(safe[~zero_mask])
        replacement = float(non_zero.min()) if non_zero.size else 1.0
        safe[zero_mask] = replacement if replacement > 0 else 1.0
    return background, safe


def run_preprocess_pipeline(
    plan,
    *,
    background_frames,
    cache_dir,
    token=None,
    progress=None,
):
    if plan.actual_backend != "cpu":
        raise ValueError("EM 预处理 CUDA 后端尚未验证")
    token = token or CancellationToken()
    source, loaded_here = _source_array(plan.request.source)
    if tuple(source.shape) != tuple(plan.request.shape):
        _close_loaded_mapping(source, loaded_here)
        raise ValueError(
            f"输入引用尺寸已变化: expected={plan.request.shape}, actual={source.shape}"
        )

    frame_count = min(max(1, int(background_frames)), plan.request.shape[0])
    background = np.empty(plan.request.shape[1:], dtype=plan.precision.output_dtype)
    sink = None
    destination = None
    try:
        if plan.output_to_disk:
            sink = ComputeNpySink(
                cache_dir,
                plan.request.task_id,
                plan.request.attempt_id,
                "data_processed",
                plan.output_shape,
                plan.precision.output_dtype,
                token=token,
            )
        else:
            destination = np.empty(plan.output_shape, dtype=plan.precision.output_dtype)

        total = plan.chunk_count
        if progress is not None:
            progress(0, total * 2, "正在分块计算背景中位数")
        for completed, (input_index, output_index) in enumerate(_chunk_ranges(plan), 1):
            token.raise_if_cancelled()
            spatial_index = output_index[1:]
            background_values = np.asarray(
                source[(slice(0, frame_count), *input_index[1:])],
                dtype=plan.precision.compute_dtype,
            )
            raw_background, safe_background = _safe_background(background_values)
            background[spatial_index] = raw_background.astype(
                plan.precision.output_dtype, copy=False
            )
            if progress is not None:
                progress(completed * 2 - 1, total * 2, "正在分块计算背景中位数")
            values = np.asarray(source[input_index], dtype=plan.precision.compute_dtype)
            result = ((values - safe_background) / safe_background).astype(
                plan.precision.output_dtype, copy=False
            )
            if sink is not None:
                sink.write_block(output_index, result)
            else:
                destination[output_index] = result
            if progress is not None:
                progress(completed * 2, total * 2, "正在分块写入预处理结果")

        token.raise_if_cancelled()
        output = sink.commit() if sink is not None else destination
        return PreprocessRunResult(output=output, background=background, plan=plan)
    except Exception:
        if sink is not None:
            sink.abort()
        raise
    finally:
        _close_loaded_mapping(source, loaded_here)
