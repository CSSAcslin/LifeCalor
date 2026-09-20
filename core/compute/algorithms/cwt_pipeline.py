from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tasks.model import CancellationToken

from compute.backends.cpu import cwt_block_cpu
from compute.executor import _chunk_ranges, _close_loaded_mapping, _source_array
from compute.io import ComputeNpySink


@dataclass(frozen=True)
class CwtRunResult:
    output: object
    scales: np.ndarray
    frequencies: np.ndarray
    whole_mean: np.ndarray
    plan: object


def _axis_matches(reference, candidate):
    return reference.shape == candidate.shape and np.allclose(
        reference, candidate, rtol=1e-7, atol=1e-10
    )


def run_cwt_pipeline(plan, params, *, cache_dir, token=None, progress=None):
    if plan.actual_backend != "cpu":
        raise ValueError("CWT CUDA 后端尚未通过科学一致性验证")
    token = token or CancellationToken()
    source, loaded_here = _source_array(plan.request.source)
    if tuple(source.shape) != tuple(plan.request.shape):
        _close_loaded_mapping(source, loaded_here)
        raise ValueError(
            f"输入引用尺寸已变化: expected={plan.request.shape}, actual={source.shape}"
        )

    sink = None
    destination = None
    scales = frequencies = None
    whole_sum = np.zeros(plan.output_shape[0], dtype=np.float64)
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
            progress(0, total, "正在使用 CPU 执行 CWT")
        for completed, (input_index, output_index) in enumerate(_chunk_ranges(plan), 1):
            token.raise_if_cancelled()
            result, block_scales, block_frequencies = cwt_block_cpu(
                source[input_index],
                params,
                compute_dtype=plan.precision.compute_dtype,
                output_dtype=plan.precision.output_dtype,
            )
            if scales is None:
                scales = block_scales
                frequencies = block_frequencies
            elif (
                not _axis_matches(scales, block_scales)
                or not _axis_matches(frequencies, block_frequencies)
            ):
                raise RuntimeError("CPU CWT 分块返回了不一致的尺度或频率轴")
            if sink is not None:
                sink.write_block(output_index, result)
            else:
                destination[output_index] = result
            whole_sum += np.asarray(result, dtype=np.float64).sum(axis=(1, 2))
            if progress is not None:
                progress(completed, total, "正在使用 CPU 执行 CWT")

        token.raise_if_cancelled()
        output = sink.commit() if sink is not None else destination
        return CwtRunResult(
            output=output,
            scales=np.asarray(scales),
            frequencies=np.asarray(frequencies),
            whole_mean=whole_sum / (plan.request.shape[1] * plan.request.shape[2]),
            plan=plan,
        )
    except Exception:
        if sink is not None:
            sink.abort()
        raise
    finally:
        _close_loaded_mapping(source, loaded_here)
