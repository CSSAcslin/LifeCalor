from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from tasks.model import CancellationToken, TaskCancelled

from compute.backends.cpu import stft_block_cpu
from compute.executor import (
    _chunk_ranges,
    _close_loaded_mapping,
    _source_array,
    split_spatial_tile,
)
from compute.io import ComputeNpySink
from compute.worker import CudaBackendError, CudaWorkerClient


@dataclass(frozen=True)
class StftRunResult:
    output: object
    frequencies: np.ndarray
    times: np.ndarray
    selected_indices: np.ndarray
    whole_mean: np.ndarray
    plan: object
    fallback_reason: str = ""


def _axes_match(reference, candidate):
    return reference.shape == candidate.shape and np.allclose(
        reference, candidate, rtol=1e-7, atol=1e-10
    )


def _execute_stft(plan, params, *, cache_dir, token, progress, device_index):
    source, loaded_here = _source_array(plan.request.source)
    if tuple(source.shape) != tuple(plan.request.shape):
        _close_loaded_mapping(source, loaded_here)
        raise ValueError(
            f"输入引用尺寸已变化: expected={plan.request.shape}, actual={source.shape}"
        )
    sink = None
    destination = None
    worker = None
    whole_sum = np.zeros(plan.output_shape[0], dtype=np.float64)
    frequencies = times = selected = None

    def store_block(output_index, result):
        nonlocal whole_sum
        if sink is not None:
            sink.write_block(output_index, result)
        else:
            destination[output_index] = result
        whole_sum += np.asarray(result, dtype=np.float64).sum(axis=(1, 2))

    def compute_gpu(block, output_index, block_id, retry=0):
        nonlocal frequencies, times, selected
        try:
            result, block_frequencies, block_times, block_selected = worker.execute(
                "stft",
                task_id=plan.request.task_id,
                attempt_id=plan.request.attempt_id,
                block_id=block_id,
                block=block,
                params=params,
                compute_dtype=plan.precision.compute_dtype,
                output_dtype=plan.precision.output_dtype,
                token=token,
            )
        except CudaBackendError as exc:
            split = split_spatial_tile(block, output_index)
            if exc.out_of_memory and retry < 3 and split:
                for child_number, (child, child_index) in enumerate(split):
                    compute_gpu(
                        child,
                        child_index,
                        f"{block_id}.{child_number}",
                        retry + 1,
                    )
                return
            raise
        if frequencies is None:
            frequencies = block_frequencies
            times = block_times
            selected = block_selected
        elif (
            not _axes_match(frequencies, block_frequencies)
            or not _axes_match(times, block_times)
            or not np.array_equal(selected, block_selected)
        ):
            raise RuntimeError("CUDA STFT 分块返回了不一致的时间轴或频率选择")
        store_block(output_index, result)

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
        if plan.actual_backend == "gpu":
            worker = CudaWorkerClient(device_index)

        total = plan.chunk_count
        if progress is not None:
            progress(0, total, f"正在使用 {plan.actual_backend.upper()} 执行 STFT")
        for completed, (input_index, output_index) in enumerate(_chunk_ranges(plan), 1):
            token.raise_if_cancelled()
            block = source[input_index]
            if plan.actual_backend == "gpu":
                compute_gpu(block, output_index, completed - 1)
            else:
                result, block_frequencies, block_times, block_selected = stft_block_cpu(
                    block,
                    params,
                    compute_dtype=plan.precision.compute_dtype,
                    output_dtype=plan.precision.output_dtype,
                )
                if frequencies is None:
                    frequencies = block_frequencies
                    times = block_times
                    selected = block_selected
                elif (
                    not _axes_match(frequencies, block_frequencies)
                    or not _axes_match(times, block_times)
                    or not np.array_equal(selected, block_selected)
                ):
                    raise RuntimeError("CPU STFT 分块返回了不一致的时间轴或频率选择")
                store_block(output_index, result)
            if progress is not None:
                progress(completed, total, f"正在使用 {plan.actual_backend.upper()} 执行 STFT")

        token.raise_if_cancelled()
        output = sink.commit() if sink is not None else destination
        return StftRunResult(
            output=output,
            frequencies=np.asarray(frequencies),
            times=np.asarray(times),
            selected_indices=np.asarray(selected, dtype=np.intp),
            whole_mean=whole_sum / (plan.request.shape[1] * plan.request.shape[2]),
            plan=plan,
        )
    except Exception:
        if sink is not None:
            sink.abort()
        raise
    finally:
        if worker is not None:
            worker.close()
        _close_loaded_mapping(source, loaded_here)


def run_stft_pipeline(
    plan,
    params,
    *,
    cache_dir,
    token: CancellationToken | None = None,
    progress=None,
    allow_cpu_fallback=True,
    device_index=0,
):
    token = token or CancellationToken()
    try:
        return _execute_stft(
            plan,
            params,
            cache_dir=cache_dir,
            token=token,
            progress=progress,
            device_index=device_index,
        )
    except TaskCancelled:
        raise
    except CudaBackendError as exc:
        if plan.actual_backend != "gpu" or not allow_cpu_fallback:
            raise
        fallback_reason = f"GPU 执行失败，已按设置回退 CPU: {exc}"
        cpu_plan = replace(
            plan,
            actual_backend="cpu",
            backend_reason=fallback_reason,
            peak_device_bytes=0,
        )
        result = _execute_stft(
            cpu_plan,
            params,
            cache_dir=cache_dir,
            token=token,
            progress=progress,
            device_index=device_index,
        )
        return replace(result, fallback_reason=fallback_reason)
