from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass, field, replace
from typing import Mapping

import numpy as np
from scipy.ndimage import convolve

from tasks.model import CancellationToken, TaskCancelled

from compute.algorithms.lifetime import (
    LifetimeFitStatus,
    fit_lifetime,
    has_correlated_window,
)
from compute.executor import _close_loaded_mapping, _source_array
from compute.io import ComputeOutputSet
from compute.worker import CudaBackendError, CudaWorkerClient


@dataclass(frozen=True)
class LifetimeRunResult:
    lifetime_map: np.ndarray
    r_squared_map: np.ndarray
    plan: object
    named_outputs: Mapping[str, np.ndarray] = field(default_factory=dict)
    model_type: str = "single"
    fallback_reason: str = ""


def spatial_kernel(kernel_type, half_size):
    size = max(1, int(half_size) * 2 - 1)
    if size == 1:
        return np.ones((1, 1), dtype=np.float64)
    if kernel_type == "smooth":
        if size == 3:
            return np.array(
                [[0.1, 0.1, 0.1], [0.1, 0.2, 0.1], [0.1, 0.1, 0.1]]
            )
        center = size // 2
        maximum = np.sqrt(2 * center ** 2)
        yy, xx = np.indices((size, size))
        distance = np.sqrt((yy - center) ** 2 + (xx - center) ** 2)
        kernel = np.where(distance == 0, 2.0, 2.0 - distance / maximum)
        return kernel / kernel.sum()
    if kernel_type == "gaussian":
        sigma = max(0.3 * ((size - 1) * 0.5 - 1) + 0.8, 0.1)
        axis = np.linspace(-(size - 1) / 2.0, (size - 1) / 2.0, size)
        xx, yy = np.meshgrid(axis, axis)
        kernel = np.exp(-0.5 * (xx ** 2 + yy ** 2) / sigma ** 2)
        return kernel / kernel.sum()
    if kernel_type == "sharpen":
        if size == 3:
            return np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        kernel = -np.ones((size, size)) / (size * size - 1)
        kernel[size // 2, size // 2] = 2
        return kernel
    if kernel_type == "edge":
        if size == 3:
            return np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        kernel = np.ones((size, size))
        kernel[size // 2, size // 2] = -(size * size - 1)
        return kernel
    if kernel_type == "laplacian":
        if size == 3:
            return np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        kernel = np.zeros((size, size))
        center = size // 2
        kernel[center, center] = -4
        if center > 0:
            kernel[center - 1, center] = 1
            kernel[center + 1, center] = 1
            kernel[center, center - 1] = 1
            kernel[center, center + 1] = 1
        return kernel
    if kernel_type == "average":
        return np.ones((size, size)) / (size * size)
    return None


def _fit_lifetime_block(arguments):
    block, data_type, time_points, model_type, fit_params = arguments
    _, height, width = block.shape
    shape = (height, width)
    r_squared_map = np.zeros(shape, dtype=np.float64)
    fit_status = np.full(
        shape, int(LifetimeFitStatus.NOT_EVALUATED), dtype=np.uint8
    )
    if model_type == "single":
        outputs = {
            "lifetime_map": np.zeros(shape, dtype=np.float64),
            "r_squared_map": r_squared_map,
            "fit_status": fit_status,
        }
    elif model_type == "double":
        outputs = {
            "tau1_map": np.zeros(shape, dtype=np.float64),
            "tau2_map": np.zeros(shape, dtype=np.float64),
            "amplitude1_map": np.zeros(shape, dtype=np.float64),
            "amplitude2_map": np.zeros(shape, dtype=np.float64),
            "baseline_map": np.zeros(shape, dtype=np.float64),
            "r_squared_map": r_squared_map,
            "fit_status": fit_status,
        }
    else:
        raise ValueError(f"Unsupported lifetime model: {model_type}")

    for row in range(height):
        for column in range(width):
            trace = block[:, row, column]
            if not has_correlated_window(trace, time_points):
                fit_status[row, column] = int(
                    LifetimeFitStatus.NO_CORRELATED_WINDOW
                )
                continue
            try:
                result = fit_lifetime(
                    data_type,
                    trace,
                    time_points,
                    fit_params,
                    model_type=model_type,
                )
                fit_status[row, column] = int(result.status)
                if np.isfinite(result.r_squared):
                    r_squared_map[row, column] = result.r_squared
                if model_type == "single":
                    if np.isfinite(result.lifetime):
                        outputs["lifetime_map"][row, column] = result.lifetime
                else:
                    parameters = np.asarray(result.parameters, dtype=np.float64)
                    if parameters.shape == (5,):
                        names = (
                            "amplitude1_map",
                            "tau1_map",
                            "amplitude2_map",
                            "tau2_map",
                            "baseline_map",
                        )
                        for name, value in zip(names, parameters):
                            if np.isfinite(value):
                                outputs[name][row, column] = value
            except Exception:
                fit_status[row, column] = int(LifetimeFitStatus.FIT_FAILED)
                continue
    return outputs


def _tiles(plan):
    _, height, width = plan.request.shape
    _, row_step, column_step = plan.chunk_shape
    for row_start in range(0, height, row_step):
        row_stop = min(height, row_start + row_step)
        for column_start in range(0, width, column_step):
            column_stop = min(width, column_start + column_step)
            yield row_start, row_stop, column_start, column_stop


def _prepare_block(source, tile, kernel):
    row_start, row_stop, column_start, column_stop = tile
    if kernel is None:
        return np.asarray(source[:, row_start:row_stop, column_start:column_stop])
    radius = kernel.shape[0] // 2
    expanded_rows = slice(max(0, row_start - radius), min(source.shape[1], row_stop + radius))
    expanded_columns = slice(
        max(0, column_start - radius), min(source.shape[2], column_stop + radius)
    )
    expanded = np.asarray(source[:, expanded_rows, expanded_columns]).copy()
    for frame in range(expanded.shape[0]):
        expanded[frame] = convolve(expanded[frame], kernel, mode="mirror")
    local_row = row_start - expanded_rows.start
    local_column = column_start - expanded_columns.start
    return expanded[
        :,
        local_row:local_row + (row_stop - row_start),
        local_column:local_column + (column_stop - column_start),
    ]


def _split_lifetime_tile(block, tile):
    row_start, row_stop, column_start, column_stop = tile
    height, width = block.shape[1:]
    if height <= 1 and width <= 1:
        return ()
    if height >= width and height > 1:
        first = height // 2
        middle = row_start + first
        return (
            (block[:, :first, :], (row_start, middle, column_start, column_stop)),
            (block[:, first:, :], (middle, row_stop, column_start, column_stop)),
        )
    first = width // 2
    middle = column_start + first
    return (
        (block[:, :, :first], (row_start, row_stop, column_start, middle)),
        (block[:, :, first:], (row_start, row_stop, middle, column_stop)),
    )


def _execute_lifetime_pipeline(
    plan,
    *,
    data_type,
    time_points,
    fit_params,
    model_type="single",
    pre_kernel=None,
    cpu_workers=1,
    token=None,
    progress=None,
    device_index=0,
    cache_dir=None,
):
    if model_type not in {"single", "double"}:
        raise ValueError(f"Unsupported lifetime model: {model_type}")
    token = token or CancellationToken()
    source, loaded_here = _source_array(plan.request.source)
    output_dtypes = {
        "lifetime_map": "float64",
        "r_squared_map": "float64",
        "fit_status": "uint8",
    } if model_type == "single" else {
        "tau1_map": "float64",
        "tau2_map": "float64",
        "amplitude1_map": "float64",
        "amplitude2_map": "float64",
        "baseline_map": "float64",
        "r_squared_map": "float64",
        "fit_status": "uint8",
    }
    output_set = None
    if plan.output_to_disk:
        if cache_dir is None:
            raise ValueError("寿命多结果落盘需要可写缓存目录")
        output_set = ComputeOutputSet(
            cache_dir,
            plan.request.task_id,
            plan.request.attempt_id,
            {
                name: (plan.output_shape, dtype)
                for name, dtype in output_dtypes.items()
            },
            token=token,
        )
        named_outputs = {}
    elif model_type == "single":
        named_outputs = {
            "lifetime_map": np.zeros(plan.output_shape, dtype=np.float64),
            "r_squared_map": np.zeros(plan.output_shape, dtype=np.float64),
            "fit_status": np.full(
                plan.output_shape,
                int(LifetimeFitStatus.NOT_EVALUATED),
                dtype=np.uint8,
            ),
        }
    else:
        named_outputs = {
            name: np.zeros(
                plan.output_shape,
                dtype=np.uint8 if name == "fit_status" else np.float64,
            )
            for name in (
                "tau1_map",
                "tau2_map",
                "amplitude1_map",
                "amplitude2_map",
                "baseline_map",
                "r_squared_map",
                "fit_status",
            )
        }
    tiles = list(_tiles(plan))
    completed = 0
    pool = None
    worker = None

    def store(tile, result):
        row_start, row_stop, column_start, column_stop = tile
        index = (
            slice(row_start, row_stop),
            slice(column_start, column_stop),
        )
        for name, values in result.items():
            if output_set is not None:
                output_set.write_block(name, index, values)
            else:
                named_outputs[name][index] = values

    def finish_tile():
        nonlocal completed
        completed += 1
        if progress is not None:
            progress(completed, len(tiles), "正在执行寿命拟合")

    try:
        if progress is not None:
            progress(0, len(tiles), "正在准备寿命拟合")
        workers = max(1, int(cpu_workers))
        if plan.actual_backend == "gpu":
            worker = CudaWorkerClient(device_index)

            def compute_gpu(block, tile, block_id, retry=0):
                try:
                    result = worker.execute(
                        plan.request.algorithm,
                        task_id=plan.request.task_id,
                        attempt_id=plan.request.attempt_id,
                        block_id=block_id,
                        block=block,
                        time_points=np.asarray(time_points, dtype=np.float64),
                        data_type=data_type,
                        fit_params=dict(fit_params),
                        model_type=model_type,
                        compute_dtype=plan.precision.compute_dtype,
                        output_dtype=plan.precision.output_dtype,
                        token=token,
                        timeout=600.0,
                    )
                except CudaBackendError as exc:
                    split = _split_lifetime_tile(block, tile)
                    if exc.out_of_memory and retry < 3 and split:
                        for child_number, (child, child_tile) in enumerate(split):
                            compute_gpu(
                                child,
                                child_tile,
                                f"{block_id}.{child_number}",
                                retry + 1,
                            )
                        return
                    raise
                store(tile, result)

            for block_id, tile in enumerate(tiles):
                token.raise_if_cancelled()
                block = _prepare_block(source, tile, pre_kernel)
                compute_gpu(block, tile, block_id)
                finish_tile()
        elif workers == 1:
            for tile in tiles:
                token.raise_if_cancelled()
                block = _prepare_block(source, tile, pre_kernel)
                store(
                    tile,
                    _fit_lifetime_block(
                        (block, data_type, time_points, model_type, fit_params)
                    ),
                )
                finish_tile()
        else:
            pool = mp.get_context("spawn").Pool(processes=workers)
            batch_size = max(1, workers * 2)
            for offset in range(0, len(tiles), batch_size):
                pending = []
                for tile in tiles[offset:offset + batch_size]:
                    token.raise_if_cancelled()
                    block = _prepare_block(source, tile, pre_kernel)
                    pending.append((tile, pool.apply_async(
                        _fit_lifetime_block,
                        ((block, data_type, time_points, model_type, fit_params),),
                    )))
                for tile, async_result in pending:
                    while not async_result.ready():
                        if token.is_cancelled:
                            raise TaskCancelled("寿命计算已取消")
                        async_result.wait(0.05)
                    store(tile, async_result.get())
                    finish_tile()
            pool.close()
            pool.join()
            pool = None
        token.raise_if_cancelled()
        if output_set is not None:
            named_outputs = output_set.commit()
        r_squared_map = named_outputs["r_squared_map"]
        if model_type == "single":
            lifetime_map = named_outputs["lifetime_map"]
        else:
            lifetime_map = named_outputs["tau1_map"]
        return LifetimeRunResult(
            lifetime_map,
            r_squared_map,
            plan,
            named_outputs=named_outputs,
            model_type=model_type,
        )
    except Exception:
        if output_set is not None:
            output_set.abort()
        if pool is not None:
            pool.terminate()
            pool.join()
        raise
    finally:
        if worker is not None:
            worker.close()
        _close_loaded_mapping(source, loaded_here)


def run_lifetime_pipeline(
    plan,
    *,
    data_type,
    time_points,
    fit_params,
    model_type="single",
    pre_kernel=None,
    cpu_workers=1,
    token=None,
    progress=None,
    allow_cpu_fallback=True,
    device_index=0,
    cache_dir=None,
):
    token = token or CancellationToken()
    try:
        return _execute_lifetime_pipeline(
            plan,
            data_type=data_type,
            time_points=time_points,
            fit_params=fit_params,
            model_type=model_type,
            pre_kernel=pre_kernel,
            cpu_workers=cpu_workers,
            token=token,
            progress=progress,
            device_index=device_index,
            cache_dir=cache_dir,
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
        result = _execute_lifetime_pipeline(
            cpu_plan,
            data_type=data_type,
            time_points=time_points,
            fit_params=fit_params,
            model_type=model_type,
            pre_kernel=pre_kernel,
            cpu_workers=cpu_workers,
            token=token,
            progress=progress,
            device_index=device_index,
            cache_dir=cache_dir,
        )
        return replace(result, fallback_reason=fallback_reason)
