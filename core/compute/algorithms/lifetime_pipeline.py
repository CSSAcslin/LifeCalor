from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import convolve

from tasks.model import CancellationToken, TaskCancelled

from compute.algorithms.lifetime import fit_lifetime, has_correlated_window
from compute.executor import _close_loaded_mapping, _source_array


@dataclass(frozen=True)
class LifetimeRunResult:
    lifetime_map: np.ndarray
    r_squared_map: np.ndarray
    plan: object


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
    lifetime_map = np.zeros((height, width), dtype=np.float64)
    r_squared_map = np.zeros((height, width), dtype=np.float64)
    for row in range(height):
        for column in range(width):
            trace = block[:, row, column]
            if not has_correlated_window(trace, time_points):
                continue
            try:
                result = fit_lifetime(
                    data_type,
                    trace,
                    time_points,
                    fit_params,
                    model_type=model_type,
                )
                if isinstance(result.lifetime, tuple):
                    raise ValueError("Double-exponential map reduction is not defined")
                if np.isfinite(result.lifetime):
                    lifetime_map[row, column] = result.lifetime
                if np.isfinite(result.r_squared):
                    r_squared_map[row, column] = result.r_squared
            except Exception:
                continue
    return lifetime_map, r_squared_map


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
):
    if plan.actual_backend != "cpu":
        raise ValueError("寿命拟合 GPU 后端尚未通过数值与性能验收")
    if model_type != "single":
        raise ValueError(
            "双指数热图尚未定义如何将两个寿命归约为单个像素值；"
            "请使用选区双指数拟合。"
        )
    token = token or CancellationToken()
    source, loaded_here = _source_array(plan.request.source)
    lifetime_map = np.zeros(plan.output_shape, dtype=np.float64)
    r_squared_map = np.zeros(plan.output_shape, dtype=np.float64)
    tiles = list(_tiles(plan))
    completed = 0
    pool = None

    def store(tile, result):
        nonlocal completed
        row_start, row_stop, column_start, column_stop = tile
        lifetime_map[row_start:row_stop, column_start:column_stop] = result[0]
        r_squared_map[row_start:row_stop, column_start:column_stop] = result[1]
        completed += 1
        if progress is not None:
            progress(completed, len(tiles), "正在执行寿命拟合")

    try:
        if progress is not None:
            progress(0, len(tiles), "正在准备寿命拟合")
        workers = max(1, int(cpu_workers))
        if workers == 1:
            for tile in tiles:
                token.raise_if_cancelled()
                block = _prepare_block(source, tile, pre_kernel)
                store(
                    tile,
                    _fit_lifetime_block(
                        (block, data_type, time_points, model_type, fit_params)
                    ),
                )
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
            pool.close()
            pool.join()
            pool = None
        token.raise_if_cancelled()
        return LifetimeRunResult(lifetime_map, r_squared_map, plan)
    except Exception:
        if pool is not None:
            pool.terminate()
            pool.join()
        raise
    finally:
        _close_loaded_mapping(source, loaded_here)
