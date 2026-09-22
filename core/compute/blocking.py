from __future__ import annotations

from dataclasses import dataclass
from itertools import product


@dataclass(frozen=True)
class BlockPlan:
    block_id: int
    core_slices: tuple[slice, ...]
    read_slices: tuple[slice, ...]
    local_core_slices: tuple[slice, ...]
    output_slices: tuple[slice, ...]
    pad_width: tuple[tuple[int, int], ...]


def _dimensions(values, ndim, name):
    values = tuple(int(value) for value in values)
    if len(values) != ndim or any(value < 0 for value in values):
        raise ValueError(f"{name} 必须包含 {ndim} 个非负整数")
    return values


def iter_block_plans(shape, *, core_shape, halo_before=None, halo_after=None):
    shape = tuple(int(value) for value in shape)
    if not shape or any(value <= 0 for value in shape):
        raise ValueError(f"无效数据尺寸: {shape}")
    ndim = len(shape)
    core_shape = _dimensions(core_shape, ndim, "core_shape")
    if any(value <= 0 for value in core_shape):
        raise ValueError("core_shape 必须大于 0")
    halo_before = _dimensions(halo_before or (0,) * ndim, ndim, "halo_before")
    halo_after = _dimensions(halo_after or (0,) * ndim, ndim, "halo_after")
    starts = [range(0, size, step) for size, step in zip(shape, core_shape)]
    for block_id, coordinates in enumerate(product(*starts)):
        core, read, local, padding = [], [], [], []
        for size, start, step, before, after in zip(shape, coordinates, core_shape, halo_before, halo_after):
            stop = min(size, start + step)
            desired_start, desired_stop = start - before, stop + after
            read_start, read_stop = max(0, desired_start), min(size, desired_stop)
            pad_before, pad_after = max(0, -desired_start), max(0, desired_stop - size)
            local_start = pad_before + start - read_start
            core.append(slice(start, stop))
            read.append(slice(read_start, read_stop))
            local.append(slice(local_start, local_start + stop - start))
            padding.append((pad_before, pad_after))
        core_tuple = tuple(core)
        yield BlockPlan(block_id, core_tuple, tuple(read), tuple(local), core_tuple, tuple(padding))
