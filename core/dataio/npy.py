from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import numpy as np

from tasks.model import CancellationToken

ProgressCallback = Callable[[int, int, str], None]
DEFAULT_CHUNK_BYTES = 32 * 1024 * 1024


def _emit(callback, current: int, total: int, message: str) -> None:
    if callback is not None:
        callback(int(current), int(total), message)


def _chunk_slices(shape: tuple[int, ...], dtype, chunk_bytes: int):
    if not shape:
        yield ()
        return
    row_bytes = max(1, int(np.prod(shape[1:], dtype=np.int64)) * np.dtype(dtype).itemsize)
    rows = max(1, int(chunk_bytes) // row_bytes)
    for start in range(0, shape[0], rows):
        yield slice(start, min(shape[0], start + rows))


def write_npy_atomic(path, array: np.ndarray, progress: ProgressCallback = None,
                     token: CancellationToken | None = None, message: str = "正在写入缓存",
                     chunk_bytes: int = DEFAULT_CHUNK_BYTES) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.writing.npy")
    source = np.asarray(array)
    total = int(source.nbytes)
    token = token or CancellationToken()
    _emit(progress, 0, total, message)
    destination = None
    completed = False
    try:
        destination = np.lib.format.open_memmap(
            temporary,
            mode="w+",
            dtype=source.dtype,
            shape=source.shape,
            fortran_order=bool(source.flags.f_contiguous and not source.flags.c_contiguous),
        )
        copied = 0
        for index in _chunk_slices(source.shape, source.dtype, chunk_bytes):
            token.raise_if_cancelled()
            destination[index] = source[index]
            copied += int(np.asarray(source[index]).nbytes)
            _emit(progress, min(copied, total), total, message)
        destination.flush()
        del destination
        destination = None
        token.raise_if_cancelled()
        os.replace(temporary, target)
        completed = True
        _emit(progress, total, total, f"{message}完成")
        return target
    finally:
        if destination is not None:
            del destination
        if not completed:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def copy_npy_to_memory(path, progress: ProgressCallback = None, token: CancellationToken | None = None,
                       message: str = "正在读取缓存", chunk_bytes: int = DEFAULT_CHUNK_BYTES) -> np.ndarray:
    source = np.load(Path(path), mmap_mode="r", allow_pickle=False)
    token = token or CancellationToken()
    total = int(source.nbytes)
    destination = np.empty(source.shape, dtype=source.dtype, order="F" if source.flags.f_contiguous else "C")
    copied = 0
    _emit(progress, 0, total, message)
    for index in _chunk_slices(source.shape, source.dtype, chunk_bytes):
        token.raise_if_cancelled()
        destination[index] = source[index]
        copied += int(np.asarray(source[index]).nbytes)
        _emit(progress, min(copied, total), total, message)
    token.raise_if_cancelled()
    _emit(progress, total, total, f"{message}完成")
    return destination
