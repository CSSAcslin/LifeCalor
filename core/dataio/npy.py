from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import numpy as np

from tasks.model import CancellationToken

ProgressCallback = Callable[[int, int, str], None]
DEFAULT_CHUNK_BYTES = 32 * 1024 * 1024


class AtomicNpySink:
    """Incrementally write one NPY file and expose it only after commit."""

    def __init__(self, target, shape, dtype, *, progress: ProgressCallback = None,
                 token: CancellationToken | None = None, message="正在写入计算结果",
                 fortran_order=False):
        self.target = Path(target)
        self.target.parent.mkdir(parents=True, exist_ok=True)
        self.temporary = self.target.with_name(f".{self.target.name}.writing.npy")
        self.shape = tuple(int(value) for value in shape)
        self.dtype = np.dtype(dtype)
        self.total_bytes = int(np.prod(self.shape, dtype=object)) * self.dtype.itemsize
        self.progress = progress
        self.token = token or CancellationToken()
        self.message = str(message)
        self.written_bytes = 0
        self._committed = False
        self._closed = False
        self._array = np.lib.format.open_memmap(
            self.temporary,
            mode="w+",
            dtype=self.dtype,
            shape=self.shape,
            fortran_order=bool(fortran_order),
        )
        _emit(self.progress, 0, self.total_bytes, self.message)

    def write_block(self, index, values) -> None:
        if self._closed:
            raise RuntimeError("结果写入器已经关闭")
        self.token.raise_if_cancelled()
        block = np.asarray(values)
        expected = self._array[index].shape
        if block.shape != expected:
            raise ValueError(f"结果块尺寸不匹配: expected={expected}, actual={block.shape}")
        self._array[index] = block
        block_bytes = int(np.prod(expected, dtype=object)) * self.dtype.itemsize
        self.written_bytes = min(self.total_bytes, self.written_bytes + block_bytes)
        _emit(self.progress, self.written_bytes, self.total_bytes, self.message)

    def _close_mapping(self) -> None:
        if self._closed:
            return
        self._array.flush()
        mapping = getattr(self._array, "_mmap", None)
        if mapping is not None:
            mapping.close()
        self._array = None
        self._closed = True

    def commit(self) -> Path:
        if self._committed:
            return self.target
        self.token.raise_if_cancelled()
        if self.written_bytes != self.total_bytes:
            raise RuntimeError("计算结果尚未完整写入，不能提交")
        self._close_mapping()
        self.token.raise_if_cancelled()
        os.replace(self.temporary, self.target)
        self._committed = True
        _emit(self.progress, self.total_bytes, self.total_bytes, f"{self.message}完成")
        return self.target

    def abort(self) -> None:
        try:
            self._close_mapping()
        finally:
            if not self._committed:
                try:
                    self.temporary.unlink(missing_ok=True)
                except OSError:
                    pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None or not self._committed:
            self.abort()
        return False


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
