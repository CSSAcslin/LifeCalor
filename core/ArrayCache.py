import logging
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import numpy as np

DEFAULT_CACHE_THRESHOLD_BYTES = 512 * 1024 * 1024
ProgressCallback = Optional[Callable[[int, int, str], None]]


@dataclass(frozen=True)
class ArrayCacheConfig:
    cache_dir: Path
    threshold_bytes: int = DEFAULT_CACHE_THRESHOLD_BYTES
    slow_write_seconds: float = 5.0

    def ensure_dir(self) -> Path:
        path = Path(self.cache_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path


@dataclass(frozen=True)
class ArrayRef:
    path: Path
    shape: tuple
    dtype: str
    nbytes: int
    created_at: float
    field_name: str

    def load(self, mmap_mode: str = "r+"):
        if not Path(self.path).exists():
            raise FileNotFoundError(f"缓存文件不存在: {self.path}")
        return np.load(self.path, mmap_mode=mmap_mode)


class ArrayStore:
    def __init__(self, config: ArrayCacheConfig, progress_callback: ProgressCallback = None):
        self.config = config
        self.progress_callback = progress_callback
        self.config.ensure_dir()

    def put_array(self, array: np.ndarray, owner_id: str, field_name: str) -> ArrayRef:
        cache_dir = self.config.ensure_dir()
        safe_owner = _safe_name(owner_id)
        safe_field = _safe_name(field_name)
        path = cache_dir / f"{safe_owner}_{safe_field}_{uuid.uuid4().hex}.npy"
        total = int(array.nbytes)
        start = time.perf_counter()
        logging.info(
            "准备写入缓存: field=%s shape=%s dtype=%s nbytes=%s path=%s",
            safe_field,
            tuple(array.shape),
            array.dtype,
            total,
            path,
        )
        if self.progress_callback:
            self.progress_callback(0, total, f"正在写入缓存: {safe_field}")
        try:
            np.save(path, array)
        except Exception:
            logging.exception(
                "缓存读取失败: field=%s shape=%s dtype=%s nbytes=%s path=%s",
                safe_field,
                tuple(array.shape),
                array.dtype,
                total,
                path,
            )
            raise
        elapsed = time.perf_counter() - start
        if elapsed >= float(self.config.slow_write_seconds):
            logging.warning(
                "缓存写入耗时较长: field=%s shape=%s dtype=%s nbytes=%s elapsed=%.3fs path=%s",
                safe_field,
                tuple(array.shape),
                array.dtype,
                total,
                elapsed,
                path,
            )
        if self.progress_callback:
            self.progress_callback(total, total, f"缓存写入完成: {safe_field}")
        return ArrayRef(
            path=path,
            shape=tuple(array.shape),
            dtype=str(array.dtype),
            nbytes=total,
            created_at=time.time(),
            field_name=safe_field,
        )

    def load_ref(self, ref: ArrayRef, mmap_mode=None):
        if not isinstance(ref, ArrayRef):
            return ref
        if self.progress_callback:
            self.progress_callback(0, ref.nbytes, f"正在读取缓存: {ref.field_name}")
        try:
            loaded = ref.load(mmap_mode=mmap_mode)
            if mmap_mode is None:
                loaded = np.array(loaded)
        except Exception:
            logging.exception(
                "缓存读取失败: field=%s shape=%s dtype=%s nbytes=%s path=%s",
                ref.field_name,
                ref.shape,
                ref.dtype,
                ref.nbytes,
                ref.path,
            )
            raise
        if self.progress_callback:
            self.progress_callback(ref.nbytes, ref.nbytes, f"读取缓存完成: {ref.field_name}")
        return loaded

    def delete_ref(self, ref: ArrayRef) -> bool:
        if not isinstance(ref, ArrayRef):
            return False
        path = Path(ref.path)
        try:
            if path.exists() and path.is_file():
                path.unlink()
                logging.info("已删除缓存文件: %s", path)
                return True
        except Exception:
            logging.exception("删除缓存文件失败: %s", path)
        return False

    def clear_all(self) -> int:
        cache_dir = self.config.ensure_dir()
        deleted = 0
        for path in cache_dir.glob("*.npy"):
            try:
                if path.is_file():
                    path.unlink()
                    deleted += 1
            except Exception:
                logging.exception("删除缓存文件失败: %s", path)
        logging.info("缓存目录清理完成: deleted=%s dir=%s", deleted, cache_dir)
        return deleted

    def cleanup_orphans(self, active_refs: Iterable[ArrayRef]) -> int:
        cache_dir = self.config.ensure_dir()
        active_paths = {Path(ref.path).resolve() for ref in active_refs if isinstance(ref, ArrayRef)}
        deleted = 0
        for path in cache_dir.glob("*.npy"):
            try:
                if path.resolve() not in active_paths:
                    path.unlink()
                    deleted += 1
            except Exception:
                logging.exception("清理孤立缓存文件失败: %s", path)
        logging.info("孤立缓存清理完成: deleted=%s active=%s dir=%s", deleted, len(active_paths), cache_dir)
        return deleted


def _safe_name(value: Any) -> str:
    text = str(value)
    for char in ('/', '\\', ':', '*', '?', '"', '<', '>', '|', ' '):
        text = text.replace(char, '_')
    return text[:80]


def array_nbytes(value: Any) -> int:
    return int(value.nbytes) if isinstance(value, np.ndarray) else 0


def should_cache_array(value: Any, config: ArrayCacheConfig) -> bool:
    return isinstance(value, np.ndarray) and array_nbytes(value) > int(config.threshold_bytes)


def resolve_array(value: Any, mmap_mode: str = "r+"):
    if isinstance(value, ArrayRef):
        return value.load(mmap_mode=mmap_mode)
    return value


def cache_large_arrays_in_mapping(mapping: dict, store: ArrayStore, owner_id: str) -> dict:
    cached = {}
    for key, value in (mapping or {}).items():
        if should_cache_array(value, store.config):
            cached[key] = store.put_array(value, owner_id=owner_id, field_name=f"out_processed_{key}")
        else:
            cached[key] = value
    return cached
