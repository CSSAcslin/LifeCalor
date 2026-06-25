import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

DEFAULT_CACHE_THRESHOLD_BYTES = 512 * 1024 * 1024
ProgressCallback = Optional[Callable[[int, int, str], None]]


@dataclass(frozen=True)
class ArrayCacheConfig:
    cache_dir: Path
    threshold_bytes: int = DEFAULT_CACHE_THRESHOLD_BYTES

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
        if self.progress_callback:
            self.progress_callback(0, total, f"正在写入缓存: {safe_field}")
        np.save(path, array)
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
