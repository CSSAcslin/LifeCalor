from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def estimate_resident_bytes(objects) -> int:
    """Count unique in-memory arrays without resolving ArrayRef-backed fields."""
    seen = set()
    total = 0

    def add(value):
        nonlocal total
        if isinstance(value, np.ndarray) and id(value) not in seen:
            seen.add(id(value))
            total += int(value.nbytes)
        elif isinstance(value, dict):
            for item in value.values():
                add(item)

    for obj in objects:
        if obj is None:
            continue
        values = getattr(obj, "__dict__", {})
        for key in ("_data_origin_storage", "_image_import_storage", "_data_processed_storage"):
            add(values.get(key))
        add(values.get("out_processed"))
    return total


@dataclass(frozen=True)
class MemoryBudget:
    limit_bytes: int

    @classmethod
    def from_megabytes(cls, value):
        return cls(max(256, int(value or 4096)) * 1024 * 1024)

    def ensure(self, additional_bytes: int, resident_bytes: int = 0, operation="导入"):
        projected = max(0, int(additional_bytes)) + max(0, int(resident_bytes))
        if projected > self.limit_bytes:
            need = projected / (1024 ** 3)
            limit = self.limit_bytes / (1024 ** 3)
            raise MemoryError(
                f"{operation}预计占用约 {need:.2f} GB，超过当前全局内存预算 {limit:.2f} GB。"
                "请在历史与缓存管理中提高预算、清理历史，或选择较小数据集。"
            )
        return projected


def array_nbytes(shape, dtype) -> int:
    return int(math.prod(int(value) for value in shape) * np.dtype(dtype).itemsize)