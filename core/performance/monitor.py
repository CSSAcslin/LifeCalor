from __future__ import annotations

import logging
import time
from dataclasses import dataclass

try:
    import psutil
except ImportError:
    psutil = None


@dataclass(frozen=True)
class OperationMetrics:
    name: str
    elapsed_seconds: float
    memory_delta_bytes: int
    output_bytes: int = 0


class PerformanceRecorder:
    def __init__(self, name, output_bytes=0):
        self.name = str(name)
        self.output_bytes = int(output_bytes or 0)
        self.metrics = None

    @staticmethod
    def _rss():
        return psutil.Process().memory_info().rss if psutil is not None else 0

    def __enter__(self):
        self._started = time.perf_counter()
        self._rss_started = self._rss()
        return self

    def __exit__(self, exc_type, _exc, _tb):
        self.metrics = OperationMetrics(
            self.name, time.perf_counter() - self._started,
            self._rss() - self._rss_started, self.output_bytes,
        )
        level = logging.WARNING if exc_type is not None else logging.INFO
        logging.log(
            level, "性能记录: operation=%s elapsed=%.3fs memory_delta=%d output_bytes=%d",
            self.metrics.name, self.metrics.elapsed_seconds,
            self.metrics.memory_delta_bytes, self.metrics.output_bytes,
        )
        return False