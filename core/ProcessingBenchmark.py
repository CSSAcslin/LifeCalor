from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable, Tuple


@dataclass(frozen=True)
class BenchmarkResult:
    operation: str
    elapsed_seconds: float
    output_shape: Tuple[int, ...]


def _shape_of(value: Any) -> Tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is not None:
        return tuple(shape)
    if isinstance(value, (list, tuple)):
        if value and isinstance(value[0], (list, tuple)):
            return (len(value), len(value[0]))
        return (len(value),)
    return ()


def measure_operation(operation: str, func: Callable[[], Any]) -> BenchmarkResult:
    start = perf_counter()
    output = func()
    elapsed = perf_counter() - start
    return BenchmarkResult(operation=operation, elapsed_seconds=elapsed, output_shape=_shape_of(output))
