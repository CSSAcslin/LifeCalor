from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class OperandSpec:
    alias: str
    source: Any
    payload_key: str | None = None
    slice_text: str = ""
    axes: str = ""


@dataclass
class CalculationPlan:
    expression: str
    operands: list[OperandSpec]
    result_name: str = ""
    metadata_source_alias: str = ""
    metadata_overrides: dict = field(default_factory=dict)
    metadata_defaults: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ShapeStep:
    expression: str
    input_shapes: tuple[tuple[int, ...], ...]
    output_shape: tuple[int, ...]
    dtype: str
    status: str = "valid"
    message: str = ""


@dataclass
class ValidationResult:
    valid: bool
    steps: list[ShapeStep] = field(default_factory=list)
    output_shape: tuple[int, ...] = ()
    output_dtype: str = ""
    estimated_bytes: int = 0
    error: str = ""
    warnings: list[str] = field(default_factory=list)
