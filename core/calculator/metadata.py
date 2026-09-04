from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ArrayCache import ArrayRef
from .engine import CalculationEngine
from .model import CalculationPlan, ValidationResult


EXCLUDED_KEYS = {
    "formula_used", "calculator_sources", "shape_trace", "estimated_bytes",
    "colormap_backup", "image_backup", "data_backup",
}


@dataclass
class CalculationMetadata:
    out_processed: dict
    parameters: dict
    time_point: np.ndarray | None


class CalculationMetadataPolicy:
    """Build small, complete result metadata without inheriting unrelated arrays."""

    MAX_SEQUENCE_ITEMS = 4096

    @staticmethod
    def source_mapping(source):
        values = {}
        parameters = getattr(source, "parameters", None) or {}
        out_processed = getattr(source, "out_processed", None) or {}
        values.update(parameters)
        values.update(out_processed)
        return values

    @staticmethod
    def safe_value(value):
        if isinstance(value, (np.ndarray, ArrayRef)):
            return None
        if isinstance(value, (str, int, float, bool, type(None))):
            return value
        if isinstance(value, (tuple, list)):
            if len(value) > CalculationMetadataPolicy.MAX_SEQUENCE_ITEMS:
                return None
            if all(isinstance(item, (str, int, float, bool, type(None))) for item in value):
                return type(value)(value)
        return None

    @classmethod
    def preview(cls, plan: CalculationPlan, output_shape=()):
        source = cls._metadata_source(plan)
        inherited = cls.source_mapping(source) if source is not None else {}
        merged = {}
        for key, candidate in (plan.metadata_defaults or {}).items():
            value = cls.safe_value(candidate)
            if value is not None:
                merged[key] = value
        for key, candidate in inherited.items():
            if key in EXCLUDED_KEYS:
                continue
            value = cls.safe_value(candidate)
            if value is not None:
                merged[key] = value
        merged.update({key: value for key, value in (plan.metadata_overrides or {}).items() if value not in (None, "")})
        merged["scientific_axes"] = cls.output_axes(tuple(output_shape), merged.get("scientific_axes"))
        merged.setdefault("display_axes", merged["scientific_axes"])
        return merged

    @classmethod
    def warnings(cls, plan: CalculationPlan):
        units = {}
        for spec in plan.operands:
            metadata = cls.source_mapping(spec.source)
            unit = metadata.get("value_unit", metadata.get("unit"))
            if unit not in (None, ""):
                units[spec.alias] = str(unit)
        if len(set(units.values())) > 1:
            details = "、".join(f"{alias}={unit}" for alias, unit in units.items())
            return [f"输入数值单位不一致（{details}），请在结果参数中确认输出单位"]
        return []

    @classmethod
    def build(cls, plan: CalculationPlan, validation: ValidationResult, result):
        metadata = cls.preview(plan, result.shape)
        source = cls._metadata_source(plan)
        time_point = cls._time_axis(source, result.shape, metadata)
        provenance = {
            "formula_used": plan.expression,
            "calculator_sources": [
                {
                    "alias": item.alias,
                    "name": getattr(item.source, "name", ""),
                    "timestamp": getattr(item.source, "timestamp", None),
                    "payload_key": item.payload_key,
                    "slice": item.slice_text,
                    "shape": tuple(CalculationEngine.source_info(item).shape),
                    "axes": item.axes,
                }
                for item in plan.operands
            ],
            "shape_trace": [
                {
                    "expression": step.expression,
                    "input_shapes": step.input_shapes,
                    "output_shape": step.output_shape,
                    "dtype": step.dtype,
                    "status": step.status,
                    "message": step.message,
                }
                for step in validation.steps
            ],
            "estimated_bytes": validation.estimated_bytes,
        }
        return CalculationMetadata({**metadata, **provenance}, dict(metadata), time_point)

    @staticmethod
    def output_axes(shape, inherited=None):
        inherited = str(inherited or "")
        if len(inherited) == len(shape):
            return inherited.replace("Y", "H").replace("X", "W")
        return {1: "T", 2: "HW", 3: "THW", 4: "THWC"}.get(len(shape), "?" * len(shape))

    @staticmethod
    def _metadata_source(plan):
        if not plan.operands:
            return None
        alias = plan.metadata_source_alias or plan.operands[0].alias
        return next((item.source for item in plan.operands if item.alias == alias), plan.operands[0].source)

    @staticmethod
    def _time_axis(source, result_shape, metadata):
        if source is None or len(result_shape) not in (1, 3):
            return None
        points = getattr(source, "time_point", None)
        if points is not None:
            points = np.asarray(points).reshape(-1)
            if points.size == result_shape[0]:
                return points.copy()
        step = metadata.get("time_step")
        if step not in (None, 0):
            return np.arange(result_shape[0], dtype=np.float64) * float(step)
        fps = metadata.get("fps")
        if fps not in (None, 0):
            return np.arange(result_shape[0], dtype=np.float64) / float(fps)
        return None
