"""Resolve scientific temporal coordinates and their units at analysis boundaries."""

import logging
import weakref

import numpy as np


_SECOND_BASED_RESULTS = {"ROI_stft", "ROI_cwt", "stft_quality", "cwt_quality"}
_TIME_UNITS = {"s", "ms", "us", "μs", "µs", "ns", "ps", "fs"}


def source_time_metadata(data):
    """Follow an ImagingData weak parent without requiring display array creation."""
    metadata = {}
    parent_ref = getattr(data, "parent_data", None)
    if isinstance(parent_ref, weakref.ReferenceType):
        parent = parent_ref()
        if parent is not None and parent is not data:
            metadata.update(getattr(parent, "parameters", None) or {})
            metadata.update(getattr(parent, "out_processed", None) or {})
            source_format = getattr(parent, "format_import", None)
            if source_format:
                metadata.setdefault("source_format_import", source_format)
    metadata.update(getattr(data, "parameters", None) or {})
    metadata.update(getattr(data, "out_processed", None) or {})
    return metadata


def _positive(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if np.isfinite(result) and result > 0 else 0.0


def scientific_time_axis(data, length):
    """Return physical coordinates and a known time unit, or the unit 'unknown'."""
    metadata = source_time_metadata(data)
    source_format = (
        getattr(data, "format_import", None)
        or metadata.get("source_format_import")
        or getattr(data, "source_format", None)
        or metadata.get("source_format", "")
    )
    processed_type = getattr(data, "type_processed", None) or getattr(data, "source_format", None) or source_format
    fps = _positive(metadata.get("fps"))
    time_step = _positive(metadata.get("time_step"))
    window_step = _positive(metadata.get("window_step"))
    unit = metadata.get("time_axis_unit") or metadata.get("time_unit") or "unknown"
    if unit == "µs":
        unit = "μs"
    if unit not in _TIME_UNITS:
        unit = "unknown"
    physical = metadata.get("time_axis_kind") == "physical"
    second_based = (
        metadata.get("time_basis") == "fps"
        or processed_type in _SECOND_BASED_RESULTS
        or source_format in {"avi_EM", "tif_EM"}
    )
    if processed_type in _SECOND_BASED_RESULTS or (second_based and not physical):
        unit = "s"

    axis = getattr(data, "time_point", None)
    if axis is not None:
        axis = np.asarray(axis).reshape(-1)
        if axis.size != length:
            axis = None
    if axis is None:
        candidate = metadata.get("time_series")
        if isinstance(candidate, (np.ndarray, list, tuple)) and np.size(candidate) == length:
            axis = np.asarray(candidate).reshape(-1)
            if processed_type in _SECOND_BASED_RESULTS:
                unit = "s"

    legacy_frames = not physical and (
        source_format == "tif_series"
        or (source_format == "sif" and metadata.get("data_type") == "sif_folder")
        or metadata.get("time_axis_kind") == "frames"
    )
    if axis is not None:
        if legacy_frames:
            return axis * (time_step or 1.0), unit
        return axis, unit

    logging.warning(
        "数据时间轴长度与结果不匹配，已重建: name=%s, result=%s",
        getattr(data, "source_name", getattr(data, "name", "")), length,
    )
    coordinates = np.arange(length, dtype=np.float64)
    if fps and window_step:
        return coordinates * window_step / fps, "s"
    if second_based and fps:
        return coordinates / fps, "s"
    if time_step:
        return coordinates * time_step, unit
    if fps:
        return coordinates / fps, "s"
    return coordinates, "unknown"
