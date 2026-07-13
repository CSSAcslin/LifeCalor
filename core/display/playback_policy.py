from __future__ import annotations

import math

import numpy as np


def normalized_fps(fps) -> float | None:
    """Normalize optional FPS metadata to a finite positive float or None."""
    try:
        value = float(fps)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value > 0 else None


def valid_time_axis(time_point, frame_count: int | None = None):
    if time_point is None:
        return None
    try:
        axis = np.asarray(time_point).reshape(-1)
    except Exception:
        return None
    if frame_count is not None and axis.size < int(frame_count):
        return None
    return axis


def playback_interval_ms(fps, frame_count: int, default_total_ms: int = 15_000) -> int:
    """Return a safe timer interval for data with optional or invalid FPS metadata."""
    value = normalized_fps(fps)
    if value is not None:
        return max(1, round(1000.0 / value))
    return max(1, int(default_total_ms) // max(1, int(frame_count)))


def timeline_label(frame: int, fps=None, time_point=None) -> str:
    axis = valid_time_axis(time_point)
    if axis is not None and 0 <= int(frame) < axis.size:
        return str(axis[int(frame)])
    value = normalized_fps(fps)
    if value is None:
        return str(int(frame))
    seconds = int(frame) / value
    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)
    subframes = int(round((seconds - int(seconds)) * value))
    digits = max(1, len(str(max(1, int(round(value))))))
    return f"{minutes:02d}:{remaining_seconds:02d}:{subframes:0{digits}d}"
