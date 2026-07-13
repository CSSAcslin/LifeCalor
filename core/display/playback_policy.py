from __future__ import annotations

import math


def playback_interval_ms(fps, frame_count: int, default_total_ms: int = 15_000) -> int:
    """Return a safe timer interval for data with optional or invalid FPS metadata."""
    try:
        numeric_fps = float(fps)
    except (TypeError, ValueError):
        numeric_fps = 0.0
    if math.isfinite(numeric_fps) and numeric_fps > 0:
        return max(1, round(1000.0 / numeric_fps))
    return max(1, int(default_total_ms) // max(1, int(frame_count)))
