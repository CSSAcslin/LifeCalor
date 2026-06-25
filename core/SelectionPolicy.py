from typing import Any, Callable, Iterable, Optional

import numpy as np


def _matches_type(processed_type: Any, aim_type: Any) -> bool:
    if aim_type == "all":
        return True
    if isinstance(aim_type, str):
        return processed_type == aim_type
    return processed_type in aim_type


def select_data(mode: int, raw_data: Any, processed_data: Any, aim_type: Any = "all", picker: Optional[Callable[[], Any]] = None) -> Any:
    if raw_data is None and processed_data is None:
        return None
    if mode in (0, 2):
        return picker() if picker is not None else None
    if mode != 1:
        return None
    if aim_type == "data":
        return raw_data
    if aim_type == "all":
        return picker() if picker is not None else None
    if processed_data is None:
        return None
    if aim_type == "processed":
        return processed_data
    if _matches_type(getattr(processed_data, "type_processed", None), aim_type):
        return processed_data
    history: Iterable[Any] = getattr(processed_data, "history", []) or []
    return next((data for data in reversed(list(history)) if _matches_type(getattr(data, "type_processed", None), aim_type)), None)


def rect_mask_from_canvas(canvas: Any):
    rect_mask = canvas.v_rect_roi
    x, y, w, h = rect_mask[0][0], rect_mask[0][1], rect_mask[1], rect_mask[2]
    if w == 0 or h == 0:
        return None
    mask = np.zeros(canvas.data.framesize, dtype=bool)
    mask[y:y + h, x:x + w] = True
    return mask
