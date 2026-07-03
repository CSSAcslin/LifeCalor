from __future__ import annotations

import logging
from typing import Any, Callable

from .policy import rect_mask_from_canvas


def select_roi(
    mode: int,
    image_display: Any,
    select: bool = False,
    roi_dialog_factory: Callable[..., Any] | None = None,
    parent: Any = None,
    warning: Callable[[Any, str, str], Any] | None = None,
):
    mask = None
    if mode == 2:
        mask = image_display.get_draw_roi()[1]
        if mask is None:
            logging.warning("选中画布没有绘制有效的ROI")
            return None
        return mask

    if mode == 0:
        if roi_dialog_factory is None:
            raise ValueError("roi_dialog_factory is required in ROI selection mode")
        dialog = roi_dialog_factory(image_display.get_all_canvas_info(), parent)
        if not dialog.exec_():
            return None
        aim_id = dialog.canvas_id
        if dialog.roi_type == 'pixel_roi':
            return image_display.get_draw_roi(aim_id)[1]
        if dialog.roi_type == 'v_line':
            if warning is not None:
                warning(parent, "警告", "不支持该类型")
            return None
        if dialog.roi_type == 'v_rect':
            return rect_mask_from_canvas(image_display.display_canvas[aim_id])
        if dialog.roi_type == 'anchor':
            return image_display.display_canvas[aim_id].anchor_mask
        return None

    if mode == 1 and select:
        mask = image_display.get_draw_roi()[1]
        if mask is None:
            logging.warning("选中画布没有绘制有效的ROI")
            return None
        return mask

    return None
