from __future__ import annotations

import logging
from typing import Any, Callable

from .policy import select_data
from .roi import select_roi


class SelectionController:
    def __init__(
        self,
        window: Any,
        data_dialog_factory: Callable[..., Any] | None = None,
        roi_dialog_factory: Callable[..., Any] | None = None,
        warning: Callable[[Any, str, str], Any] | None = None,
        logger: Any = logging,
    ):
        self.window = window
        self.data_dialog_factory = data_dialog_factory
        self.roi_dialog_factory = roi_dialog_factory
        self.warning = warning
        self.logger = logger

    def select_data(self, aim_type: str | list = "all"):
        aim_data = select_data(
            mode=self.window.mode,
            raw_data=self.window.data,
            processed_data=self.window.processed_data,
            aim_type=aim_type,
            picker=self.pick_data,
        )
        if self.window.data is None and self.window.processed_data is None:
            self.logger.warning("无数据可处理，请先加载数据")
        elif aim_data is None:
            self.logger.warning("未找到可处理的目标数据")
        return aim_data

    def pick_data(self, need_all=True):
        dialog_factory = self.data_dialog_factory or self._default_data_dialog_factory
        dialog = dialog_factory(
            datadict=self.window.get_data_all(),
            processed_datadict=self.window.get_processed_data_all(),
            add_canvas=False,
            parent=self.window,
        )
        if not dialog.exec_():
            return None

        selected_timestamp, selected_table = dialog.get_selected_timestamp()
        aim_data = self._find_selected_data(selected_timestamp, selected_table)
        if aim_data is None:
            self._warn("数据错误", "没有选取数据")
            return None
        self.logger.info(self._selection_log_message(selected_table, aim_data))
        return aim_data

    def select_roi(self, select=False):
        roi_dialog_factory = self.roi_dialog_factory or self._default_roi_dialog_factory
        return select_roi(
            mode=self.window.mode,
            image_display=self.window.image_display,
            select=select,
            roi_dialog_factory=roi_dialog_factory,
            parent=self.window,
            warning=self._warn,
        )

    def _find_selected_data(self, selected_timestamp, selected_table):
        if selected_table == 'data':
            history = getattr(getattr(self.window, 'data', None), 'history', []) or []
        else:
            history = getattr(getattr(self.window, 'processed_data', None), 'history', []) or []
        return next((data for data in history if data.timestamp == selected_timestamp), None)

    def _selection_log_message(self, selected_table, data):
        if selected_table == 'data':
            return f"数据选择成功（初始导入）：{data.name}"
        return f"数据选择成功（处理过）：{data.name}"

    def _warn(self, *args):
        if len(args) == 2:
            parent = self.window
            title, message = args
            warning_args = (title, message)
        elif len(args) == 3:
            parent, title, message = args
            warning_args = (parent, title, message)
        else:
            raise TypeError("_warn expects title/message or parent/title/message")
        if self.warning is not None:
            return self.warning(*warning_args)
        from diagnostics import report_warning
        return report_warning(parent, title, message)

    @staticmethod
    def _default_data_dialog_factory(**kwargs):
        from ExtraDialog import DataViewAndSelectPop
        return DataViewAndSelectPop(**kwargs)

    @staticmethod
    def _default_roi_dialog_factory(*args, **kwargs):
        from ExtraDialog import ROIInfoDialog
        return ROIInfoDialog(*args, **kwargs)
