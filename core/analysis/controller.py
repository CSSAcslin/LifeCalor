from __future__ import annotations

from contextlib import contextmanager

import numpy as np
from PyQt5.QtWidgets import QAction

from diagnostics import report_warning
from selection.policy import select_data

from .catalog import ANALYSIS_ACTIONS, ACTION_BY_ID
from .dialog import AnalysisInputDialog


class AnalysisController:
    def __init__(self, window, dialog_factory=AnalysisInputDialog):
        self.window = window
        self.dialog_factory = dialog_factory
        self.actions = {}
        self.current_action = None
        self.current_input = None
        self.current_method = "mean"

    def build_menu(self, menu_bar):
        menu = menu_bar.addMenu("分析与处理")
        menu.setObjectName("AnalysisMenu")
        groups = {}
        for spec in ANALYSIS_ACTIONS:
            if not spec.category:
                continue
            group = groups.setdefault(spec.category, None)
            if group is None:
                group = menu.addMenu(spec.category)
                groups[spec.category] = group
            action = QAction(spec.title, self.window)
            action.setObjectName(spec.action_id)
            action.triggered.connect(
                lambda checked=False, key=spec.action_id: self.open(key)
            )
            group.addAction(action)
            self.actions[spec.action_id] = action
        curves = menu.addMenu("曲线分析")
        graph = self.window.graph_plot
        for name in ("action_psd", "action_hist", "action_time", "action_peak"):
            curves.addAction(getattr(graph, name))
        curves.aboutToShow.connect(graph.refresh_analysis_actions)
        menu.aboutToShow.connect(graph.refresh_analysis_actions)
        menu.addSeparator()
        spec = ACTION_BY_ID["motion.heartbeat"]
        motion = menu.addAction(spec.title)
        motion.setObjectName(spec.action_id)
        motion.triggered.connect(lambda checked=False: self.open("motion.heartbeat"))
        self.actions[spec.action_id] = motion
        return menu

    def open(self, action_id):
        spec = ACTION_BY_ID[action_id]
        self.window.show_analysis_page(spec)
        if spec.page:
            return
        return self.execute(action_id)

    @contextmanager
    def _prepared(self, spec, dialog):
        mask = None if dialog.mask is None else np.array(dialog.mask, dtype=bool, copy=True)
        previous = (self.current_action, self.current_input, self.current_method)
        self.current_action = spec
        self.current_input = dialog.source
        self.current_method = dialog.method
        try:
            with self.window.selection_controller.prepared(dialog.source, mask):
                yield
        finally:
            self.current_action, self.current_input, self.current_method = previous

    def execute(self, action_id):
        spec = ACTION_BY_ID[action_id]
        window = self.window
        window.show_analysis_page(spec)
        sources = window.calculator_sources()
        if not sources:
            report_warning(window, "输入数据", "请先导入数据，或从历史数据恢复可用数组。")
            return False
        aim_type = list(spec.processed_types) if spec.processed_types else "data"
        preferred = select_data(
            window.mode, window.data, window.processed_data, aim_type
        )
        dialog = self.dialog_factory(window, spec, sources, preferred)
        if not dialog.exec_():
            return False
        with self._prepared(spec, dialog):
            return getattr(window, spec.handler)()

    def connect_button(self, button, action_id):
        button.clicked.connect(lambda checked=False: self.execute(action_id))
