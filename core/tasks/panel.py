from __future__ import annotations

import time

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QDockWidget, QHBoxLayout, QLabel, QProgressBar, QPushButton, QTreeWidget,
    QTreeWidgetItem, QWidget,
)

from .model import TaskStatus


class TaskPanel(QDockWidget):
    """Compact multi-task view with independent progress and cancellation."""

    STATUS_TEXT = {
        TaskStatus.PENDING: "等待中", TaskStatus.RUNNING: "运行中",
        TaskStatus.CANCELLING: "正在取消", TaskStatus.CANCELLED: "已取消",
        TaskStatus.FAILED: "失败", TaskStatus.COMPLETED: "已完成",
    }

    def __init__(self, coordinator, parent=None):
        super().__init__("任务", parent)
        self.setObjectName("TaskPanelDock")
        self.coordinator = coordinator
        self._rows = {}
        self.tree = QTreeWidget()
        self.tree.setColumnCount(6)
        self.tree.setHeaderLabels(["任务", "来源/类别", "状态", "进度", "耗时", "操作"])
        self.tree.setAlternatingRowColors(True)
        self.setWidget(self.tree)
        coordinator.task_added.connect(self._add)
        coordinator.task_updated.connect(self._update)
        coordinator.task_finished.connect(self._update)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh_elapsed)
        self.timer.start(1000)

    def _add(self, task):
        item = QTreeWidgetItem(self.tree)
        item.setText(0, task.name)
        item.setText(1, task.category)
        progress = QProgressBar()
        progress.setRange(0, 0)
        progress.setTextVisible(True)
        cancel = QPushButton("取消")
        cancel.setToolTip("请求取消此任务；不会在界面线程中等待后台线程退出")
        cancel.clicked.connect(lambda _checked=False, task_id=task.task_id: self.coordinator.cancel_task(task_id))
        container = QWidget()
        row = QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(cancel)
        self.tree.setItemWidget(item, 3, progress)
        self.tree.setItemWidget(item, 5, container)
        self._rows[task.task_id] = (item, progress, cancel)
        self._update(task)

    def _update(self, task):
        if task.task_id not in self._rows:
            self._add(task)
            return
        item, progress, cancel = self._rows[task.task_id]
        item.setText(0, task.name)
        item.setText(1, task.category)
        item.setText(2, self.STATUS_TEXT.get(task.status, str(task.status)))
        if task.total > 0:
            percent = max(0, min(100, int(task.current * 100 / task.total)))
            progress.setRange(0, 100)
            progress.setValue(percent)
            progress.setFormat(f"{percent}%")
        else:
            progress.setRange(0, 0 if task.status in {TaskStatus.PENDING, TaskStatus.RUNNING} else 1)
            if task.status not in {TaskStatus.PENDING, TaskStatus.RUNNING}:
                progress.setValue(1)
        cancel.setEnabled(task.cancellable and task.status in {TaskStatus.PENDING, TaskStatus.RUNNING})
        detail = task.error or task.message or task.name
        for column in range(6):
            item.setToolTip(column, f"任务 ID: {task.task_id}\n{detail}")
        self._set_elapsed(task)

    def _set_elapsed(self, task):
        item, _, _ = self._rows[task.task_id]
        start = task.started_at or task.created_at
        end = task.finished_at or time.time()
        elapsed = max(0.0, end - start)
        item.setText(4, f"{elapsed:.1f} s" if elapsed < 60 else f"{elapsed / 60:.1f} min")

    def refresh_elapsed(self):
        for task in self.coordinator.registry.active():
            if task.task_id in self._rows:
                self._set_elapsed(task)