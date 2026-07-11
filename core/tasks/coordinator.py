from __future__ import annotations

import logging

from PyQt5.QtCore import QObject, pyqtSignal

from .model import TaskRecord, TaskRegistry, TaskStatus


class TaskCoordinator(QObject):
    task_added = pyqtSignal(object)
    task_updated = pyqtSignal(object)
    task_finished = pyqtSignal(object)
    foreground_changed = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.registry = TaskRegistry()

    def create_task(self, name: str, category: str, foreground: bool = True, cancellable: bool = True,
                    cancel_callback=None) -> TaskRecord:
        task = self.registry.create(name, category, foreground, cancellable)
        task.cancel_callback = cancel_callback
        self.task_added.emit(task)
        if foreground:
            self.foreground_changed.emit(task)
        logging.info("任务已创建: id=%s category=%s name=%s", task.task_id, category, name)
        return task

    def start(self, task_id: str, total: int = 0, message: str = "") -> None:
        task = self._require(task_id)
        task.start(total, message)
        self.task_updated.emit(task)

    def progress(self, task_id: str, current: int, total: int = 0, message: str = "") -> None:
        task = self._require(task_id)
        task.advance(current, total, message)
        self.task_updated.emit(task)

    def complete(self, task_id: str, message: str = "") -> None:
        task = self._require(task_id)
        if task.status in {TaskStatus.COMPLETED, TaskStatus.CANCELLED, TaskStatus.FAILED}:
            return
        task.complete(message)
        self.task_updated.emit(task)
        self.task_finished.emit(task)
        logging.info("任务完成: id=%s category=%s name=%s", task.task_id, task.category, task.name)

    def cancelled(self, task_id: str, message: str = "任务已取消") -> None:
        task = self._require(task_id)
        if task.status in {TaskStatus.COMPLETED, TaskStatus.CANCELLED, TaskStatus.FAILED}:
            return
        task.cancel(message)
        self.task_updated.emit(task)
        self.task_finished.emit(task)
        logging.warning("任务已取消: id=%s category=%s name=%s", task.task_id, task.category, task.name)

    def fail(self, task_id: str, error: str) -> None:
        task = self._require(task_id)
        if task.status in {TaskStatus.COMPLETED, TaskStatus.CANCELLED, TaskStatus.FAILED}:
            return
        task.fail(error)
        self.task_updated.emit(task)
        self.task_finished.emit(task)
        logging.error("任务失败: id=%s category=%s name=%s error=%s", task.task_id, task.category, task.name, error, extra={"lifecalor_user_reported": True})

    def cancel_task(self, task_id: str) -> bool:
        requested = self.registry.request_cancel(task_id)
        task = self.registry.get(task_id)
        if requested and task is not None:
            self.task_updated.emit(task)
            logging.info("任务取消请求已发送: id=%s name=%s", task.task_id, task.name)
        return requested

    def cancel_foreground_task(self) -> bool:
        task = self.registry.request_cancel_foreground()
        if task is None:
            return False
        self.task_updated.emit(task)
        logging.info("前台任务取消请求已发送: id=%s name=%s", task.task_id, task.name)
        return True

    def active_tasks(self) -> list[TaskRecord]:
        return self.registry.active()

    def _require(self, task_id: str) -> TaskRecord:
        task = self.registry.get(task_id)
        if task is None:
            raise KeyError(f"未知任务: {task_id}")
        return task
