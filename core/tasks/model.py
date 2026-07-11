from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    FAILED = "failed"
    COMPLETED = "completed"


class TaskCancelled(RuntimeError):
    pass


class CancellationToken:
    def __init__(self):
        self._event = threading.Event()

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def cancel(self) -> None:
        self._event.set()

    def raise_if_cancelled(self) -> None:
        if self.is_cancelled:
            raise TaskCancelled("任务已取消")


@dataclass
class TaskRecord:
    name: str
    category: str
    foreground: bool = True
    cancellable: bool = True
    task_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    status: TaskStatus = TaskStatus.PENDING
    current: int = 0
    total: int = 0
    message: str = ""
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    token: CancellationToken = field(default_factory=CancellationToken, repr=False)
    cancel_callback: Optional[Callable[[], None]] = field(default=None, repr=False)

    def start(self, total: int = 0, message: str = "") -> None:
        self.status = TaskStatus.RUNNING
        self.current = 0
        self.total = max(0, int(total))
        self.message = message
        self.error = None
        self.started_at = time.time()

    def advance(self, current: int, total: Optional[int] = None, message: str = "") -> None:
        if self.status not in (TaskStatus.RUNNING, TaskStatus.CANCELLING):
            return
        self.current = max(0, int(current))
        if total is not None:
            self.total = max(0, int(total))
        if message:
            self.message = message

    def request_cancel(self) -> bool:
        if not self.cancellable or self.status not in (TaskStatus.PENDING, TaskStatus.RUNNING):
            return False
        self.status = TaskStatus.CANCELLING
        self.token.cancel()
        if self.cancel_callback is not None:
            self.cancel_callback()
        return True

    def complete(self, message: str = "") -> None:
        self.status = TaskStatus.COMPLETED
        if self.total:
            self.current = self.total
        if message:
            self.message = message
        self.finished_at = time.time()

    def cancel(self, message: str = "任务已取消") -> None:
        self.status = TaskStatus.CANCELLED
        self.message = message
        self.finished_at = time.time()

    def fail(self, error: str) -> None:
        self.status = TaskStatus.FAILED
        self.error = str(error)
        self.finished_at = time.time()


class TaskRegistry:
    """Thread-safe task store that can back a future multi-task progress panel."""

    def __init__(self):
        self._tasks: dict[str, TaskRecord] = {}
        self._foreground_task_id: Optional[str] = None
        self._lock = threading.RLock()

    def create(self, name: str, category: str, foreground: bool = True, cancellable: bool = True) -> TaskRecord:
        task = TaskRecord(name=name, category=category, foreground=foreground, cancellable=cancellable)
        with self._lock:
            self._tasks[task.task_id] = task
            if foreground:
                self._foreground_task_id = task.task_id
        return task

    def get(self, task_id: str) -> Optional[TaskRecord]:
        with self._lock:
            return self._tasks.get(task_id)

    def all(self) -> list[TaskRecord]:
        with self._lock:
            return list(self._tasks.values())

    def active(self) -> list[TaskRecord]:
        active_states = {TaskStatus.PENDING, TaskStatus.RUNNING, TaskStatus.CANCELLING}
        return [task for task in self.all() if task.status in active_states]

    def foreground(self) -> Optional[TaskRecord]:
        with self._lock:
            task = self._tasks.get(self._foreground_task_id or "")
        if task is not None and task.status in {TaskStatus.PENDING, TaskStatus.RUNNING, TaskStatus.CANCELLING}:
            return task
        active = [task for task in self.active() if task.foreground]
        return active[-1] if active else None

    def request_cancel(self, task_id: str) -> bool:
        task = self.get(task_id)
        return bool(task and task.request_cancel())

    def request_cancel_foreground(self) -> Optional[TaskRecord]:
        task = self.foreground()
        if task is not None and task.request_cancel():
            return task
        return None
