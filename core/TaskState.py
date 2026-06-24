from dataclasses import dataclass
from enum import Enum
from typing import Optional


class TaskStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    CANCELLING = "cancelling"
    FAILED = "failed"
    COMPLETED = "completed"


@dataclass
class TaskState:
    name: str
    status: TaskStatus = TaskStatus.IDLE
    current: int = 0
    total: int = 0
    error: Optional[str] = None

    def start(self, total: int = 0) -> None:
        self.status = TaskStatus.RUNNING
        self.current = 0
        self.total = max(0, int(total))
        self.error = None

    def advance(self, current: int, total: Optional[int] = None) -> None:
        if self.status != TaskStatus.RUNNING:
            raise RuntimeError(f"Cannot advance task '{self.name}' while status is {self.status.value}")
        self.current = max(0, int(current))
        if total is not None:
            self.total = max(0, int(total))

    def request_cancel(self) -> None:
        if self.status == TaskStatus.RUNNING:
            self.status = TaskStatus.CANCELLING

    def complete(self) -> None:
        self.status = TaskStatus.COMPLETED
        if self.total and self.current < self.total:
            self.current = self.total
        self.error = None

    def fail(self, error: str) -> None:
        self.status = TaskStatus.FAILED
        self.error = str(error)
