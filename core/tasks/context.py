from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from functools import wraps

from .model import CancellationToken


@dataclass(frozen=True)
class TaskContext:
    task_id: str
    token: CancellationToken


class TaskContextQueue:
    """Match queued Qt slot invocations with task-owned cancellation tokens."""

    def __init__(self):
        self._items = deque()
        self._lock = threading.RLock()

    def enqueue(self, task_id, token):
        context = TaskContext(
            task_id=str(task_id or ""),
            token=token or CancellationToken(),
        )
        with self._lock:
            self._items.append(context)
        return context

    def next(self):
        with self._lock:
            return self._items.popleft() if self._items else None

    def __len__(self):
        with self._lock:
            return len(self._items)


def task_scoped(method):
    """Activate the next queued task context immediately before a worker slot."""

    @wraps(method)
    def wrapped(self, *args, **kwargs):
        activate = getattr(self, "_activate_task_context", None)
        if callable(activate) and not activate():
            return False
        return method(self, *args, **kwargs)

    return wrapped
