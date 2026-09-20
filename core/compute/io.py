from __future__ import annotations

import os
import shutil
import time
import uuid
from pathlib import Path

import numpy as np

from ArrayCache import ArrayRef
from dataio.npy import AtomicNpySink
from tasks.model import CancellationToken

from .model import plan_execution_details
from .planner import normalize_progress


def _safe_name(value) -> str:
    text = str(value)
    for char in ('/', '\\', ':', '*', '?', '"', '<', '>', '|', ' '):
        text = text.replace(char, "_")
    return text[:80] or "result"


class TaskProgressReporter:
    """Route large byte counters through the existing task coordinator."""

    def __init__(self, coordinator, task_id: str, maximum: int = 10000):
        self.coordinator = coordinator
        self.task_id = task_id
        self.maximum = int(maximum)

    def __call__(self, current: int, total: int, message: str) -> None:
        scaled, maximum = normalize_progress(current, total, self.maximum)
        self.coordinator.progress(self.task_id, scaled, maximum, message)


def bind_plan_to_task(coordinator, plan, stage="资源规划"):
    details = plan_execution_details(plan)
    coordinator.configure_execution(
        plan.request.task_id,
        attempt_id=plan.request.attempt_id,
        stage=stage,
        requested_backend=plan.requested_backend.value,
        actual_backend=plan.actual_backend,
        precision=plan.precision.policy.value,
        device=plan.selected_device,
        backend_reason=plan.backend_reason,
        resource_summary=details["resource_summary"],
        execution_details=details,
    )
    return TaskProgressReporter(coordinator, plan.request.task_id)

class ComputeNpySink:
    """Task-scoped output sink that becomes an ArrayRef only after commit."""

    def __init__(
        self,
        cache_dir,
        task_id: str,
        attempt_id: int,
        field_name: str,
        shape,
        dtype,
        *,
        progress=None,
        token: CancellationToken | None = None,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.task_id = _safe_name(task_id)
        self.attempt_id = max(0, int(attempt_id))
        self.field_name = _safe_name(field_name)
        self.shape = tuple(int(value) for value in shape)
        self.dtype = np.dtype(dtype)
        self.token = token or CancellationToken()
        self.task_dir = self.cache_dir / ".compute_tasks" / self.task_id / str(self.attempt_id)
        self.task_dir.mkdir(parents=True, exist_ok=True)
        self.stage_path = self.task_dir / f"{self.field_name}.npy"
        self.final_path = self.cache_dir / (
            f"{self.task_id}_{self.field_name}_{uuid.uuid4().hex}.npy"
        )
        self._sink = AtomicNpySink(
            self.stage_path,
            self.shape,
            self.dtype,
            progress=progress,
            token=self.token,
            message=f"正在写入计算结果: {self.field_name}",
        )
        self._committed_ref = None
        self._minimum = None
        self._maximum = None
        self._sum = 0.0
        self._count = 0

    def write_block(self, index, values) -> None:
        block = np.asarray(values)
        self._sink.write_block(index, block)
        if block.size and not np.iscomplexobj(block):
            block_min = block.min()
            block_max = block.max()
            self._minimum = block_min if self._minimum is None else min(self._minimum, block_min)
            self._maximum = block_max if self._maximum is None else max(self._maximum, block_max)
            self._sum += block.sum(dtype=np.float64)
            self._count += int(block.size)

    def commit(self) -> ArrayRef:
        if self._committed_ref is not None:
            return self._committed_ref
        staged = self._sink.commit()
        try:
            self.token.raise_if_cancelled()
            os.replace(staged, self.final_path)
        except Exception:
            Path(staged).unlink(missing_ok=True)
            raise
        self._cleanup_task_dir()
        self._committed_ref = ArrayRef(
            path=self.final_path,
            shape=self.shape,
            dtype=self.dtype.name,
            nbytes=int(np.prod(self.shape, dtype=object)) * self.dtype.itemsize,
            created_at=time.time(),
            field_name=self.field_name,
            min_value=self._minimum,
            max_value=self._maximum,
            mean_value=(self._sum / self._count) if self._count else None,
        )
        return self._committed_ref

    def abort(self) -> None:
        self._sink.abort()
        self.stage_path.unlink(missing_ok=True)
        self._cleanup_task_dir()

    def _cleanup_task_dir(self) -> None:
        root = self.cache_dir / ".compute_tasks"
        try:
            shutil.rmtree(self.task_dir, ignore_errors=True)
            parent = self.task_dir.parent
            if parent.exists() and not any(parent.iterdir()):
                parent.rmdir()
            if root.exists() and not any(root.iterdir()):
                root.rmdir()
        except OSError:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None or self._committed_ref is None:
            self.abort()
        return False
