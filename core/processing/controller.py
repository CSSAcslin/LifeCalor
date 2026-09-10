from __future__ import annotations

from collections import defaultdict, deque

from PyQt5 import sip
from PyQt5.QtCore import QThread

from ThreadController import is_thread_active
from tasks import TaskStatus


class ProcessingController:
    """Coordinate persistent workers with per-operation task records."""

    DEFAULT_NAMES = {
        "calculation": "寿命计算",
        "em_processing": "数据处理",
        "roi_processing": "ROI 处理",
    }

    def __init__(self, window, coordinator):
        self.window = window
        self.coordinator = coordinator
        self._queues = defaultdict(deque)

    def begin(self, thread_name: str, category: str, name: str | None = None):
        thread = getattr(self.window, thread_name, None)
        worker = self._worker(category)

        def cancel():
            method = getattr(worker, "cancel", None) or getattr(worker, "stop", None)
            if callable(method):
                method()
            if thread is not None and hasattr(thread, "requestInterruption"):
                thread.requestInterruption()

        task = self.coordinator.create_task(
            name or self.DEFAULT_NAMES.get(category, category), category,
            foreground=True, cancellable=True, cancel_callback=cancel,
        )
        token_setter = getattr(worker, "set_cancellation_token", None)
        if callable(token_setter):
            token_setter(task.token)
        if worker is not None and hasattr(worker, "abortion"):
            worker.abortion = False
        self._queues[category].append(task.task_id)
        self.coordinator.start(task.task_id, message=task.name)
        if thread is not None and not is_thread_active(thread, expected_type=QThread, is_deleted=sip.isdeleted):
            thread.start()
        return task

    def active(self, category: str):
        queue = self._queues[category]
        while queue:
            task = self.coordinator.registry.get(queue[0])
            if task is not None and task.status in {
                TaskStatus.PENDING, TaskStatus.RUNNING, TaskStatus.CANCELLING,
            }:
                return task
            queue.popleft()
        return None

    def progress(self, category: str, current: int, total: int = 0, message: str = ""):
        task = self.active(category)
        if task is not None:
            self.coordinator.progress(task.task_id, current, total, message or task.name)
            return task
        self.window.update_progress(current, total)
        return None

    def complete(self, category: str, message: str = "数据处理完成"):
        task = self.active(category)
        if task is not None:
            self.coordinator.complete(task.task_id, message)
        return task

    def cancelled(self, category: str, message: str = "任务已取消"):
        task = self.active(category)
        if task is not None:
            self.coordinator.cancelled(task.task_id, message)
        return task

    def fail(self, error, categories=("calculation", "em_processing", "roi_processing")):
        task = self.coordinator.registry.get(getattr(error, "task_id", None) or "")
        if task is None:
            for category in categories:
                task = self.active(category)
                if task is not None:
                    break
        if task is None:
            return None
        error.task_id = task.task_id
        task.diagnostic = error
        self.coordinator.fail(task.task_id, error.message)
        return task

    def complete_for_result(self, process_type: str):
        calculation_results = {
            "ROI_lifetime", "lifetime_distribution", "diffusion",
            "heat_transfer", "signal_average",
        }
        em_results = {
            "EM_pre_processed", "stft_quality", "cwt_quality", "ROI_stft", "ROI_cwt",
            "Accumulated_time_amplitude_map", "Single_channel_signal",
            "2D_Fourier_transform", "2D_Inverse_Fourier_transform", "Heartbeat",
        }
        if process_type in calculation_results:
            return self.complete("calculation")
        if process_type in em_results:
            return self.complete("em_processing")
        if process_type == "Roi_applied":
            return self.complete("roi_processing")
        return None

    def _worker(self, category):
        return {
            "calculation": getattr(self.window, "cal_thread", None),
            "em_processing": getattr(self.window, "mass_data_processor", None),
            "roi_processing": getattr(self.window, "dat_thread", None),
        }.get(category)