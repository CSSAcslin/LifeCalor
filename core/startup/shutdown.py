from __future__ import annotations

import atexit
import logging
import sys

from PyQt5 import sip
from PyQt5.QtCore import QEvent, QObject, QThread, QTimer


class ApplicationResources:
    """Detach GUI logging before Qt releases its owned objects."""

    def __init__(self, app):
        self.app = app
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        self._prepared = False
        self.closed = False

    def prepare_quit(self):
        """Schedule deletion without reentering the Qt close/signal stack."""
        if self._prepared:
            return
        self._prepared = True
        from ConsoleUtils import ConsoleHandler

        root = logging.getLogger()
        for handler in tuple(root.handlers):
            if isinstance(handler, ConsoleHandler):
                root.removeHandler(handler)
                handler.close()
        if type(sys.stdout).__name__ == "StreamLogger":
            sys.stdout = self.stdout
        if type(sys.stderr).__name__ == "StreamLogger":
            sys.stderr = self.stderr

        logging.info("应用退出资源准备：等待 Qt 事件循环退栈")
        for handler in root.handlers:
            handler.flush()

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.prepare_quit()
        logging.info("应用日志解绑完成，Qt 对象按自身所有权退出")
        atexit.unregister(self.close)


def install_application_cleanup(app):
    resources = getattr(app, "_lifecalor_application_resources", None)
    if resources is None:
        resources = ApplicationResources(app)
        app._lifecalor_application_resources = resources
        app.aboutToQuit.connect(resources.prepare_quit)
        atexit.register(resources.close)
    return resources


class WindowShutdownController(QObject):
    """Keep the GUI responsive until cooperative workers have stopped."""

    WORKERS = (
        ("import_thread", "imp_thread"),
        ("data_thread", "dat_thread"),
        ("process_thread", "proc_thread"),
        ("calc_thread", "cal_thread"),
        ("avi_thread", "mass_data_processor"),
    )

    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self._finished = False
        self._threads = []
        self._pending_threads = set()
        self._preview_pool = None
        self._timer = QTimer(self)
        self._timer.setInterval(25)
        self._timer.timeout.connect(self._poll)
        window.installEventFilter(self)

    def eventFilter(self, watched, event):
        if watched is self.window and event.type() == QEvent.Close:
            if self._finished:
                return False
            event.ignore()
            self.request_close()
            return True
        return False

    def _collect_threads(self):
        threads = set(self.window.findChildren(QThread))
        threads.update(
            value for value in vars(self.window).values()
            if isinstance(value, QThread)
        )
        compute_dialog = sys.modules.get("compute.dialog")
        if compute_dialog is not None:
            threads.update(
                thread for thread in compute_dialog._ACTIVE_PROBE_THREADS
                if isinstance(thread, QThread)
            )
        return [thread for thread in threads if not sip.isdeleted(thread)]

    def request_close(self):
        if getattr(self.window, "_lifecalor_closing", False):
            return
        self.window._lifecalor_closing = True
        self.window.setEnabled(False)
        save_timer = getattr(self.window, "save_timer", None)
        if save_timer is not None:
            save_timer.stop()
        save_params = getattr(self.window, "save_params", None)
        if save_params is not None:
            save_params()
        display = getattr(self.window, "image_display", None)
        if display is not None:
            display.layout_manager.flush_save()

        coordinator = getattr(self.window, "task_coordinator", None)
        if coordinator is not None:
            for task in coordinator.active_tasks():
                coordinator.cancel_task(task.task_id)
        for thread_name, worker_name in self.WORKERS:
            worker = getattr(self.window, worker_name, None)
            thread = getattr(self.window, thread_name, None)
            if worker is None or sip.isdeleted(worker):
                continue
            cancel = getattr(worker, "cancel", None) or getattr(worker, "stop", None)
            if cancel is not None:
                cancel()
            if isinstance(thread, QThread) and not sip.isdeleted(thread):
                thread.finished.connect(worker.deleteLater)

        self._threads = self._collect_threads()
        history = getattr(self.window, "history_controller", None)
        if history is not None:
            self._threads.extend(history.prepare_shutdown())
        self._preview_pool = getattr(self.window, "_convolution_preview_pool", None)
        if self._preview_pool is not None:
            self._preview_pool.prepare_shutdown()
        if display is not None:
            display.del_canvas(-1)
        for thread in self._threads:
            try:
                thread.finished.disconnect(thread.deleteLater)
            except TypeError:
                pass
            thread.finished.connect(self._thread_finished)
            if thread.isRunning():
                self._pending_threads.add(thread)
            thread.requestInterruption()
            thread.quit()
        logging.info("正在关闭应用，已请求任务取消和工作线程退出")
        self._timer.start()

    def _thread_finished(self):
        self._pending_threads.discard(self.sender())

    def _poll(self):
        if self._preview_pool is not None and self._preview_pool.shutdown_pending():
            return
        self._pending_threads = {
            thread for thread in self._pending_threads if not sip.isdeleted(thread)
        }
        if self._pending_threads:
            return
        if any(
            not sip.isdeleted(thread) and not thread.wait(0)
            for thread in self._threads
        ):
            return
        self._timer.stop()
        self._finished = True
        for thread in self._threads:
            if not sip.isdeleted(thread):
                thread.deleteLater()
        logging.info("工作线程已退出，正在关闭工作区")
        self.window.close()
