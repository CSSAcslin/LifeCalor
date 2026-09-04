from __future__ import annotations

import logging
import time
from collections import OrderedDict

from PyQt5.QtCore import QObject, pyqtSignal

from .model import AppError
from .presenter import ErrorPresenter


class ErrorBroker(QObject):
    present_requested = pyqtSignal(object, object)

    def __init__(self, parent=None, dedup_seconds=5.0, presenter=None):
        super().__init__(parent)
        self.dedup_seconds = float(dedup_seconds)
        self.presenter = presenter or ErrorPresenter(self)
        self.present_requested.connect(self.presenter.enqueue)
        self._recent = OrderedDict()

    def report(self, parent, error: AppError):
        if error._reported:
            return error
        error._reported = True
        now = time.monotonic()
        previous = self._recent.get(error.fingerprint)
        duplicate = previous is not None and now - previous[0] <= self.dedup_seconds
        repeat_count = previous[1] + 1 if duplicate else 1
        self._recent[error.fingerprint] = (now, repeat_count)
        self._recent.move_to_end(error.fingerprint)
        while len(self._recent) > 256:
            self._recent.popitem(last=False)

        if duplicate:
            logging.warning(
                "重复错误已抑制: error_id=%s repeat=%d title=%s stage=%s",
                error.error_id, repeat_count, error.title, error.stage or "",
                extra={"lifecalor_user_reported": True, "ui_silent": True},
            )
            return error

        self._log(error)
        if error.severity in {"error", "critical"} and error.popup_policy != "never":
            self.present_requested.emit(parent, error)
        return error

    @staticmethod
    def _log(error):
        method = {
            "information": logging.info,
            "info": logging.info,
            "warning": logging.warning,
            "error": logging.error,
            "critical": logging.critical,
        }.get(error.severity, logging.error)
        message = f"[{error.error_id}] {error.title}: {error.message}"
        if error.task_id:
            message += f"\ntask_id: {error.task_id}"
        if error.details:
            message += f"\n{error.details}"
        method(message, extra={"lifecalor_user_reported": True, "ui_silent": True})

    def reset(self):
        self._recent.clear()
        self.presenter.reset()


_BROKER = None


def get_error_broker():
    global _BROKER
    if _BROKER is None:
        _BROKER = ErrorBroker()
    return _BROKER