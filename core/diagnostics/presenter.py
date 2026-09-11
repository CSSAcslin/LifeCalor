from __future__ import annotations

import os
from collections import deque

from PyQt5.QtCore import QObject, QUrl
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import QApplication, QMessageBox, QWidget


class ErrorPresenter(QObject):
    """Present structured errors one at a time without stacking dialogs."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._queue = deque()
        self._active = None

    @property
    def pending_count(self):
        return len(self._queue)

    @property
    def is_showing(self):
        return self._active is not None

    def enqueue(self, parent, error):
        self._queue.append((parent, error))
        self._show_next()

    def _show_next(self):
        if self._active is not None or not self._queue:
            return
        if QApplication.instance() is None:
            self._queue.clear()
            return
        parent, error = self._queue.popleft()
        parent = parent if isinstance(parent, QWidget) else None
        try:
            box = QMessageBox(parent)
        except RuntimeError:
            box = QMessageBox()
        box.setIcon(QMessageBox.Critical)
        box.setWindowTitle(error.title)
        box.setText(error.message)
        box.setInformativeText(f"错误 ID: {error.error_id}")
        if error.details:
            box.setDetailedText(error.details)
        copy_button = box.addButton("复制详情", QMessageBox.ActionRole)
        log_button = box.addButton("打开日志目录", QMessageBox.ActionRole)
        box.addButton(QMessageBox.Ok)
        box.buttonClicked.connect(
            lambda button, c=copy_button, l=log_button, p=parent, e=error:
            self._handle_action(button, c, l, p, e)
        )
        box.finished.connect(self._finished)
        self._active = box
        box.show()

    @staticmethod
    def _handle_action(button, copy_button, log_button, parent, error):
        if button is copy_button:
            details = error.details or ""
            QApplication.clipboard().setText(
                f"{error.title}\n{error.message}\n错误 ID: {error.error_id}\n{details}".rstrip()
            )
        elif button is log_button:
            target = parent
            log_file = None
            while target is not None:
                log_file = getattr(target, "log_file", None)
                if log_file:
                    break
                target = target.parent() if callable(getattr(target, "parent", None)) else None
            if not log_file:
                log_file = error.context.get("log_file")
            if log_file:
                QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.dirname(str(log_file))))

    def _finished(self, _result):
        active = self._active
        self._active = None
        if active is not None:
            active.deleteLater()
        self._show_next()

    def reset(self):
        self._queue.clear()
        if self._active is not None:
            self._active.close()
            self._active = None
