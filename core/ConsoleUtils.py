import os
import logging
import sys
import traceback
import time

from PyQt5.QtCore import QObject, pyqtSignal, Qt
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,
                             QTextEdit, QLineEdit, QPushButton, QProgressBar)


class ConsoleHandler(QObject, logging.Handler):
    append_log = pyqtSignal(str)
    unhandled_error = pyqtSignal(object, object, object)
    legacy_error = pyqtSignal(str, str)

    def __init__(self, parent):
        super().__init__()
        self.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.append_log.connect(parent.log_to_console)
        self.unhandled_error.connect(parent.handle_unhandled_exception)
        self.legacy_error.connect(parent.handle_logged_error)

        self._last_uncaught = None
        self._last_uncaught_at = 0.0
        self._suppressed_uncaught = 0
        sys.excepthook = self.handle_uncaught_exception

    def emit(self, record):
        msg = self.format(record)
        self.append_log.emit(msg)
        if record.levelno >= logging.ERROR and not getattr(record, 'lifecalor_user_reported', False):
            self.legacy_error.emit(record.levelname, record.getMessage())

    def handle_uncaught_exception(self, exc_type, exc_value, exc_traceback):
        """处理所有未捕获的异常"""
        if issubclass(exc_type, KeyboardInterrupt):
            # 忽略键盘中断(CTRL+C)
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        error_msg = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        fingerprint = (exc_type.__name__, str(exc_value), error_msg)
        now = time.monotonic()
        if fingerprint == self._last_uncaught and now - self._last_uncaught_at < 2.0:
            self._suppressed_uncaught += 1
            return
        if self._suppressed_uncaught:
            logging.warning("已抑制 %d 条短时间重复的未捕获异常", self._suppressed_uncaught)
            self._suppressed_uncaught = 0
        self._last_uncaught = fingerprint
        self._last_uncaught_at = now
        logging.critical("未捕获异常:\n%s", error_msg, extra={"lifecalor_user_reported": True})
        self.unhandled_error.emit(exc_type, exc_value, exc_traceback)





class ConsoleWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # 日志输出区域
        self.console_output = QTextEdit()
        self.console_output.setReadOnly(True)
        self.console_output.setStyleSheet("""
            QTextEdit {
                background-color: black;
                color: #00FF00;
                font-family: Consolas;
                font-size: 10pt;
            }
        """)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid grey;
                border-radius: 3px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #05B8CC;
                width: 10px;
            }
        """)
        self.progress_bar.hide()

        layout.addWidget(self.console_output)
        layout.addWidget(self.progress_bar)
        # layout.addLayout(command_layout)
