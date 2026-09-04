import os
import logging
import sys
import traceback
import threading

from PyQt5.QtCore import QObject, pyqtSignal, Qt
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,
                             QTextEdit, QLineEdit, QPushButton, QProgressBar)


class ConsoleHandler(QObject, logging.Handler):
    append_log = pyqtSignal(str)
    unhandled_error = pyqtSignal(object, object, object)

    def __init__(self, parent):
        super().__init__()
        self.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.append_log.connect(parent.log_to_console)
        self.unhandled_error.connect(parent.handle_unhandled_exception)

        sys.excepthook = self.handle_uncaught_exception
        threading.excepthook = self.handle_thread_exception

    def emit(self, record):
        msg = self.format(record)
        self.append_log.emit(msg)


    def handle_uncaught_exception(self, exc_type, exc_value, exc_traceback):
        """Forward one complete uncaught exception to the GUI error broker."""
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        self.unhandled_error.emit(exc_type, exc_value, exc_traceback)

    def handle_thread_exception(self, args):
        if args.exc_type is SystemExit:
            return
        self.unhandled_error.emit(args.exc_type, args.exc_value, args.exc_traceback)



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
