from __future__ import annotations

import atexit
import logging
import os
import sys
import traceback
from logging.handlers import RotatingFileHandler
from pathlib import Path

from app_metadata import APP_NAME


_FILE_HANDLER_MARKER = "lifecalor_file_handler"
_BUFFER_HANDLER_MARKER = "lifecalor_startup_buffer"
_LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
_QT_FILE_HANDLER = None
_QT_MESSAGE_HANDLER_INSTALLED = False
_PREVIOUS_QT_MESSAGE_HANDLER = None


def _qt_message_to_file(message_type, context, message):
    """Write Qt diagnostics directly to disk without calling GUI handlers."""
    handler = _QT_FILE_HANDLER
    if handler is None or handler._closed:
        return
    try:
        text = str(message)
        if "shared QObject was deleted directly" in text:
            text += "\n" + "".join(traceback.format_stack())
        levels = {
            0: logging.DEBUG,
            1: logging.WARNING,
            2: logging.ERROR,
            3: logging.CRITICAL,
            4: logging.INFO,
        }
        record = logging.LogRecord(
            "qt", levels.get(int(message_type), logging.WARNING),
            context.file or "", context.line, "Qt: %s", (text,), None,
        )
        handler.handle(record)
        handler.flush()
    except Exception:
        # A Qt message callback must not raise back into Qt, including during exit.
        pass


def _install_qt_file_logging(handler):
    global _QT_FILE_HANDLER, _QT_MESSAGE_HANDLER_INSTALLED, _PREVIOUS_QT_MESSAGE_HANDLER
    _QT_FILE_HANDLER = handler
    if not _QT_MESSAGE_HANDLER_INSTALLED:
        from PyQt5.QtCore import qInstallMessageHandler

        _PREVIOUS_QT_MESSAGE_HANDLER = qInstallMessageHandler(_qt_message_to_file)
        _QT_MESSAGE_HANDLER_INSTALLED = True
        atexit.register(uninstall_qt_file_logging)


def uninstall_qt_file_logging():
    """Detach the Python callback before Qt and Python tear down their objects."""
    global _QT_FILE_HANDLER, _QT_MESSAGE_HANDLER_INSTALLED, _PREVIOUS_QT_MESSAGE_HANDLER
    _QT_FILE_HANDLER = None
    if not _QT_MESSAGE_HANDLER_INSTALLED:
        return False
    from PyQt5.QtCore import qInstallMessageHandler

    qInstallMessageHandler(_PREVIOUS_QT_MESSAGE_HANDLER)
    _PREVIOUS_QT_MESSAGE_HANDLER = None
    _QT_MESSAGE_HANDLER_INSTALLED = False
    atexit.unregister(uninstall_qt_file_logging)
    return True


class StartupBufferHandler(logging.Handler):
    """Keep early records available for the in-app console without logging twice."""

    def __init__(self):
        super().__init__(logging.INFO)
        self.records = []
        self.setFormatter(logging.Formatter(_LOG_FORMAT))

    def emit(self, record):
        self.records.append(self.format(record))


def resolve_log_path() -> Path:
    frozen = bool(getattr(sys, "frozen", False) or hasattr(sys, "_MEIPASS"))
    if frozen:
        local_app_data = os.environ.get("LOCALAPPDATA")
        base = Path(local_app_data) if local_app_data else Path.home() / "AppData" / "Local"
        directory = base / APP_NAME
    else:
        directory = Path(__file__).resolve().parents[1]
    directory.mkdir(parents=True, exist_ok=True)
    return directory / "carrier_lifetime.log"


def ensure_file_logging(log_file=None):
    path = Path(log_file or resolve_log_path()).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for handler in list(logger.handlers):
        if getattr(handler, _FILE_HANDLER_MARKER, False):
            try:
                same_path = Path(handler.baseFilename).resolve() == path
            except (AttributeError, OSError):
                same_path = False
            if same_path:
                _install_qt_file_logging(handler)
                return handler
            logger.removeHandler(handler)
            handler.close()
    handler = RotatingFileHandler(
        path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    setattr(handler, _FILE_HANDLER_MARKER, True)
    logger.addHandler(handler)
    _install_qt_file_logging(handler)
    return handler


def install_early_logging(log_file=None) -> Path:
    path = Path(log_file or resolve_log_path()).resolve()
    ensure_file_logging(path)
    logger = logging.getLogger()
    if not any(getattr(handler, _BUFFER_HANDLER_MARKER, False) for handler in logger.handlers):
        buffer_handler = StartupBufferHandler()
        setattr(buffer_handler, _BUFFER_HANDLER_MARKER, True)
        logger.addHandler(buffer_handler)
    logging.info("启动引导已开始，日志位置: %s", path)
    return path


def take_startup_messages() -> list[str]:
    logger = logging.getLogger()
    messages = []
    for handler in list(logger.handlers):
        if getattr(handler, _BUFFER_HANDLER_MARKER, False):
            messages.extend(handler.records)
            logger.removeHandler(handler)
            handler.close()
    return messages
