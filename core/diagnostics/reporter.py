from __future__ import annotations

import logging
import traceback
import time
from dataclasses import dataclass, field
from typing import Any

from PyQt5.QtWidgets import QMessageBox


@dataclass
class AppError:
    title: str
    message: str
    stage: str | None = None
    severity: str = "critical"
    details: str | None = None
    original: BaseException | None = None
    context: dict[str, Any] = field(default_factory=dict)


def _data_context(data: Any) -> dict[str, Any]:
    if data is None:
        return {}
    context = {}
    for attr in ("name", "source_name", "type_processed", "source_format"):
        value = getattr(data, attr, None)
        if value is not None:
            context[attr] = value
    for attr in ("shape", "datashape", "imageshape"):
        value = getattr(data, attr, None)
        if value is not None:
            context[attr] = tuple(value) if isinstance(value, (list, tuple)) else value
    dtype = getattr(data, "dtype", None) or getattr(data, "datatype", None)
    if dtype is not None:
        context["dtype"] = str(dtype)
    return context


def format_exception_details(exc: BaseException, stage: str | None = None, data: Any = None) -> str:
    lines = []
    if stage:
        lines.append(f"阶段: {stage}")
    lines.append(f"异常: {type(exc).__name__}: {exc}")
    context = _data_context(data)
    for key, value in context.items():
        lines.append(f"{key}: {value}")
    lines.append("Traceback:")
    lines.append("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).rstrip())
    return "\n".join(lines)


_RECENT_POPUPS = {}


def _should_popup(error: AppError) -> bool:
    if error.severity not in {"error", "critical"}:
        return False
    fingerprint = (error.title, error.message, error.stage)
    now = time.monotonic()
    previous = _RECENT_POPUPS.get(fingerprint, 0.0)
    _RECENT_POPUPS[fingerprint] = now
    return now - previous > 1.0

def show_app_error(parent: Any, error: AppError) -> None:
    details = error.details or ""
    if error.original is not None and not details:
        details = format_exception_details(error.original, error.stage, error.context.get("data"))
    log_message = f"{error.title}: {error.message}"
    log_method = {
        "information": logging.info,
        "info": logging.info,
        "warning": logging.warning,
        "error": logging.error,
        "critical": logging.critical,
    }.get(error.severity, logging.error)
    if details:
        log_method("%s\n%s", log_message, details, extra={"lifecalor_user_reported": True})
    else:
        log_method(log_message, extra={"lifecalor_user_reported": True})
    if _should_popup(error):
        QMessageBox.critical(parent, error.title, error.message)



def report_warning(parent: Any, title: str, message: str, stage: str | None = None) -> AppError:
    error = AppError(title, str(message), stage=stage, severity="warning")
    show_app_error(parent, error)
    target = parent
    while target is not None:
        update_status = getattr(target, "update_status", None)
        if callable(update_status):
            update_status(str(message), "warning")
            break
        target = getattr(target, "parent", lambda: None)() if callable(getattr(target, "parent", None)) else None
    return error


def report_error(parent: Any, title: str, message: str, stage: str | None = None) -> AppError:
    error = AppError(title, str(message), stage=stage, severity="error")
    show_app_error(parent, error)
    return error


def report_exception(
    parent: Any,
    title: str,
    message: str,
    exc: BaseException,
    stage: str | None = None,
    data: Any = None,
    severity: str = "critical",
) -> AppError:
    error = AppError(
        title=title,
        message=message,
        stage=stage,
        severity=severity,
        details=format_exception_details(exc, stage, data),
        original=exc,
        context={"data": data} if data is not None else {},
    )
    show_app_error(parent, error)
    return error
