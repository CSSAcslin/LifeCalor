from __future__ import annotations

import logging
import traceback
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


def _message_box_method(severity: str):
    if severity == "warning":
        return QMessageBox.warning
    if severity == "information":
        return QMessageBox.information
    return QMessageBox.critical


def show_app_error(parent: Any, error: AppError) -> None:
    details = error.details or ""
    if error.original is not None and not details:
        details = format_exception_details(error.original, error.stage, error.context.get("data"))
    log_message = f"{error.title}: {error.message}"
    if details:
        logging.error("%s\n%s", log_message, details)
    else:
        logging.error(log_message)
    _message_box_method(error.severity)(parent, error.title, error.message)


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
