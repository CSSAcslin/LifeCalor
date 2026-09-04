from __future__ import annotations

import traceback
from typing import Any

from .broker import get_error_broker
from .model import AppError


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


def show_app_error(parent: Any, error: AppError) -> AppError:
    if error.original is not None and not error.details:
        error.details = format_exception_details(
            error.original, error.stage, error.context.get("data")
        )
    return get_error_broker().report(parent, error)


def report_warning(parent: Any, title: str, message: str, stage: str | None = None) -> AppError:
    error = AppError(title, str(message), stage=stage, severity="warning")
    show_app_error(parent, error)
    target = parent
    while target is not None:
        update_status = getattr(target, "update_status", None)
        if callable(update_status):
            update_status(str(message), "warning")
            break
        target = target.parent() if callable(getattr(target, "parent", None)) else None
    return error


def report_error(parent: Any, title: str, message: str, stage: str | None = None) -> AppError:
    error = AppError(title, str(message), stage=stage, severity="error")
    return show_app_error(parent, error)


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
    return show_app_error(parent, error)