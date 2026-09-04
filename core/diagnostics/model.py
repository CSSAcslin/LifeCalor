from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AppError:
    title: str
    message: str
    stage: str | None = None
    severity: str = "critical"
    details: str | None = None
    original: BaseException | None = None
    context: dict[str, Any] = field(default_factory=dict)
    task_id: str | None = None
    popup_policy: str = "auto"
    error_id: str = field(default_factory=lambda: uuid.uuid4().hex[:10].upper())
    timestamp: float = field(default_factory=time.time)
    _reported: bool = field(default=False, init=False, repr=False, compare=False)

    @property
    def fingerprint(self) -> tuple[str, str, str, str]:
        exception_type = type(self.original).__name__ if self.original is not None else ""
        return self.title, self.message, self.stage or "", exception_type