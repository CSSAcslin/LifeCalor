from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np


ProgressCallback = Callable[[int, int, str], None]


@dataclass(frozen=True)
class ImportRequest:
    format_id: str
    path: Path
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ImportProbe:
    format_id: str
    shape: tuple[int, ...]
    dtype: str
    axes: str
    color_mode: str = "grayscale"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ImportedPayload:
    data: np.ndarray
    display_data: np.ndarray
    time_point: np.ndarray | None
    format_import: str
    parameters: dict[str, Any]
    name: str


class Importer:
    format_id = ""
    extensions: tuple[str, ...] = ()

    def matches(self, path: Path) -> bool:
        return path.suffix.lower() in self.extensions

    def probe(self, request: ImportRequest) -> ImportProbe:
        raise NotImplementedError

    def read(self, request: ImportRequest, progress: ProgressCallback | None = None, token=None) -> ImportedPayload:
        raise NotImplementedError
