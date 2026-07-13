from __future__ import annotations

from pathlib import Path

from .model import ImportRequest
from .readers import NpyImporter, TiffImporter


class ImporterRegistry:
    def __init__(self, importers=()):
        self._importers = {}
        for importer in importers:
            self.register(importer)

    def register(self, importer):
        self._importers[importer.format_id] = importer
        return importer

    def resolve(self, format_id: str, path) -> object:
        path = Path(path)
        if format_id and format_id != "auto":
            try:
                importer = self._importers[format_id]
            except KeyError as exc:
                raise ValueError(f"不支持的导入格式: {format_id}") from exc
            if not importer.matches(path):
                raise ValueError(f"文件扩展名 {path.suffix or '(无)'} 与 {format_id} 格式不匹配")
            return importer
        matches = [importer for importer in self._importers.values() if importer.matches(path)]
        if len(matches) != 1:
            raise ValueError(f"无法根据扩展名识别导入格式: {path.suffix or '(无)'}")
        return matches[0]

    def probe(self, format_id, path, options=None):
        request = ImportRequest(format_id, Path(path), dict(options or {}))
        return self.resolve(format_id, path).probe(request)

    def read(self, format_id, path, options=None, progress=None, token=None):
        request = ImportRequest(format_id, Path(path), dict(options or {}))
        return self.resolve(format_id, path).read(request, progress=progress, token=token)


def default_importer_registry():
    return ImporterRegistry((NpyImporter(), TiffImporter()))
