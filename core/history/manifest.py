from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from ArrayCache import ArrayRef
from DataManager import Data, ProcessedData, collect_array_refs

MANIFEST_FILENAME = "history_manifest.json"
SCHEMA_VERSION = 1


def _json_safe(value: Any):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return {
            "kind": "ndarray_summary",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "nbytes": int(value.nbytes),
        }
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def array_ref_to_dict(ref: ArrayRef) -> dict:
    return {
        "path": str(ref.path),
        "shape": list(ref.shape),
        "dtype": ref.dtype,
        "nbytes": int(ref.nbytes),
        "created_at": float(ref.created_at),
        "field_name": ref.field_name,
    }


def array_refs_by_field(item) -> dict[str, ArrayRef]:
    refs: dict[str, ArrayRef] = {}
    values = getattr(item, "__dict__", {})
    for field_name in ("_data_origin_storage", "_image_import_storage", "_data_processed_storage"):
        ref = values.get(field_name)
        if isinstance(ref, ArrayRef):
            refs[ref.field_name or field_name] = ref
    for key, value in (getattr(item, "out_processed", None) or {}).items():
        if isinstance(value, ArrayRef):
            refs[f"out_processed.{key}"] = value
    return refs


def cache_status_for_history_item(item) -> dict:
    cached_refs = []
    seen_paths = set()
    for ref in collect_array_refs(item):
        key = str(Path(ref.path).resolve())
        if key not in seen_paths:
            cached_refs.append(ref)
            seen_paths.add(key)
    memory_bytes = 0
    values = getattr(item, "__dict__", {})
    for value in values.values():
        if isinstance(value, np.ndarray):
            memory_bytes += int(value.nbytes)
    for value in (getattr(item, "out_processed", None) or {}).values():
        if isinstance(value, np.ndarray):
            memory_bytes += int(value.nbytes)
    return {
        "cached_count": len(cached_refs),
        "cached_bytes": sum(int(ref.nbytes) for ref in cached_refs),
        "memory_bytes": memory_bytes,
    }


def build_manifest_item(item, kind: str | None = None) -> dict:
    kind = kind or item.__class__.__name__
    arrays = {field: array_ref_to_dict(ref) for field, ref in array_refs_by_field(item).items()}
    metadata = {
        "parameters": _json_safe(getattr(item, "parameters", {})),
        "out_processed_metadata": _json_safe({
            key: value for key, value in (getattr(item, "out_processed", None) or {}).items()
            if not isinstance(value, ArrayRef) and not isinstance(value, np.ndarray)
        }),
        "time_point": _json_safe(getattr(item, "time_point", None)),
    }
    return {
        "id": f"{kind}:{getattr(item, 'serial_number', '')}:{getattr(item, 'timestamp', '')}",
        "kind": kind,
        "name": getattr(item, "name", ""),
        "type_processed": getattr(item, "type_processed", ""),
        "format_import": getattr(item, "format_import", ""),
        "timestamp": getattr(item, "timestamp", None),
        "timestamp_inherited": getattr(item, "timestamp_inherited", None),
        "serial_number": getattr(item, "serial_number", None),
        "shape": list(getattr(item, "datashape", ()) or ()),
        "dtype": str(getattr(item, "datatype", "")),
        "ndim": getattr(item, "ndim", None),
        "datamin": _json_safe(getattr(item, "datamin", None)),
        "datamax": _json_safe(getattr(item, "datamax", None)),
        "metadata": metadata,
        "arrays": arrays,
        "cache_status": cache_status_for_history_item(item),
        "saved_at": time.time(),
    }


class HistoryManifestStore:
    def __init__(self, cache_dir: Path | str):
        self.cache_dir = Path(cache_dir)
        self.path = self.cache_dir / MANIFEST_FILENAME

    def empty_manifest(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "saved_at": time.time(),
            "items": [],
        }

    def load(self) -> dict:
        if not self.path.exists():
            return self.empty_manifest()
        with self.path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if data.get("schema_version") != SCHEMA_VERSION:
            data["schema_version"] = SCHEMA_VERSION
        data.setdefault("items", [])
        return data

    def save(self, manifest: dict) -> dict:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        manifest = dict(manifest)
        manifest["schema_version"] = SCHEMA_VERSION
        manifest["saved_at"] = time.time()
        manifest.setdefault("items", [])
        with self.path.open("w", encoding="utf-8") as fh:
            json.dump(manifest, fh, ensure_ascii=False, indent=2)
        return manifest

    def upsert(self, item: dict) -> dict:
        manifest = self.load()
        items = [existing for existing in manifest.get("items", []) if existing.get("id") != item.get("id")]
        items.append(item)
        manifest["items"] = items
        self.save(manifest)
        return item

    def validate_item_files(self, item: dict) -> dict:
        missing = []
        for array_info in (item.get("arrays") or {}).values():
            path = Path(array_info.get("path", ""))
            if not path.exists():
                missing.append(str(path))
        return {"ok": not missing, "missing": missing}

    def valid_items(self) -> list[dict]:
        return [item for item in self.load().get("items", []) if self.validate_item_files(item)["ok"]]

    def remove_missing_items(self) -> int:
        manifest = self.load()
        old_items = manifest.get("items", [])
        new_items = [item for item in old_items if self.validate_item_files(item)["ok"]]
        manifest["items"] = new_items
        self.save(manifest)
        return len(old_items) - len(new_items)
