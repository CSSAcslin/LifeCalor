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


def array_ref_from_dict(info: dict) -> ArrayRef:
    return ArrayRef(
        path=Path(info["path"]),
        shape=tuple(info.get("shape", ())),
        dtype=str(info.get("dtype", "")),
        nbytes=int(info.get("nbytes", 0)),
        created_at=float(info.get("created_at", time.time())),
        field_name=str(info.get("field_name", "")),
    )


def _restore_common_fields(instance, item: dict):
    object.__setattr__(instance, "name", item.get("name", ""))
    object.__setattr__(instance, "timestamp", item.get("timestamp"))
    object.__setattr__(instance, "serial_number", item.get("serial_number"))
    object.__setattr__(instance, "datashape", tuple(item.get("shape", ())))
    dtype = item.get("dtype") or "float64"
    try:
        dtype_value = np.dtype(dtype)
    except TypeError:
        dtype_value = dtype
    object.__setattr__(instance, "datatype", dtype_value)
    object.__setattr__(instance, "ndim", item.get("ndim"))
    object.__setattr__(instance, "datamin", item.get("datamin"))
    object.__setattr__(instance, "datamax", item.get("datamax"))
    shape = tuple(item.get("shape", ()))
    if len(shape) == 3:
        object.__setattr__(instance, "timelength", shape[0])
        object.__setattr__(instance, "framesize", (shape[1], shape[2]))
    elif len(shape) == 2:
        object.__setattr__(instance, "timelength", 1)
        object.__setattr__(instance, "framesize", (shape[0], shape[1]))
    elif len(shape) == 1:
        object.__setattr__(instance, "timelength", shape[0])
        object.__setattr__(instance, "framesize", shape[0])


def restore_history_item(item: dict):
    kind = item.get("kind")
    arrays = item.get("arrays") or {}
    metadata = item.get("metadata") or {}

    if kind == "Data":
        instance = Data.__new__(Data)
        _restore_common_fields(instance, item)
        object.__setattr__(instance, "format_import", item.get("format_import", ""))
        object.__setattr__(instance, "parameters", metadata.get("parameters") or {})
        object.__setattr__(instance, "time_point", None)
        object.__setattr__(instance, "out_processed", {})
        object.__setattr__(instance, "ROI_applied", False)
        if "data_origin" in arrays:
            ref = array_ref_from_dict(arrays["data_origin"])
            object.__setattr__(instance, "_data_origin_storage", ref)
            object.__setattr__(instance, "data_origin", ref)
        if "image_import" in arrays:
            ref = array_ref_from_dict(arrays["image_import"])
            object.__setattr__(instance, "_image_import_storage", ref)
            object.__setattr__(instance, "image_import", ref)
        else:
            object.__setattr__(instance, "_image_import_storage", None)
            object.__setattr__(instance, "image_import", None)
        return instance

    if kind == "ProcessedData":
        instance = ProcessedData.__new__(ProcessedData)
        _restore_common_fields(instance, item)
        object.__setattr__(instance, "timestamp_inherited", item.get("timestamp_inherited"))
        object.__setattr__(instance, "type_processed", item.get("type_processed", ""))
        object.__setattr__(instance, "time_point", None)
        object.__setattr__(instance, "ROI_applied", False)
        object.__setattr__(instance, "ROI_mask", None)
        out_processed = dict(metadata.get("out_processed_metadata") or {})
        for field_name, array_info in arrays.items():
            if field_name == "data_processed":
                ref = array_ref_from_dict(array_info)
                object.__setattr__(instance, "_data_processed_storage", ref)
                object.__setattr__(instance, "data_processed", ref)
            elif field_name.startswith("out_processed."):
                key = field_name.split(".", 1)[1]
                out_processed[key] = array_ref_from_dict(array_info)
        object.__setattr__(instance, "out_processed", out_processed)
        return instance

    raise ValueError(f"不支持恢复的历史类型: {kind}")



def array_refs_for_item(item: dict) -> list[ArrayRef]:
    refs: list[ArrayRef] = []
    for array_info in (item.get("arrays") or {}).values():
        try:
            refs.append(array_ref_from_dict(array_info))
        except (KeyError, TypeError, ValueError):
            continue
    return refs


def array_refs_for_manifest(manifest: dict) -> list[ArrayRef]:
    refs: list[ArrayRef] = []
    for item in manifest.get("items", []) or []:
        refs.extend(array_refs_for_item(item))
    return refs


def _ref_path_key(ref: ArrayRef) -> str:
    return str(Path(ref.path).resolve())


def manifest_path_keys(manifest: dict, exclude_item_id: str | None = None) -> set[str]:
    keys: set[str] = set()
    for item in manifest.get("items", []) or []:
        if exclude_item_id is not None and item.get("id") == exclude_item_id:
            continue
        for ref in array_refs_for_item(item):
            keys.add(_ref_path_key(ref))
    return keys

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

    def clear_items(self) -> int:
        manifest = self.load()
        old_items = manifest.get("items", [])
        manifest["items"] = []
        self.save(manifest)
        return len(old_items)

    def delete_item(self, item_id: str, delete_cache_files: bool = False) -> dict:
        manifest = self.load()
        old_items = list(manifest.get("items", []))
        target = next((item for item in old_items if item.get("id") == item_id), None)
        if target is None:
            return {"removed": False, "deleted_files": 0}

        remaining_items = [item for item in old_items if item.get("id") != item_id]
        deleted_files = 0
        if delete_cache_files:
            remaining_paths = manifest_path_keys({"items": remaining_items})
            seen_paths: set[str] = set()
            for ref in array_refs_for_item(target):
                key = _ref_path_key(ref)
                if key in remaining_paths or key in seen_paths:
                    continue
                seen_paths.add(key)
                path = Path(ref.path)
                if path.exists() and path.is_file():
                    path.unlink()
                    deleted_files += 1

        manifest["items"] = remaining_items
        self.save(manifest)
        return {"removed": True, "deleted_files": deleted_files}
