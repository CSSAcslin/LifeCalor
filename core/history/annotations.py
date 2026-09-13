from __future__ import annotations

import copy
import unicodedata
from typing import Any, Iterable, Mapping, Sequence

try:
    import emoji as _emoji
except ImportError:  # Packaging diagnostics surface this when tag editing is used.
    _emoji = None


ANNOTATIONS_VERSION = 1
MAX_DISPLAY_NAME_LENGTH = 128
MAX_TAGS = 3
_UNSET = object()


def empty_annotations() -> dict:
    return {
        "version": ANNOTATIONS_VERSION,
        "display_name": "",
        "tags": [],
        "fields": [],
    }


def normalize_display_name(value: Any) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFC", str(value)).strip()
    if len(text) > MAX_DISPLAY_NAME_LENGTH:
        raise ValueError(f"显示名称不能超过 {MAX_DISPLAY_NAME_LENGTH} 个字符")
    if any(char in "\r\n" or unicodedata.category(char) == "Cc" for char in text):
        raise ValueError("显示名称不能包含换行或控制字符")
    return text


def normalize_tags(values: Iterable[Any] | None) -> list[str]:
    tags: list[str] = []
    for value in values or []:
        tag = unicodedata.normalize("NFC", str(value)).strip()
        if not tag:
            continue
        if any(char in "\r\n" or unicodedata.category(char) == "Cc" for char in tag):
            raise ValueError("标签不能包含换行或控制字符")
        if tag not in tags:
            tags.append(tag)
    if len(tags) > MAX_TAGS:
        raise ValueError(f"每项数据最多只能设置 {MAX_TAGS} 个标签")
    return tags


def validate_emoji_tags(values: Iterable[Any] | None) -> list[str]:
    tags = normalize_tags(values)
    if _emoji is None:
        raise RuntimeError("表情标签功能需要安装 emoji==2.14.1")
    invalid = [tag for tag in tags if not _emoji.is_emoji(tag)]
    if invalid:
        raise ValueError(f"每个标签必须是一个完整表情: {'、'.join(invalid)}")
    return tags


def normalize_field_path(path: Sequence[Any]) -> list[str]:
    if not isinstance(path, (list, tuple)) or not path:
        raise ValueError("子结果路径不能为空")
    return [str(part) for part in path]


def normalize_annotations(value: Mapping[str, Any] | None) -> dict:
    source = copy.deepcopy(dict(value or {}))
    result = source
    result["version"] = int(source.get("version") or ANNOTATIONS_VERSION)
    result["display_name"] = normalize_display_name(source.get("display_name", ""))
    result["tags"] = normalize_tags(source.get("tags", []))

    fields = []
    for raw in source.get("fields", []) or []:
        if not isinstance(raw, Mapping):
            continue
        field_info = copy.deepcopy(dict(raw))
        try:
            field_info["path"] = normalize_field_path(field_info.get("path"))
        except ValueError:
            continue
        field_info["display_name"] = normalize_display_name(field_info.get("display_name", ""))
        if "tags" in field_info:
            field_info["tags"] = normalize_tags(field_info.get("tags"))
        fields.append(field_info)
    result["fields"] = fields
    return result


def copy_annotations(value: Mapping[str, Any] | None) -> dict:
    return normalize_annotations(value)


def annotations_for(value: Any) -> dict:
    if isinstance(value, Mapping):
        return normalize_annotations(value.get("annotations"))
    return normalize_annotations(getattr(value, "annotations", None))


def _field_annotation(annotations: Mapping[str, Any], path: Sequence[Any]):
    normalized_path = normalize_field_path(path)
    for field_info in annotations.get("fields", []) or []:
        if field_info.get("path") == normalized_path:
            return field_info
    return None


def display_name_for(value: Any, path: Sequence[Any] | None = None) -> str:
    annotations = annotations_for(value)
    if path is not None:
        field_info = _field_annotation(annotations, path)
        if field_info and field_info.get("display_name"):
            return field_info["display_name"]
        normalized_path = normalize_field_path(path)
        return normalized_path[-1]
    original = value.get("name", "") if isinstance(value, Mapping) else getattr(value, "name", "")
    return annotations.get("display_name") or str(original or "")


def compact_display_name(value: Any, max_length: int = 10) -> str:
    text = str(value or "")
    max_length = max(4, int(max_length))
    if len(text) <= max_length:
        return text
    available = max_length - 3
    head = (available + 1) // 2
    tail = available // 2
    return f"{text[:head]}...{text[-tail:]}" if tail else f"{text[:head]}..."


def tags_for(value: Any, path: Sequence[Any] | None = None) -> list[str]:
    annotations = annotations_for(value)
    if path is not None:
        field_info = _field_annotation(annotations, path)
        if field_info is not None and "tags" in field_info:
            return list(field_info["tags"])
    return list(annotations.get("tags", []))


def history_identity(value: Any, kind: str | None = None) -> tuple:
    if isinstance(value, Mapping):
        resolved_kind = kind or value.get("kind") or ""
        serial_number = value.get("serial_number")
        timestamp = value.get("timestamp")
    else:
        resolved_kind = kind or value.__class__.__name__
        serial_number = getattr(value, "serial_number", None)
        timestamp = getattr(value, "timestamp", None)
    return str(resolved_kind), serial_number, timestamp


def same_history_identity(left: Any, right: Any, kind: str | None = None) -> bool:
    return history_identity(left, kind) == history_identity(right, kind)


def manifest_item_id(value: Any, kind: str | None = None) -> str:
    resolved_kind, serial_number, timestamp = history_identity(value, kind)
    return f"{resolved_kind}:{serial_number if serial_number is not None else ''}:{timestamp if timestamp is not None else ''}"


def bind_manifest_source(value: Any, cache_dir, item_id: str) -> None:
    object.__setattr__(value, "_history_manifest_dir", str(cache_dir) if cache_dir is not None else None)
    object.__setattr__(value, "_history_manifest_id", item_id or None)


def updated_annotations(
    value: Mapping[str, Any] | None,
    *,
    display_name: Any = _UNSET,
    tags: Iterable[Any] | object = _UNSET,
    path: Sequence[Any] | None = None,
    follow_parent_tags: bool = False,
) -> dict:
    annotations = normalize_annotations(value)
    if path is None:
        if display_name is not _UNSET:
            annotations["display_name"] = normalize_display_name(display_name)
        if tags is not _UNSET:
            annotations["tags"] = normalize_tags(tags)
        return annotations

    normalized_path = normalize_field_path(path)
    field_info = _field_annotation(annotations, normalized_path)
    if field_info is None:
        field_info = {"path": normalized_path, "display_name": ""}
        annotations["fields"].append(field_info)
    if display_name is not _UNSET:
        field_info["display_name"] = normalize_display_name(display_name)
    if follow_parent_tags:
        field_info.pop("tags", None)
    elif tags is not _UNSET:
        field_info["tags"] = normalize_tags(tags)
    return annotations


def assign_annotations(value: Any, annotations: Mapping[str, Any] | None) -> dict:
    normalized = copy_annotations(annotations)
    object.__setattr__(value, "annotations", normalized)
    return normalized
