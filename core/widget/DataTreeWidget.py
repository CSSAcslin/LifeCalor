from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QHeaderView, QTreeWidget, QTreeWidgetItem, QWidget

from ArrayCache import ArrayRef
from DataManager import Data, ProcessedData
from dataio.classification import DataCategory, DataDescriptor, describe_source, describe_value
from history.annotations import display_name_for, history_identity, tags_for
from widget.DataFilterControls import metadata_matches


FILTER_ROLE = Qt.UserRole + 3


@dataclass(frozen=True)
class DataTreeEntry:
    source: object
    label: str
    kind: str
    shape: tuple[int, ...] = ()
    dtype: str = ""
    payload_key: Optional[str] = None
    selectable: bool = False
    descriptor: Optional[DataDescriptor] = None
    original_label: str = ""
    field_path: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()

    def resolve(self):
        if self.payload_key is not None:
            if hasattr(self.source, "out_processed_array"):
                return self.source.out_processed_array(self.payload_key)
            return self.source.out_processed[self.payload_key]
        if isinstance(self.source, ProcessedData):
            return self.source.data_processed
        if isinstance(self.source, Data):
            return self.source.data_origin
        return np.asarray(self.source)


ActionFactory = Callable[["DataHistoryTreeWidget", QTreeWidgetItem, DataTreeEntry], Optional[QWidget]]


class DataHistoryTreeWidget(QTreeWidget):
    """Reusable Data/ProcessedData history tree with an injectable action column."""

    entry_double_clicked = pyqtSignal(object)

    def __init__(self, parent=None, action_factory: ActionFactory | None = None, action_title="操作"):
        super().__init__(parent)
        self.action_factory = action_factory
        self.node_map = {}
        self.action_column = 7
        self.setColumnCount(8)
        self.setHeaderLabels(["名称 / Key", "标签", "来源", "数据类型", "尺寸 & 大小", "数值范围", "创建时间 / 值", action_title])
        header = self.header()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        for column in (1, 2, 3, 4):
            header.setSectionResizeMode(column, QHeaderView.ResizeToContents)
        for column in (5, 6, self.action_column):
            header.setSectionResizeMode(column, QHeaderView.Interactive)
        self.setColumnWidth(0, 300)
        self.setAlternatingRowColors(True)
        self.setAnimated(True)
        self.setIndentation(20)
        self.itemDoubleClicked.connect(self._emit_entry)

    def refresh_data(self, data_history=None, processed_history=None):
        state = self._capture_view_state()
        self.clear()
        self.node_map = {}
        data_history = list(Data.get_history_list() if data_history is None else data_history)
        processed_history = list(ProcessedData.get_history_list() if processed_history is None else processed_history)

        for data_obj in data_history:
            item = QTreeWidgetItem(self)
            entry = DataTreeEntry(
                data_obj, display_name_for(data_obj), f"原始 ({data_obj.format_import})",
                tuple(data_obj.datashape), str(data_obj.datatype), selectable=True,
                original_label=data_obj.name,
                tags=tuple(tags_for(data_obj)),
            )
            self._configure_entry(item, entry, data_obj.datamin, data_obj.datamax, data_obj.timestamp)
            if data_obj.parameters:
                self._add_mapping(item, "Parameters", data_obj.parameters, data_obj, selectable_arrays=False)
            self.node_map[data_obj.timestamp] = item

        orphans = []
        for proc_obj in sorted(processed_history, key=lambda value: getattr(value, "timestamp", 0)):
            parent = self._find_parent(proc_obj.timestamp_inherited)
            if parent is None:
                orphans.append(proc_obj)
                continue
            item = self._add_processed(parent, proc_obj)
            self.node_map[proc_obj.timestamp] = item

        if orphans:
            root = QTreeWidgetItem(self)
            root.setText(0, "历史处理记录（无关联源数据）")
            for proc_obj in orphans:
                self._add_processed(root, proc_obj)
            root.setExpanded(True)
        self.expandToDepth(0)
        self._restore_view_state(state)

    def _item_state_key(self, item):
        entry = item.data(0, Qt.UserRole)
        if isinstance(entry, DataTreeEntry):
            return (
                "entry",
                history_identity(entry.source),
                tuple(entry.field_path),
                entry.payload_key,
            )
        parent = item.parent()
        parent_key = self._item_state_key(parent) if parent is not None else ("root",)
        return ("group", parent_key, item.text(0))

    def _capture_view_state(self):
        if self.topLevelItemCount() == 0:
            return None
        expanded = set()
        selected = set()

        def visit(item):
            key = self._item_state_key(item)
            if item.isExpanded():
                expanded.add(key)
            if item.isSelected():
                selected.add(key)
            for index in range(item.childCount()):
                visit(item.child(index))

        for index in range(self.topLevelItemCount()):
            visit(self.topLevelItem(index))
        return {
            "expanded": expanded,
            "selected": selected,
            "scroll": self.verticalScrollBar().value(),
        }

    def _restore_view_state(self, state):
        if not state:
            return

        def visit(item):
            key = self._item_state_key(item)
            item.setExpanded(key in state["expanded"])
            item.setSelected(key in state["selected"])
            for index in range(item.childCount()):
                visit(item.child(index))

        for index in range(self.topLevelItemCount()):
            visit(self.topLevelItem(index))
        self.verticalScrollBar().setValue(state["scroll"])

    def _find_parent(self, timestamp):
        if timestamp is None:
            return None
        for candidate, item in self.node_map.items():
            try:
                if abs(float(candidate) - float(timestamp)) < 1e-6:
                    return item
            except (TypeError, ValueError, OverflowError):
                continue
        return None

    def _add_processed(self, parent, proc_obj):
        item = QTreeWidgetItem(parent)
        shape = tuple(getattr(proc_obj, "datashape", ()) or ())
        dtype = str(getattr(proc_obj, "datatype", "")) if shape else ""
        entry = DataTreeEntry(
            proc_obj, display_name_for(proc_obj), proc_obj.type_processed, shape, dtype,
            selectable=bool(shape), original_label=proc_obj.name,
            tags=tuple(tags_for(proc_obj)),
        )
        self._configure_entry(
            item, entry,
            getattr(proc_obj, "datamin", None), getattr(proc_obj, "datamax", None), proc_obj.timestamp,
        )
        if proc_obj.out_processed:
            self._add_mapping(item, "Other Results", proc_obj.out_processed, proc_obj, selectable_arrays=True)
        return item

    def _add_mapping(self, parent, title, mapping, source, selectable_arrays):
        root = QTreeWidgetItem(parent)
        root.setText(0, title)
        root.setText(2, DataCategory.STRUCTURED.value)
        root.setToolTip(0, title)
        for key, value in mapping.items():
            item = QTreeWidgetItem(root)
            if isinstance(value, (ArrayRef, np.ndarray)):
                shape, dtype = tuple(value.shape), str(value.dtype)
                descriptor = describe_source(source, str(key))
                field_path = ("out_processed", str(key))
                entry = DataTreeEntry(
                    source, display_name_for(source, field_path), type(value).__name__, shape, dtype,
                    str(key), selectable_arrays, descriptor, str(key), field_path,
                    tuple(tags_for(source, field_path)),
                )
                self._configure_entry(item, entry, None, None, None)
            elif isinstance(value, dict):
                item.setText(0, str(key))
                item.setText(2, "dict")
                item.setText(3, DataCategory.STRUCTURED.value)
                item.setText(6, self._format_value(value))
                self._set_tooltips(item)
                self._add_mapping(item, "内容", value, source, selectable_arrays=False)
            else:
                descriptor = describe_value(value, semantic_hint=str(key))
                item.setText(0, str(key))
                item.setText(2, type(value).__name__)
                item.setText(3, descriptor.label)
                item.setText(4, self._shape_text(descriptor.shape, descriptor.dtype))
                item.setText(6, self._format_value(value))
                self._set_tooltips(item)

    def _configure_entry(self, item, entry, minimum, maximum, timestamp):
        item.setData(0, Qt.UserRole, entry)
        item.setData(0, Qt.UserRole + 2, history_identity(entry.source))
        item.setText(0, entry.label)
        item.setText(1, " ".join(entry.tags))
        item.setText(2, entry.kind)
        descriptor = entry.descriptor or describe_source(entry.source, entry.payload_key)
        item.setText(3, descriptor.label)
        item.setText(4, self._shape_text(entry.shape, entry.dtype))
        item.setText(5, self._range_text(minimum, maximum))
        item.setToolTip(0, f"显示名称：{entry.label}\n原始名称：{entry.original_label or entry.label}")
        item.setToolTip(1, " ".join(entry.tags) if entry.tags else "无标签")
        item.setToolTip(3, descriptor.reason)
        item.setData(0, FILTER_ROLE, {
            "display_name": entry.label,
            "original_name": entry.original_label or entry.label,
            "source_name": getattr(entry.source, "name", ""),
            "payload_key": entry.payload_key or "",
            "category": descriptor.label,
            "tags": tuple(entry.tags),
        })
        if timestamp is not None:
            try:
                item.setText(6, time.strftime("%y/%m/%d %H:%M:%S", time.localtime(float(timestamp))))
            except (TypeError, ValueError, OverflowError, OSError):
                item.setText(6, str(timestamp))
        self._set_tooltips(item)
        if entry.selectable and self.action_factory is not None:
            widget = self.action_factory(self, item, entry)
            if widget is not None:
                self.setItemWidget(item, self.action_column, widget)

    @staticmethod
    def _shape_text(shape, dtype):
        if not shape:
            return "None"
        count = math.prod(int(value) for value in shape)
        try:
            size = count * np.dtype(dtype).itemsize
        except TypeError:
            size = 0
        units = ["B", "KB", "MB", "GB", "TB"]
        amount = float(size)
        unit = units[-1]
        for candidate in units:
            unit = candidate
            if amount < 1024 or candidate == units[-1]:
                break
            amount /= 1024
        return f"{'x'.join(map(str, shape))}\n{amount:.1f} {unit}"

    @staticmethod
    def _range_text(minimum, maximum):
        if minimum is None or maximum is None:
            return ""
        try:
            return f"{minimum:.4g} ~ {maximum:.4g}"
        except (TypeError, ValueError):
            return "不可直接比较"

    @staticmethod
    def _format_value(value):
        if isinstance(value, dict):
            return f"{len(value)} 项"
        if isinstance(value, (list, tuple, set, frozenset)):
            return f"{type(value).__name__}，{len(value)} 项"
        if isinstance(value, (np.ndarray, ArrayRef)):
            return f"shape={tuple(value.shape)}, dtype={value.dtype}"
        text = str(value)
        return text if len(text) <= 120 else text[:117] + "..."

    @staticmethod
    def _set_tooltips(item):
        for column in range(item.columnCount()):
            if not item.toolTip(column):
                item.setToolTip(column, item.text(column))

    def selected_entry(self):
        item = self.currentItem()
        return item.data(0, Qt.UserRole) if item is not None else None

    def available_filter_values(self):
        categories = set()
        tags = []

        def visit(item):
            metadata = item.data(0, FILTER_ROLE)
            if metadata:
                if metadata.get("category"):
                    categories.add(metadata["category"])
                for tag in metadata.get("tags", ()):
                    if tag not in tags:
                        tags.append(tag)
            for index in range(item.childCount()):
                visit(item.child(index))

        root = self.invisibleRootItem()
        for index in range(root.childCount()):
            visit(root.child(index))
        return sorted(categories), tags

    def filter_text(self, text):
        self.filter_entries(query=text)

    def filter_entries(self, query="", category="", tags=(), untagged=False):
        query = (query or "").strip()

        def visit(item, ancestor_match=False):
            metadata = item.data(0, FILTER_ROLE)
            own = bool(metadata) and metadata_matches(metadata, query, category, tags, untagged)
            child_match = False
            for index in range(item.childCount()):
                child_match = visit(item.child(index), ancestor_match or own) or child_match
            no_filter = not query and not category and not tags and not untagged
            visible = no_filter or ancestor_match or own or child_match
            item.setHidden(not visible)
            if not no_filter and child_match:
                item.setExpanded(True)
            return visible

        root = self.invisibleRootItem()
        for index in range(root.childCount()):
            visit(root.child(index))

    def expand_except_metadata(self):
        def visit(item):
            if item.text(0) in {"Parameters", "Other Results"}:
                item.setExpanded(False)
                return
            item.setExpanded(True)
            for index in range(item.childCount()):
                visit(item.child(index))

        root = self.invisibleRootItem()
        for index in range(root.childCount()):
            visit(root.child(index))

    def _emit_entry(self, item, _column):
        entry = item.data(0, Qt.UserRole)
        if entry is not None and entry.selectable:
            self.entry_double_clicked.emit(entry)
