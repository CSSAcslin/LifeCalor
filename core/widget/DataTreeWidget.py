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
        self.setColumnCount(7)
        self.setHeaderLabels(["名称 / Key", "来源", "数据类型", "尺寸 & 大小", "数值范围", "创建时间 / 值", action_title])
        header = self.header()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        for column in (1, 2, 3):
            header.setSectionResizeMode(column, QHeaderView.ResizeToContents)
        for column in (4, 5, 6):
            header.setSectionResizeMode(column, QHeaderView.Interactive)
        self.setColumnWidth(0, 300)
        self.setAlternatingRowColors(True)
        self.setAnimated(True)
        self.setIndentation(20)
        self.itemDoubleClicked.connect(self._emit_entry)

    def refresh_data(self, data_history=None, processed_history=None):
        self.clear()
        self.node_map = {}
        data_history = list(Data.get_history_list() if data_history is None else data_history)
        processed_history = list(ProcessedData.get_history_list() if processed_history is None else processed_history)

        for data_obj in data_history:
            item = QTreeWidgetItem(self)
            entry = DataTreeEntry(
                data_obj, data_obj.name, f"原始 ({data_obj.format_import})",
                tuple(data_obj.datashape), str(data_obj.datatype), selectable=True,
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
        entry = DataTreeEntry(proc_obj, proc_obj.name, proc_obj.type_processed, shape, dtype, selectable=bool(shape))
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
                entry = DataTreeEntry(
                    source, str(key), type(value).__name__, shape, dtype,
                    str(key), selectable_arrays, descriptor,
                )
                self._configure_entry(item, entry, None, None, None)
            elif isinstance(value, dict):
                item.setText(0, str(key))
                item.setText(1, "dict")
                item.setText(2, DataCategory.STRUCTURED.value)
                item.setText(5, self._format_value(value))
                self._set_tooltips(item)
                self._add_mapping(item, "内容", value, source, selectable_arrays=False)
            else:
                descriptor = describe_value(value, semantic_hint=str(key))
                item.setText(0, str(key))
                item.setText(1, type(value).__name__)
                item.setText(2, descriptor.label)
                item.setText(3, self._shape_text(descriptor.shape, descriptor.dtype))
                item.setText(5, self._format_value(value))
                self._set_tooltips(item)

    def _configure_entry(self, item, entry, minimum, maximum, timestamp):
        item.setData(0, Qt.UserRole, entry)
        item.setText(0, entry.label)
        item.setText(1, entry.kind)
        descriptor = entry.descriptor or describe_source(entry.source, entry.payload_key)
        item.setText(2, descriptor.label)
        item.setText(3, self._shape_text(entry.shape, entry.dtype))
        item.setText(4, self._range_text(minimum, maximum))
        item.setToolTip(2, descriptor.reason)
        if timestamp is not None:
            try:
                item.setText(5, time.strftime("%y/%m/%d %H:%M:%S", time.localtime(float(timestamp))))
            except (TypeError, ValueError, OverflowError, OSError):
                item.setText(5, str(timestamp))
        self._set_tooltips(item)
        if entry.selectable and self.action_factory is not None:
            widget = self.action_factory(self, item, entry)
            if widget is not None:
                self.setItemWidget(item, 6, widget)

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
            item.setToolTip(column, item.text(column))

    def selected_entry(self):
        item = self.currentItem()
        return item.data(0, Qt.UserRole) if item is not None else None

    def filter_text(self, text):
        query = (text or "").strip().lower()

        def visit(item):
            own = any(query in item.text(column).lower() for column in range(min(6, item.columnCount())))
            child_match = False
            for index in range(item.childCount()):
                child_match = visit(item.child(index)) or child_match
            visible = not query or own or child_match
            item.setHidden(not visible)
            if query and child_match:
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
