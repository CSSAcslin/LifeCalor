from __future__ import annotations

import time
from typing import Any

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication, QDialog, QDialogButtonBox, QHBoxLayout, QLabel, QPushButton,
    QTabWidget, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)

from dataio.classification import DataCategory, describe_source, describe_value


class CanvasDataDetailsDialog(QDialog):
    """Modeless, metadata-only inspector for one image canvas."""

    def __init__(self, canvas, parent=None):
        super().__init__(parent)
        self.canvas = canvas
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.setWindowTitle("画布数据详情")
        self.resize(820, 620)
        layout = QVBoxLayout(self)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)
        footer = QHBoxLayout()
        copy_button = QPushButton("复制摘要")
        copy_button.setToolTip("复制当前画布、数据类型、尺寸和来源摘要")
        copy_button.clicked.connect(self.copy_summary)
        refresh_button = QPushButton("刷新")
        refresh_button.setToolTip("重新读取元数据；不会加载或扫描数组")
        refresh_button.clicked.connect(self.refresh)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.close)
        footer.addWidget(copy_button)
        footer.addWidget(refresh_button)
        footer.addStretch(1)
        footer.addWidget(buttons)
        layout.addLayout(footer)
        self.refresh()

    def _source(self):
        reference = getattr(self.canvas.data, "parent_data", None)
        if callable(reference):
            try:
                return reference()
            except ReferenceError:
                return None
        return None

    def refresh(self):
        self.tabs.clear()
        image = self.canvas.data
        source = self._source()
        descriptor = describe_source(source) if source is not None else describe_value(
            shape=getattr(image, "imageshape", ()), dtype=getattr(image, "datatype", ""),
            axes=(), time_length=getattr(image, "totalframes", None),
        )
        name = getattr(image, "source_name", self.canvas.windowTitle())
        self.summary.setText(
            f"{name}  |  {descriptor.label}  |  shape={descriptor.shape}  |  "
            f"dtype={descriptor.dtype or getattr(image, 'datatype', '')}"
        )
        self.tabs.addTab(self._summary_tree(source, image, descriptor), "数据概要")
        self.tabs.addTab(self._mapping_tree(getattr(source, "parameters", {})), "Parameters")
        self.tabs.addTab(self._mapping_tree(getattr(source, "out_processed", {})), "Other Results")
        self.tabs.addTab(self._canvas_tree(image), "画布状态")

    def _new_tree(self):
        tree = QTreeWidget()
        tree.setColumnCount(3)
        tree.setHeaderLabels(["参数", "类型", "值 / 摘要"])
        tree.header().setStretchLastSection(True)
        tree.setAlternatingRowColors(True)
        return tree

    def _summary_tree(self, source, image, descriptor):
        tree = self._new_tree()
        values = {
            "名称": getattr(image, "source_name", ""),
            "来源对象": type(source).__name__ if source is not None else "已释放",
            "来源格式": getattr(image, "source_format", ""),
            "数据分类": descriptor.label,
            "分类依据": descriptor.reason,
            "Shape": descriptor.shape or getattr(image, "imageshape", ()),
            "dtype": descriptor.dtype or str(getattr(image, "datatype", "")),
            "时间帧数": getattr(image, "totalframes", 1),
            "fps": getattr(image, "fps", None),
            "时间戳": self._time_text(getattr(image, "timestamp_inherited", None)),
        }
        self._append_mapping(tree.invisibleRootItem(), values)
        return tree

    def _mapping_tree(self, mapping):
        tree = self._new_tree()
        if isinstance(mapping, dict) and mapping:
            self._append_mapping(tree.invisibleRootItem(), mapping)
        else:
            QTreeWidgetItem(tree, ["（无）", "", ""])
        return tree

    def _canvas_tree(self, image):
        tree = self._new_tree()
        values = {
            "画布编号": self.canvas.id,
            "当前帧": self.canvas.current_time_idx,
            "是否时序数据": self.canvas.is_temporal,
            "伪彩": self.canvas.colormap if self.canvas.use_colormap else "未启用",
            "显示下限": self.canvas.min_value,
            "显示上限": self.canvas.max_value,
            "同步播放": self.canvas.is_sync_enabled,
            "渲染状态": self.canvas.render_status,
            "ROI 已应用": getattr(image, "ROI_applied", False),
        }
        self._append_mapping(tree.invisibleRootItem(), values)
        return tree

    def _append_mapping(self, parent, mapping, depth=0):
        for key, value in mapping.items():
            descriptor = describe_value(value, semantic_hint=str(key))
            item = QTreeWidgetItem(parent)
            item.setText(0, str(key))
            item.setText(1, descriptor.label)
            item.setText(2, self._format_value(value))
            for column in range(3):
                item.setToolTip(column, item.text(column))
            if isinstance(value, dict) and depth < 8:
                self._append_mapping(item, value, depth + 1)

    @staticmethod
    def _format_value(value: Any) -> str:
        if isinstance(value, dict):
            return f"{len(value)} 项"
        if isinstance(value, np.ndarray) or value.__class__.__name__ == "ArrayRef":
            return f"shape={tuple(value.shape)}, dtype={value.dtype}"
        if isinstance(value, (list, tuple, set, frozenset)):
            if len(value) <= 12:
                return repr(value)
            return f"{type(value).__name__}，{len(value)} 项"
        text = str(value)
        return text if len(text) <= 240 else text[:237] + "..."

    @staticmethod
    def _time_text(value):
        try:
            return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(value)))
        except (TypeError, ValueError, OverflowError, OSError):
            return str(value or "")

    def copy_summary(self):
        QApplication.clipboard().setText(self.summary.text())