from __future__ import annotations

import math

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog, QDialogButtonBox, QLabel, QTreeWidget, QTreeWidgetItem, QVBoxLayout,
)

from .readers import list_hdf5_datasets


class Hdf5DatasetDialog(QDialog):
    def __init__(self, path, parent=None):
        super().__init__(parent)
        self.path = path
        self.dataset_path = None
        self.setWindowTitle("选择 HDF5 数据集")
        self.resize(720, 480)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("选择一个数值数据集；读取前仅查看 Shape、dtype 和预计大小。"))
        self.tree = QTreeWidget()
        self.tree.setColumnCount(4)
        self.tree.setHeaderLabels(["数据集", "Shape", "dtype", "预计大小"])
        self.tree.setAlternatingRowColors(True)
        layout.addWidget(self.tree, 1)
        for dataset, shape, dtype in list_hdf5_datasets(path):
            try:
                size = math.prod(shape) * np.dtype(dtype).itemsize
                size_text = self._size_text(size)
                numeric = np.dtype(dtype).kind not in {"O", "S", "U", "V"}
            except TypeError:
                size_text, numeric = "未知", False
            item = QTreeWidgetItem([dataset, str(shape), dtype, size_text])
            item.setData(0, Qt.UserRole, dataset if numeric else None)
            if not numeric:
                item.setDisabled(True)
                item.setToolTip(0, "仅支持数值数据集")
            self.tree.addTopLevelItem(item)
        self.tree.itemDoubleClicked.connect(lambda *_: self.accept())
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @staticmethod
    def _size_text(size):
        amount = float(size)
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if amount < 1024 or unit == "TB":
                return f"{amount:.2f} {unit}"
            amount /= 1024

    def accept(self):
        item = self.tree.currentItem()
        dataset = item.data(0, Qt.UserRole) if item is not None else None
        if dataset:
            self.dataset_path = dataset
            super().accept()


def choose_hdf5_dataset(path, parent=None):
    dialog = Hdf5DatasetDialog(path, parent)
    return dialog.dataset_path if dialog.exec_() == QDialog.Accepted else None