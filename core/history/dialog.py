from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import Qt, QUrl, pyqtSignal
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

SORT_ROLE = Qt.UserRole + 1


def format_bytes(num_bytes: int) -> str:
    value = float(num_bytes or 0)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} TB"


class SortableTreeWidgetItem(QTreeWidgetItem):
    def __lt__(self, other):
        column = self.treeWidget().sortColumn()
        left = self.data(column, SORT_ROLE)
        right = other.data(column, SORT_ROLE)
        if left is not None and right is not None:
            return left < right
        return super().__lt__(other)


class HistoryCacheManagerDialog(QDialog):
    select_history_requested = pyqtSignal(str, float)
    force_cache_requested = pyqtSignal(str, float)
    delete_history_requested = pyqtSignal(str, float)
    cleanup_orphans_requested = pyqtSignal()
    clear_cache_requested = pyqtSignal()
    recover_manifest_requested = pyqtSignal(str)
    delete_manifest_requested = pyqtSignal(str)
    refresh_requested = pyqtSignal()
    cancel_load_requested = pyqtSignal()

    def __init__(self, params, current_items=None, manifest_items=None, cache_summary=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("历史与缓存管理")
        self.setMinimumSize(900, 600)
        self.params = dict(params or {})
        self.current_items = list(current_items or [])
        self.manifest_items = list(manifest_items or [])
        self.cache_summary = dict(cache_summary or {})
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_current_history_tab(), "当前历史")
        self.tabs.addTab(self._build_recoverable_history_tab(), "可恢复历史")
        self.tabs.addTab(self._build_settings_tab(), "缓存设置")
        layout.addWidget(self.tabs)

        button_layout = QHBoxLayout()
        self.cancel_load_btn = QPushButton("取消读取")
        self.cancel_load_btn.clicked.connect(self.cancel_load_requested.emit)
        button_layout.addWidget(self.cancel_load_btn)
        button_layout.addStretch()
        self.close_btn = QPushButton("关闭")
        self.close_btn.clicked.connect(self.accept)
        button_layout.addWidget(self.close_btn)
        layout.addLayout(button_layout)

    def _build_current_history_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.current_tree = QTreeWidget()
        self.current_tree.setHeaderLabels(["类型", "名称", "形状", "dtype", "缓存", "缓存体积", "内存体积", "时间戳"])
        self.current_tree.setSelectionMode(QAbstractItemView.SingleSelection)
        self.current_tree.setRootIsDecorated(False)
        self.current_tree.setSortingEnabled(True)
        self.current_tree.itemDoubleClicked.connect(self._select_current_item)
        layout.addWidget(self.current_tree)

        buttons = QHBoxLayout()
        self.select_btn = QPushButton("设为当前数据")
        self.force_cache_btn = QPushButton("强制缓存落盘")
        self.delete_history_btn = QPushButton("删除当前历史")
        self.select_btn.clicked.connect(self._select_current_item)
        self.force_cache_btn.clicked.connect(self._force_cache_current_item)
        self.delete_history_btn.clicked.connect(self._delete_current_item)
        buttons.addWidget(self.select_btn)
        buttons.addWidget(self.force_cache_btn)
        buttons.addWidget(self.delete_history_btn)
        buttons.addStretch()
        layout.addLayout(buttons)

        self.refresh_current_items(self.current_items)
        return tab

    def _build_recoverable_history_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.manifest_tree = QTreeWidget()
        self.manifest_tree.setHeaderLabels(["类型", "名称", "形状", "dtype", "缓存文件", "缓存体积", "状态", "保存时间"])
        self.manifest_tree.setRootIsDecorated(False)
        self.manifest_tree.setSortingEnabled(True)
        layout.addWidget(self.manifest_tree)

        buttons = QHBoxLayout()
        self.recover_btn = QPushButton("恢复选中项")
        self.delete_manifest_btn = QPushButton("删除选中索引")
        self.refresh_manifest_btn = QPushButton("刷新")
        self.recover_btn.clicked.connect(self._recover_manifest_item)
        self.delete_manifest_btn.clicked.connect(self._delete_manifest_item)
        self.refresh_manifest_btn.clicked.connect(self.refresh_requested.emit)
        buttons.addWidget(self.recover_btn)
        buttons.addWidget(self.delete_manifest_btn)
        buttons.addWidget(self.refresh_manifest_btn)
        buttons.addStretch()
        layout.addLayout(buttons)

        self.refresh_manifest_items(self.manifest_items)
        return tab

    def _build_settings_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        summary_group = QGroupBox("缓存概览")
        summary_layout = QFormLayout()
        self.current_directory_label = QLabel(str(self.params.get("cache_directory", "")))
        summary_layout.addRow(QLabel("当前目录:"), self.current_directory_label)
        summary_layout.addRow(QLabel("缓存文件:"), QLabel(str(self.cache_summary.get("file_count", 0))))
        summary_layout.addRow(QLabel("缓存体积:"), QLabel(format_bytes(self.cache_summary.get("total_bytes", 0))))
        summary_group.setLayout(summary_layout)
        layout.addWidget(summary_group)

        settings_group = QGroupBox("缓存设置")
        settings_layout = QFormLayout()

        directory_layout = QHBoxLayout()
        self.cache_directory_edit = QLineEdit(str(self.params.get("cache_directory", "")))
        self.cache_directory_edit.setReadOnly(True)
        self.browse_btn = QPushButton("浏览")
        self.open_cache_dir_btn = QPushButton("打开目录")
        self.browse_btn.clicked.connect(self.browse_cache_directory)
        self.open_cache_dir_btn.clicked.connect(self.open_cache_directory)
        directory_layout.addWidget(self.cache_directory_edit)
        directory_layout.addWidget(self.browse_btn)
        directory_layout.addWidget(self.open_cache_dir_btn)

        self.cache_threshold_spin = QSpinBox()
        self.cache_threshold_spin.setRange(1, 1024 * 1024)
        self.cache_threshold_spin.setValue(int(self.params.get("cache_threshold_mb", 512)))
        self.cache_threshold_spin.setSuffix(" MB")

        self.memory_budget_spin = QSpinBox()
        self.memory_budget_spin.setRange(256, 1024 * 1024)
        self.memory_budget_spin.setValue(int(self.params.get("memory_budget_mb", 4096)))
        self.memory_budget_spin.setSuffix(" MB")
        self.memory_budget_spin.setToolTip(
            "限制导入和处理中允许驻留内存的数据总量；超过预算时会在分配前给出提示。"
        )
        self.cache_cleanup_startup_check = QCheckBox()
        self.cache_cleanup_startup_check.setChecked(bool(self.params.get("cache_cleanup_startup", True)))

        settings_layout.addRow(QLabel("缓存目录:"), directory_layout)
        settings_layout.addRow(QLabel("写入阈值:"), self.cache_threshold_spin)
        settings_layout.addRow(QLabel("内存预算:"), self.memory_budget_spin)
        settings_layout.addRow(QLabel("启动清理临时缓存:"), self.cache_cleanup_startup_check)
        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)

        buttons = QHBoxLayout()
        self.cleanup_btn = QPushButton("清除孤立缓存")
        self.clear_cache_btn = QPushButton("清除全部缓存")
        self.cleanup_btn.clicked.connect(self.cleanup_orphans_requested.emit)
        self.clear_cache_btn.clicked.connect(self.clear_cache_requested.emit)
        buttons.addWidget(self.cleanup_btn)
        buttons.addWidget(self.clear_cache_btn)
        buttons.addStretch()
        layout.addLayout(buttons)
        layout.addStretch()
        return tab

    def browse_cache_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "选择缓存目录", self.cache_directory_edit.text(), QFileDialog.ShowDirsOnly)
        if directory:
            self.cache_directory_edit.setText(directory)

    def get_params(self):
        params = dict(self.params)
        params["cache_directory"] = self.cache_directory_edit.text().strip()
        params["cache_threshold_mb"] = self.cache_threshold_spin.value()
        params["memory_budget_mb"] = self.memory_budget_spin.value()
        params["cache_cleanup_startup"] = self.cache_cleanup_startup_check.isChecked()
        return params

    def refresh_current_items(self, items):
        self.current_items = list(items or [])
        self.current_tree.setSortingEnabled(False)
        self.current_tree.clear()
        for item in self.current_items:
            tree_item = SortableTreeWidgetItem([
                item.get("kind", ""),
                item.get("name", ""),
                str(item.get("shape", "")),
                str(item.get("dtype", "")),
                item.get("cache_state", ""),
                format_bytes(item.get("cached_bytes", 0)),
                format_bytes(item.get("memory_bytes", 0)),
                str(item.get("timestamp", "")),
            ])
            tree_item.setData(0, Qt.UserRole, (item.get("kind"), item.get("timestamp")))
            for column, value in {5: item.get("cached_bytes", 0), 6: item.get("memory_bytes", 0), 7: float(item.get("timestamp") or 0)}.items():
                tree_item.setData(column, SORT_ROLE, value)
            self.current_tree.addTopLevelItem(tree_item)
        self.current_tree.setSortingEnabled(True)
        self.current_tree.resizeColumnToContents(0)
        self.current_tree.resizeColumnToContents(1)

    def refresh_manifest_items(self, items):
        self.manifest_items = list(items or [])
        self.manifest_tree.setSortingEnabled(False)
        self.manifest_tree.clear()
        for item in self.manifest_items:
            arrays = item.get("arrays") or {}
            status = item.get("file_status", {})
            cache_bytes = sum(int(array.get("nbytes", 0)) for array in arrays.values())
            saved_at = float(item.get("saved_at") or 0)
            tree_item = SortableTreeWidgetItem([
                item.get("kind", ""),
                item.get("name", ""),
                str(item.get("shape", "")),
                str(item.get("dtype", "")),
                str(len(arrays)),
                format_bytes(cache_bytes),
                "可用" if status.get("ok", True) else "缺失",
                f"{saved_at:.3f}" if saved_at else "",
            ])
            tree_item.setData(0, Qt.UserRole, item.get("id", ""))
            for column, value in {4: len(arrays), 5: cache_bytes, 6: 0 if status.get("ok", True) else 1, 7: saved_at}.items():
                tree_item.setData(column, SORT_ROLE, value)
            self.manifest_tree.addTopLevelItem(tree_item)
        self.manifest_tree.setSortingEnabled(True)
        self.manifest_tree.resizeColumnToContents(0)
        self.manifest_tree.resizeColumnToContents(1)

    def _selected_current_identity(self):
        item = self.current_tree.currentItem()
        if item is None:
            return None
        kind, timestamp = item.data(0, Qt.UserRole)
        try:
            timestamp = float(timestamp)
        except (TypeError, ValueError):
            return None
        return kind, timestamp

    def _selected_manifest_id(self):
        item = self.manifest_tree.currentItem()
        if item is None:
            return None
        item_id = item.data(0, Qt.UserRole)
        return str(item_id) if item_id else None

    def _select_current_item(self):
        identity = self._selected_current_identity()
        if identity:
            self.select_history_requested.emit(identity[0], identity[1])

    def _force_cache_current_item(self):
        identity = self._selected_current_identity()
        if identity:
            self.force_cache_requested.emit(identity[0], identity[1])

    def _delete_current_item(self):
        identity = self._selected_current_identity()
        if identity:
            self.delete_history_requested.emit(identity[0], identity[1])

    def open_cache_directory(self):
        directory = Path(self.cache_directory_edit.text().strip())
        directory.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(directory)))

    def _recover_manifest_item(self):
        item_id = self._selected_manifest_id()
        if item_id:
            self.recover_manifest_requested.emit(item_id)

    def _delete_manifest_item(self):
        item_id = self._selected_manifest_id()
        if item_id:
            self.delete_manifest_requested.emit(item_id)
