from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PyQt5.QtCore import Qt, QUrl, pyqtSignal
from PyQt5.QtGui import QDesktopServices, QKeySequence
from PyQt5.QtWidgets import (
    QAction,
    QAbstractItemView,
    QCheckBox,
    QComboBox,
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

from dataio.classification import describe_value
from history.annotations import display_name_for, tags_for
from widget.DataFilterControls import TagFilterButton, metadata_matches, style_filter_controls

SORT_ROLE = Qt.UserRole + 1
FILTER_ROLE = Qt.UserRole + 3


def format_timestamp(value) -> str:
    """Format a Unix timestamp for display while callers retain the numeric identity."""
    if value in (None, ""):
        return ""
    try:
        timestamp = float(value)
        return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
    except (OverflowError, OSError, TypeError, ValueError):
        return str(value)


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
    select_history_requested = pyqtSignal(object)
    force_cache_requested = pyqtSignal(object)
    delete_history_requested = pyqtSignal(object)
    cleanup_orphans_requested = pyqtSignal()
    clear_cache_requested = pyqtSignal()
    recover_manifest_requested = pyqtSignal(str)
    delete_manifest_requested = pyqtSignal(str)
    refresh_requested = pyqtSignal()
    cancel_load_requested = pyqtSignal()
    edit_current_requested = pyqtSignal(object)
    edit_manifest_requested = pyqtSignal(str)
    batch_tag_current_requested = pyqtSignal(object)
    batch_tag_manifest_requested = pyqtSignal(object)
    batch_force_cache_requested = pyqtSignal(object)
    batch_delete_history_requested = pyqtSignal(object)
    batch_recover_manifest_requested = pyqtSignal(object)
    batch_delete_manifest_requested = pyqtSignal(object)

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
        filters = QHBoxLayout()
        filters.addWidget(QLabel("搜索"))
        self.current_search = QLineEdit()
        self.current_search.setPlaceholderText("显示名称、原始名称或来源")
        self.current_search.setClearButtonEnabled(True)
        self.current_search.textChanged.connect(self._apply_current_filters)
        filters.addWidget(self.current_search, 1)
        self.current_type_filter = QComboBox()
        self.current_type_filter.currentIndexChanged.connect(self._apply_current_filters)
        filters.addWidget(self.current_type_filter)
        self.current_tag_filter = TagFilterButton()
        self.current_tag_filter.filterChanged.connect(self._apply_current_filters)
        filters.addWidget(self.current_tag_filter)
        clear_filters = QPushButton("清除筛选")
        clear_filters.setToolTip("清除名称、类型和标签筛选，不修改数据")
        clear_filters.clicked.connect(self._clear_current_filters)
        filters.addWidget(clear_filters)
        style_filter_controls(
            self.current_search, self.current_type_filter, self.current_tag_filter, clear_filters
        )
        layout.addLayout(filters)
        self.current_tree = QTreeWidget()
        self.current_tree.setHeaderLabels(["", "类型", "名称", "标签", "形状", "dtype", "缓存", "缓存体积", "内存体积", "时间"])
        self.current_tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.current_tree.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.current_tree.setRootIsDecorated(False)
        self.current_tree.setSortingEnabled(True)
        self.current_tree.itemDoubleClicked.connect(self._current_item_double_clicked)
        self.current_tree.itemChanged.connect(lambda _item, _column: self._update_action_states())
        self._install_edit_actions(self.current_tree, self._edit_current_item)
        self._install_visible_select_all(self.current_tree)
        self.current_tree.itemSelectionChanged.connect(self._update_action_states)
        layout.addWidget(self.current_tree)

        buttons = QHBoxLayout()
        self.select_btn = QPushButton("设为当前数据")
        self.force_cache_btn = QPushButton("强制缓存落盘")
        self.delete_history_btn = QPushButton("删除当前历史")
        self.edit_current_btn = QPushButton("编辑名称与标签")
        self.batch_tag_current_btn = QPushButton("批量编辑标签")
        self.select_btn.clicked.connect(self._select_current_item)
        self.force_cache_btn.clicked.connect(self._force_cache_current_item)
        self.delete_history_btn.clicked.connect(self._delete_current_item)
        self.edit_current_btn.clicked.connect(self._edit_current_item)
        self.batch_tag_current_btn.clicked.connect(self._batch_tag_current)
        buttons.addWidget(self.select_btn)
        buttons.addWidget(self.force_cache_btn)
        buttons.addWidget(self.delete_history_btn)
        buttons.addWidget(self.edit_current_btn)
        buttons.addWidget(self.batch_tag_current_btn)
        buttons.addStretch()
        layout.addLayout(buttons)
        self.current_count_label = QLabel()
        layout.addWidget(self.current_count_label)

        self.refresh_current_items(self.current_items)
        return tab

    def _build_recoverable_history_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        filters = QHBoxLayout()
        filters.addWidget(QLabel("搜索"))
        self.manifest_search = QLineEdit()
        self.manifest_search.setPlaceholderText("显示名称、原始名称或来源")
        self.manifest_search.setClearButtonEnabled(True)
        self.manifest_search.textChanged.connect(self._apply_manifest_filters)
        filters.addWidget(self.manifest_search, 1)
        self.manifest_type_filter = QComboBox()
        self.manifest_type_filter.currentIndexChanged.connect(self._apply_manifest_filters)
        filters.addWidget(self.manifest_type_filter)
        self.manifest_tag_filter = TagFilterButton()
        self.manifest_tag_filter.filterChanged.connect(self._apply_manifest_filters)
        filters.addWidget(self.manifest_tag_filter)
        clear_filters = QPushButton("清除筛选")
        clear_filters.setToolTip("清除名称、类型和标签筛选，不修改缓存索引")
        clear_filters.clicked.connect(self._clear_manifest_filters)
        filters.addWidget(clear_filters)
        style_filter_controls(
            self.manifest_search, self.manifest_type_filter, self.manifest_tag_filter, clear_filters
        )
        layout.addLayout(filters)
        self.manifest_tree = QTreeWidget()
        self.manifest_tree.setHeaderLabels(["", "类型", "名称", "标签", "形状", "dtype", "缓存文件", "缓存体积", "状态", "保存时间"])
        self.manifest_tree.setRootIsDecorated(False)
        self.manifest_tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.manifest_tree.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.manifest_tree.setSortingEnabled(True)
        self._install_edit_actions(self.manifest_tree, self._edit_manifest_item)
        self.manifest_tree.itemDoubleClicked.connect(self._manifest_item_double_clicked)
        self.manifest_tree.itemChanged.connect(lambda _item, _column: self._update_action_states())
        self._install_visible_select_all(self.manifest_tree)
        self.manifest_tree.itemSelectionChanged.connect(self._update_action_states)
        layout.addWidget(self.manifest_tree)

        buttons = QHBoxLayout()
        self.recover_btn = QPushButton("恢复选中项")
        self.delete_manifest_btn = QPushButton("删除选中索引")
        self.refresh_manifest_btn = QPushButton("刷新")
        self.edit_manifest_btn = QPushButton("编辑名称与标签")
        self.batch_tag_manifest_btn = QPushButton("批量编辑标签")
        self.recover_btn.clicked.connect(self._recover_manifest_item)
        self.delete_manifest_btn.clicked.connect(self._delete_manifest_item)
        self.refresh_manifest_btn.clicked.connect(self.refresh_requested.emit)
        self.edit_manifest_btn.clicked.connect(self._edit_manifest_item)
        self.batch_tag_manifest_btn.clicked.connect(self._batch_tag_manifest)
        buttons.addWidget(self.recover_btn)
        buttons.addWidget(self.delete_manifest_btn)
        buttons.addWidget(self.refresh_manifest_btn)
        buttons.addWidget(self.edit_manifest_btn)
        buttons.addWidget(self.batch_tag_manifest_btn)
        buttons.addStretch()
        layout.addLayout(buttons)
        self.manifest_count_label = QLabel()
        layout.addWidget(self.manifest_count_label)

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

    def _install_edit_actions(self, tree, callback):
        action = QAction("编辑名称与标签", tree)
        action.setShortcut(QKeySequence(Qt.Key_F2))
        action.setShortcutContext(Qt.WidgetWithChildrenShortcut)
        action.setToolTip("修改显示名称和最多三个表情标签；不会改变程序生成的原始名称")
        action.triggered.connect(callback)
        tree.addAction(action)
        tree.setContextMenuPolicy(Qt.ActionsContextMenu)

    def _install_visible_select_all(self, tree):
        action = QAction("选择全部可见项", tree)
        action.setShortcut(QKeySequence.SelectAll)
        action.setShortcutContext(Qt.WidgetWithChildrenShortcut)
        action.triggered.connect(lambda: self._select_visible(tree))
        tree.addAction(action)

    @staticmethod
    def _select_visible(tree):
        tree.clearSelection()
        for index in range(tree.topLevelItemCount()):
            item = tree.topLevelItem(index)
            if not item.isHidden():
                item.setCheckState(0, Qt.Checked)
                item.setSelected(True)

    @staticmethod
    def _restore_tree_selection(tree, selected_ids, checked_ids, scroll_value):
        for index in range(tree.topLevelItemCount()):
            item = tree.topLevelItem(index)
            item.setCheckState(0, Qt.Checked if item.data(0, Qt.UserRole) in checked_ids else Qt.Unchecked)
            if item.data(0, Qt.UserRole) in selected_ids and not item.isHidden():
                item.setSelected(True)
                tree.setCurrentItem(item)
        tree.verticalScrollBar().setValue(scroll_value)

    @staticmethod
    def _update_filter_controls(combo, tag_filter, metadata_items):
        selected_category = combo.currentData() if combo.count() else ""
        categories = sorted({item.get("category", "") for item in metadata_items if item.get("category")})
        tags = []
        for item in metadata_items:
            for tag in item.get("tags", ()):
                if tag not in tags:
                    tags.append(tag)
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("全部类型", "")
        for category in categories:
            combo.addItem(category, category)
        combo.setCurrentIndex(max(0, combo.findData(selected_category)))
        combo.blockSignals(False)
        tag_filter.set_tags(tags)

    @staticmethod
    def _apply_tree_filters(tree, query, category, tag_filter):
        for index in range(tree.topLevelItemCount()):
            item = tree.topLevelItem(index)
            metadata = item.data(0, FILTER_ROLE) or {}
            visible = metadata_matches(
                metadata,
                query=query,
                category=category,
                tags=tag_filter.selected_tags(),
                untagged=tag_filter.untagged_selected(),
            )
            item.setHidden(not visible)
            if not visible:
                item.setSelected(False)
                item.setCheckState(0, Qt.Unchecked)

    def _apply_current_filters(self, _value=None):
        self._apply_tree_filters(
            self.current_tree,
            self.current_search.text(),
            self.current_type_filter.currentData() or "",
            self.current_tag_filter,
        )
        self._update_action_states()

    def _apply_manifest_filters(self, _value=None):
        self._apply_tree_filters(
            self.manifest_tree,
            self.manifest_search.text(),
            self.manifest_type_filter.currentData() or "",
            self.manifest_tag_filter,
        )
        self._update_action_states()

    def _clear_current_filters(self):
        self.current_search.clear()
        self.current_type_filter.setCurrentIndex(0)
        self.current_tag_filter.clear_filter()

    def _clear_manifest_filters(self):
        self.manifest_search.clear()
        self.manifest_type_filter.setCurrentIndex(0)
        self.manifest_tag_filter.clear_filter()

    def refresh_current_items(self, items):
        self.current_items = list(items or [])
        selected_ids = {
            item.data(0, Qt.UserRole) for item in self.current_tree.selectedItems()
        }
        checked_ids = {
            self.current_tree.topLevelItem(index).data(0, Qt.UserRole)
            for index in range(self.current_tree.topLevelItemCount())
            if self.current_tree.topLevelItem(index).checkState(0) == Qt.Checked
        }
        scroll_value = self.current_tree.verticalScrollBar().value()
        self.current_tree.setSortingEnabled(False)
        self.current_tree.clear()
        filter_items = []
        for item in self.current_items:
            tags = tuple(item.get("tags") or ())
            category = item.get("data_category", "")
            tree_item = SortableTreeWidgetItem([
                "",
                item.get("kind", ""),
                item.get("name", ""),
                " ".join(tags),
                str(item.get("shape", "")),
                str(item.get("dtype", "")),
                item.get("cache_state", ""),
                format_bytes(item.get("cached_bytes", 0)),
                format_bytes(item.get("memory_bytes", 0)),
                format_timestamp(item.get("timestamp")),
            ])
            tree_item.setFlags(tree_item.flags() | Qt.ItemIsUserCheckable)
            tree_item.setCheckState(0, Qt.Unchecked)
            tree_item.setData(
                0, Qt.UserRole,
                tuple(item.get("identity") or (item.get("kind"), item.get("serial_number"), item.get("timestamp"))),
            )
            metadata = {
                "display_name": item.get("name", ""),
                "original_name": item.get("original_name", item.get("name", "")),
                "source_name": item.get("source_name", ""),
                "category": category,
                "tags": tags,
            }
            tree_item.setData(0, FILTER_ROLE, metadata)
            tree_item.setToolTip(2, f"显示名称：{metadata['display_name']}\n原始名称：{metadata['original_name']}")
            tree_item.setToolTip(3, " ".join(tags) if tags else "无标签")
            for column, value in {7: item.get("cached_bytes", 0), 8: item.get("memory_bytes", 0), 9: float(item.get("timestamp") or 0)}.items():
                tree_item.setData(column, SORT_ROLE, value)
            self.current_tree.addTopLevelItem(tree_item)
            filter_items.append(metadata)
        self.current_tree.setSortingEnabled(True)
        self._update_filter_controls(self.current_type_filter, self.current_tag_filter, filter_items)
        self._apply_current_filters()
        self._restore_tree_selection(self.current_tree, selected_ids, checked_ids, scroll_value)
        self.current_tree.setColumnWidth(0, 34)
        self.current_tree.resizeColumnToContents(1)
        self.current_tree.setColumnWidth(2, 260)
        self.current_tree.resizeColumnToContents(3)
        self._update_action_states()

    def refresh_manifest_items(self, items):
        self.manifest_items = list(items or [])
        selected_ids = {item.data(0, Qt.UserRole) for item in self.manifest_tree.selectedItems()}
        checked_ids = {
            self.manifest_tree.topLevelItem(index).data(0, Qt.UserRole)
            for index in range(self.manifest_tree.topLevelItemCount())
            if self.manifest_tree.topLevelItem(index).checkState(0) == Qt.Checked
        }
        scroll_value = self.manifest_tree.verticalScrollBar().value()
        self.manifest_tree.setSortingEnabled(False)
        self.manifest_tree.clear()
        filter_items = []
        for item in self.manifest_items:
            arrays = item.get("arrays") or {}
            status = item.get("file_status", {})
            cache_bytes = sum(int(array.get("nbytes", 0)) for array in arrays.values())
            saved_at = float(item.get("saved_at") or 0)
            tags = tuple(tags_for(item))
            display_name = display_name_for(item)
            category = item.get("data_category") or describe_value(
                shape=item.get("shape", ()),
                dtype=item.get("dtype", ""),
                semantic_hint=item.get("type_processed") or item.get("format_import"),
            ).label
            tree_item = SortableTreeWidgetItem([
                "",
                item.get("kind", ""),
                display_name,
                " ".join(tags),
                str(item.get("shape", "")),
                str(item.get("dtype", "")),
                str(len(arrays)),
                format_bytes(cache_bytes),
                "可用" if status.get("ok", True) else "缺失",
                format_timestamp(saved_at),
            ])
            tree_item.setFlags(tree_item.flags() | Qt.ItemIsUserCheckable)
            tree_item.setCheckState(0, Qt.Unchecked)
            tree_item.setData(0, Qt.UserRole, item.get("id", ""))
            metadata = {
                "display_name": display_name,
                "original_name": item.get("name", ""),
                "source_name": item.get("source_name", ""),
                "category": category,
                "tags": tags,
            }
            tree_item.setData(0, FILTER_ROLE, metadata)
            tree_item.setToolTip(2, f"显示名称：{display_name}\n原始名称：{item.get('name', '')}")
            tree_item.setToolTip(3, " ".join(tags) if tags else "无标签")
            for column, value in {6: len(arrays), 7: cache_bytes, 8: 0 if status.get("ok", True) else 1, 9: saved_at}.items():
                tree_item.setData(column, SORT_ROLE, value)
            self.manifest_tree.addTopLevelItem(tree_item)
            filter_items.append(metadata)
        self.manifest_tree.setSortingEnabled(True)
        self._update_filter_controls(self.manifest_type_filter, self.manifest_tag_filter, filter_items)
        self._apply_manifest_filters()
        self._restore_tree_selection(self.manifest_tree, selected_ids, checked_ids, scroll_value)
        self.manifest_tree.setColumnWidth(0, 34)
        self.manifest_tree.resizeColumnToContents(1)
        self.manifest_tree.setColumnWidth(2, 260)
        self.manifest_tree.resizeColumnToContents(3)
        self._update_action_states()

    def _selected_current_identity(self):
        chosen = self._chosen_items(self.current_tree)
        if len(chosen) != 1:
            return None
        item = chosen[0]
        identity = item.data(0, Qt.UserRole)
        if not isinstance(identity, tuple) or len(identity) != 3:
            return None
        kind, serial_number, timestamp = identity
        try:
            timestamp = float(timestamp)
        except (TypeError, ValueError):
            return None
        return kind, serial_number, timestamp

    def _selected_current_identities(self):
        identities = []
        for item in self._chosen_items(self.current_tree):
            value = item.data(0, Qt.UserRole)
            if isinstance(value, tuple) and len(value) == 3:
                identities.append(value)
        return identities

    def _selected_manifest_id(self):
        chosen = self._chosen_items(self.manifest_tree)
        if len(chosen) != 1:
            return None
        item = chosen[0]
        item_id = item.data(0, Qt.UserRole)
        return str(item_id) if item_id else None

    def _selected_manifest_ids(self):
        return [
            str(item.data(0, Qt.UserRole))
            for item in self._chosen_items(self.manifest_tree)
            if item.data(0, Qt.UserRole)
        ]

    def _select_current_item(self):
        identity = self._selected_current_identity()
        if identity:
            self.select_history_requested.emit(identity)

    def _force_cache_current_item(self):
        identities = self._selected_current_identities()
        if len(identities) == 1:
            self.force_cache_requested.emit(identities[0])
        elif identities:
            self.batch_force_cache_requested.emit(identities)

    def _delete_current_item(self):
        identities = self._selected_current_identities()
        if len(identities) == 1:
            self.delete_history_requested.emit(identities[0])
        elif identities:
            self.batch_delete_history_requested.emit(identities)

    def _edit_current_item(self):
        identity = self._selected_current_identity()
        if identity:
            self.edit_current_requested.emit(identity)

    def _batch_tag_current(self):
        identities = self._selected_current_identities()
        if identities:
            self.batch_tag_current_requested.emit(identities)

    def open_cache_directory(self):
        directory = Path(self.cache_directory_edit.text().strip())
        directory.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(directory)))

    def _recover_manifest_item(self):
        item_ids = self._selected_manifest_ids()
        if len(item_ids) == 1:
            self.recover_manifest_requested.emit(item_ids[0])
        elif item_ids:
            self.batch_recover_manifest_requested.emit(item_ids)

    def _delete_manifest_item(self):
        item_ids = self._selected_manifest_ids()
        if len(item_ids) == 1:
            self.delete_manifest_requested.emit(item_ids[0])
        elif item_ids:
            self.batch_delete_manifest_requested.emit(item_ids)

    def _edit_manifest_item(self):
        item_id = self._selected_manifest_id()
        if item_id:
            self.edit_manifest_requested.emit(item_id)

    def _batch_tag_manifest(self):
        item_ids = self._selected_manifest_ids()
        if item_ids:
            self.batch_tag_manifest_requested.emit(item_ids)

    @staticmethod
    def _chosen_items(tree):
        checked = [
            tree.topLevelItem(index)
            for index in range(tree.topLevelItemCount())
            if tree.topLevelItem(index).checkState(0) == Qt.Checked
            and not tree.topLevelItem(index).isHidden()
        ]
        return checked or [item for item in tree.selectedItems() if not item.isHidden()]

    def _current_item_double_clicked(self, item, column):
        if column in (2, 3):
            identity = item.data(0, Qt.UserRole)
            if identity:
                self.edit_current_requested.emit(identity)
        else:
            self._select_current_item()

    def _manifest_item_double_clicked(self, item, column):
        if column in (2, 3):
            item_id = item.data(0, Qt.UserRole)
            if item_id:
                self.edit_manifest_requested.emit(str(item_id))

    def _update_action_states(self):
        if hasattr(self, "current_tree"):
            current_count = len(self._chosen_items(self.current_tree))
            visible = sum(
                not self.current_tree.topLevelItem(index).isHidden()
                for index in range(self.current_tree.topLevelItemCount())
            )
            self.current_count_label.setText(
                f"已选 {current_count} 项 / 显示 {visible} 项 / 共 {self.current_tree.topLevelItemCount()} 项"
            )
            self.select_btn.setEnabled(current_count == 1)
            self.edit_current_btn.setEnabled(current_count == 1)
            self.force_cache_btn.setEnabled(current_count > 0)
            self.delete_history_btn.setEnabled(current_count > 0)
            self.batch_tag_current_btn.setEnabled(current_count > 0)
        if hasattr(self, "manifest_tree"):
            manifest_count = len(self._chosen_items(self.manifest_tree))
            visible = sum(
                not self.manifest_tree.topLevelItem(index).isHidden()
                for index in range(self.manifest_tree.topLevelItemCount())
            )
            self.manifest_count_label.setText(
                f"已选 {manifest_count} 项 / 显示 {visible} 项 / 共 {self.manifest_tree.topLevelItemCount()} 项"
            )
            self.recover_btn.setEnabled(manifest_count > 0)
            self.edit_manifest_btn.setEnabled(manifest_count == 1)
            self.delete_manifest_btn.setEnabled(manifest_count > 0)
            self.batch_tag_manifest_btn.setEnabled(manifest_count > 0)
