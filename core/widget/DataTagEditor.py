from __future__ import annotations

from collections.abc import Mapping

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from history.annotations import (
    annotations_for,
    display_name_for,
    normalize_field_path,
    tags_for,
    updated_annotations,
    validate_emoji_tags,
)
from widget.EmojiPicker import EmojiTagSlot


NON_DATA_PARAMETER_KEYS = {
    "fps", "time_step", "time_unit", "duration", "space_step", "space_unit",
    "axes", "axis_order", "scientific_axes", "source_axes", "display_axes",
}


def editable_fields(target) -> list[tuple[tuple[str, ...], str]]:
    fields = []
    if isinstance(target, Mapping):
        arrays = target.get("arrays") or {}
        for field_name in arrays:
            if str(field_name).startswith("out_processed."):
                key = str(field_name).split(".", 1)[1]
                fields.append((("out_processed", key), key))
        return fields

    for key, value in (getattr(target, "out_processed", None) or {}).items():
        if str(key) in NON_DATA_PARAMETER_KEYS:
            continue
        if hasattr(value, "shape") or isinstance(value, (list, tuple)):
            fields.append((("out_processed", str(key)), str(key)))
    return fields


class DataTagEditorDialog(QDialog):
    """Edit display-only annotations while keeping generated names immutable."""

    def __init__(self, target, parent=None):
        super().__init__(parent)
        self.target = target
        self.result_annotations = annotations_for(target)
        self.current_path = None
        self.setWindowTitle("编辑名称与表情标签")
        self.resize(760, 430)
        self._build_ui()
        self._populate_targets()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        hint = QLabel("显示名称不会修改程序生成的原始名称；单击标签方框中的 + 选择表情。")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        splitter = QSplitter(Qt.Horizontal)
        self.target_tree = QTreeWidget()
        self.target_tree.setHeaderLabel("编辑对象")
        self.target_tree.setMinimumWidth(230)
        self.target_tree.currentItemChanged.connect(self._target_changed)
        splitter.addWidget(self.target_tree)

        editor = QWidget()
        form = QFormLayout(editor)
        self.original_name = QLineEdit()
        self.original_name.setReadOnly(True)
        self.original_name.setToolTip("程序生成的原始名称，仅用于追溯，用户改名不会覆写此字段")
        form.addRow("原始名称", self.original_name)

        name_row = QHBoxLayout()
        self.display_name = QLineEdit()
        self.display_name.setMaxLength(128)
        self.display_name.setClearButtonEnabled(True)
        self.display_name.setToolTip("最多 128 个字符；留空表示恢复显示原始名称")
        restore = QPushButton("恢复原名")
        restore.setToolTip("清空显示别名，界面重新显示程序生成的原始名称")
        restore.clicked.connect(self.display_name.clear)
        name_row.addWidget(self.display_name, 1)
        name_row.addWidget(restore)
        form.addRow("显示名称", name_row)

        self.follow_parent = QCheckBox("跟随所属数据的标签")
        self.follow_parent.setToolTip("子结果不保存独立标签，显示时使用所属根数据的标签")
        self.follow_parent.toggled.connect(self._update_tag_enabled)
        form.addRow("子结果标签", self.follow_parent)

        self.tag_edits = []
        tag_row = QHBoxLayout()
        for index in range(3):
            edit = EmojiTagSlot()
            edit.setToolTip(f"标签 {index + 1}：单击打开表情选择框；已设置时可重新选择或清除")
            self.tag_edits.append(edit)
            tag_row.addWidget(edit)
        tag_row.addStretch(1)
        form.addRow("表情标签", tag_row)
        splitter.addWidget(editor)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Save).setText("保存")
        buttons.accepted.connect(self._accept_changes)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _root_original_name(self):
        if isinstance(self.target, Mapping):
            return str(self.target.get("name", ""))
        return str(getattr(self.target, "name", ""))

    def _populate_targets(self):
        root = QTreeWidgetItem([self._root_original_name()])
        root.setData(0, Qt.UserRole, None)
        root.setToolTip(0, "根数据")
        self.target_tree.addTopLevelItem(root)
        fields = editable_fields(self.target)
        if fields:
            group = QTreeWidgetItem(root, ["Other Results"])
            group.setFlags(group.flags() & ~Qt.ItemIsSelectable)
            for path, original_name in fields:
                item = QTreeWidgetItem(group, [display_name_for(self.target, path)])
                item.setData(0, Qt.UserRole, path)
                item.setToolTip(0, f"原始 key：{original_name}")
            root.setExpanded(True)
            group.setExpanded(True)
        self.target_tree.setCurrentItem(root)

    def _save_local_draft(self):
        if self.current_path is None and self.original_name.text() == "":
            return True
        try:
            tags = validate_emoji_tags(edit.text() for edit in self.tag_edits)
            self.result_annotations = updated_annotations(
                self.result_annotations,
                display_name=self.display_name.text(),
                tags=tags,
                path=self.current_path,
                follow_parent_tags=self.current_path is not None and self.follow_parent.isChecked(),
            )
            return True
        except (RuntimeError, ValueError) as exc:
            QMessageBox.warning(self, "名称或标签无效", str(exc))
            return False

    def _target_changed(self, current, previous):
        if previous is not None and not self._save_local_draft():
            self.target_tree.blockSignals(True)
            self.target_tree.setCurrentItem(previous)
            self.target_tree.blockSignals(False)
            return
        if current is None or not (current.flags() & Qt.ItemIsSelectable):
            return
        self.current_path = current.data(0, Qt.UserRole)
        if self.current_path is not None:
            self.current_path = tuple(normalize_field_path(self.current_path))
            original = self.current_path[-1]
        else:
            original = self._root_original_name()
        shown = display_name_for({"name": self._root_original_name(), "annotations": self.result_annotations}, self.current_path)
        self.original_name.setText(original)
        self.display_name.setText("" if shown == original else shown)

        field_has_tags = False
        if self.current_path is not None:
            for field_info in self.result_annotations.get("fields", []):
                if tuple(field_info.get("path", ())) == self.current_path:
                    field_has_tags = "tags" in field_info
                    break
        self.follow_parent.blockSignals(True)
        self.follow_parent.setChecked(self.current_path is not None and not field_has_tags)
        self.follow_parent.setVisible(self.current_path is not None)
        self.follow_parent.blockSignals(False)
        tags = tags_for({"annotations": self.result_annotations}, self.current_path)
        for index, edit in enumerate(self.tag_edits):
            edit.setText(tags[index] if index < len(tags) else "")
        self._update_tag_enabled()

    def _update_tag_enabled(self):
        enabled = self.current_path is None or not self.follow_parent.isChecked()
        for edit in self.tag_edits:
            edit.setEnabled(enabled)

    def _accept_changes(self):
        if self._save_local_draft():
            self.accept()

    def annotations(self) -> dict:
        return self.result_annotations


class BatchTagEditorDialog(QDialog):
    """Collect one explicit add/remove/clear operation for multiple root items."""

    def __init__(self, count, parent=None):
        super().__init__(parent)
        self.setWindowTitle("批量编辑表情标签")
        self.setMinimumWidth(420)
        layout = QVBoxLayout(self)
        hint = QLabel(f"将对选中的 {int(count)} 项根数据执行同一操作；子结果的独立标签不受影响。")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        form = QFormLayout()
        self.operation_combo = QComboBox()
        self.operation_combo.addItem("添加标签", "add")
        self.operation_combo.addItem("移除标签", "remove")
        self.operation_combo.addItem("清空标签", "clear")
        self.operation_combo.currentIndexChanged.connect(self._operation_changed)
        form.addRow("操作", self.operation_combo)
        self.tag_edits = []
        tag_row = QHBoxLayout()
        for index in range(3):
            edit = EmojiTagSlot()
            edit.setToolTip(f"标签 {index + 1}：单击打开表情选择框")
            self.tag_edits.append(edit)
            tag_row.addWidget(edit)
        tag_row.addStretch(1)
        form.addRow("表情标签", tag_row)
        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.Apply | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Apply).setText("应用")
        buttons.accepted.connect(self._accept_changes)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _operation_changed(self):
        enabled = self.operation_combo.currentData() != "clear"
        for edit in self.tag_edits:
            edit.setEnabled(enabled)

    def _accept_changes(self):
        try:
            if self.operation_combo.currentData() != "clear":
                tags = validate_emoji_tags(edit.text() for edit in self.tag_edits)
                if not tags:
                    raise ValueError("请至少输入一个表情标签")
        except (RuntimeError, ValueError) as exc:
            QMessageBox.warning(self, "标签无效", str(exc))
            return
        if self.operation_combo.currentData() == "clear":
            answer = QMessageBox.question(
                self, "清空标签", "将清空所有选中数据的根标签，是否继续？",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return
        self.accept()

    def operation(self):
        return self.operation_combo.currentData()

    def tags(self):
        if self.operation() == "clear":
            return []
        return validate_emoji_tags(edit.text() for edit in self.tag_edits)
