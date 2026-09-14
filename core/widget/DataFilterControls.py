from __future__ import annotations

import unicodedata

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QAction, QMenu, QToolButton


def style_filter_controls(*widgets):
    for widget in widgets:
        widget.setProperty("filterControl", True)
        widget.setFixedHeight(30)


def normalized_search_text(value) -> str:
    return unicodedata.normalize("NFC", str(value or "")).casefold().strip()


def metadata_matches(metadata: dict, query="", category="", tags=(), untagged=False) -> bool:
    query = normalized_search_text(query)
    category = str(category or "")
    available_tags = tuple(metadata.get("tags") or ())
    if query:
        searchable = " ".join(
            str(metadata.get(key, ""))
            for key in ("display_name", "original_name", "source_name", "payload_key")
        )
        if query not in normalized_search_text(searchable):
            return False
    if category and str(metadata.get("category", "")) != category:
        return False
    if untagged:
        return not available_tags
    selected = set(tags or ())
    return not selected or bool(selected.intersection(available_tags))


class TagFilterButton(QToolButton):
    filterChanged = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setPopupMode(QToolButton.InstantPopup)
        self.setProperty("filterControl", True)
        self.setToolTip("筛选包含任意一个所选表情标签的数据；“无标签”与具体标签互斥")
        self._menu = QMenu(self)
        self.setMenu(self._menu)
        self._tag_actions = {}
        self._untagged_action = None
        self.set_tags([])

    def set_tags(self, tags):
        selected = self.selected_tags()
        untagged = self.untagged_selected()
        self._menu.clear()
        self._tag_actions = {}
        clear = self._menu.addAction("全部标签")
        clear.triggered.connect(self.clear_filter)
        self._menu.addSeparator()
        self._untagged_action = self._menu.addAction("无标签")
        self._untagged_action.setCheckable(True)
        self._untagged_action.setChecked(untagged)
        self._untagged_action.toggled.connect(self._untagged_toggled)
        unique = []
        for tag in tags or []:
            if tag and tag not in unique:
                unique.append(tag)
        if unique:
            self._menu.addSeparator()
        for tag in unique:
            action = QAction(tag, self._menu)
            action.setCheckable(True)
            action.setChecked(tag in selected and not untagged)
            action.toggled.connect(self._tag_toggled)
            self._menu.addAction(action)
            self._tag_actions[tag] = action
        self._update_text()

    def selected_tags(self):
        return tuple(tag for tag, action in self._tag_actions.items() if action.isChecked())

    def untagged_selected(self):
        return bool(self._untagged_action and self._untagged_action.isChecked())

    def clear_filter(self):
        for action in self._tag_actions.values():
            action.blockSignals(True)
            action.setChecked(False)
            action.blockSignals(False)
        if self._untagged_action is not None:
            self._untagged_action.blockSignals(True)
            self._untagged_action.setChecked(False)
            self._untagged_action.blockSignals(False)
        self._update_text()
        self.filterChanged.emit()

    def _untagged_toggled(self, checked):
        if checked:
            for action in self._tag_actions.values():
                action.blockSignals(True)
                action.setChecked(False)
                action.blockSignals(False)
        self._update_text()
        self.filterChanged.emit()

    def _tag_toggled(self, checked):
        if checked and self._untagged_action is not None:
            self._untagged_action.blockSignals(True)
            self._untagged_action.setChecked(False)
            self._untagged_action.blockSignals(False)
        self._update_text()
        self.filterChanged.emit()

    def _update_text(self):
        if self.untagged_selected():
            self.setText("无标签")
        elif self.selected_tags():
            self.setText("标签 " + " ".join(self.selected_tags()))
        else:
            self.setText("全部标签")
