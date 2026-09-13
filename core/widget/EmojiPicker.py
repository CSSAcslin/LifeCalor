from __future__ import annotations

from PyQt5.QtCore import QSettings, Qt, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QGridLayout, QHBoxLayout, QLabel,
    QLineEdit, QScrollArea, QToolButton, QVBoxLayout, QWidget,
)
from widget.DataFilterControls import style_filter_controls


EMOJI_CATEGORIES = {
    "常用": [
        ("✅", "完成 对 勾"), ("⭐", "星 收藏"), ("❤️", "爱心 重要"),
        ("🔥", "火 热点"), ("💡", "灯泡 想法"), ("📌", "图钉 标记"),
        ("🔬", "显微镜 实验"), ("🧪", "试管 实验"), ("🧬", "DNA 生物"),
        ("📊", "图表 统计"), ("⚠️", "警告 注意"), ("❌", "错误 否"),
    ],
    "状态": [
        ("🟢", "绿色 正常"), ("🟡", "黄色 等待"), ("🔴", "红色 异常"),
        ("🔵", "蓝色"), ("🟣", "紫色"), ("⚫", "黑色"),
        ("⚪", "白色"), ("🟠", "橙色"), ("🟤", "棕色"),
        ("⬆️", "上升"), ("⬇️", "下降"), ("➡️", "继续"),
    ],
    "实验": [
        ("🔬", "显微镜"), ("🧪", "试管"), ("🧫", "培养皿"),
        ("🧬", "DNA"), ("⚗️", "蒸馏 化学"), ("🩸", "血液"),
        ("🧠", "大脑"), ("🫀", "心脏"), ("🌡️", "温度"),
        ("🧲", "磁铁"), ("⚡", "电 信号"), ("💊", "药物"),
    ],
    "标记": [
        ("1️⃣", "数字 一"), ("2️⃣", "数字 二"), ("3️⃣", "数字 三"),
        ("🅰️", "A"), ("🅱️", "B"), ("🆎", "AB"),
        ("➕", "加"), ("➖", "减"), ("♻️", "循环"),
        ("🔒", "锁定"), ("🔓", "解锁"), ("🏷️", "标签"),
    ],
    "表情": [
        ("😀", "开心"), ("🙂", "微笑"), ("🤔", "思考"),
        ("😮", "惊讶"), ("😢", "难过"), ("😴", "休眠"),
        ("👍", "赞"), ("👎", "不赞"), ("👏", "鼓掌"),
        ("🙏", "感谢"), ("👀", "查看"), ("💪", "强"),
    ],
}


class EmojiPickerDialog(QDialog):
    emoji_selected = pyqtSignal(str)

    def __init__(self, current="", parent=None):
        super().__init__(parent)
        self.current_emoji = str(current or "")
        self.selected_emoji = ""
        self.setWindowTitle("选择表情标签")
        self.resize(430, 390)
        layout = QVBoxLayout(self)
        header = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText("搜索表情用途，例如：实验、完成、警告")
        self.search.setClearButtonEnabled(True)
        self.category = QComboBox()
        self.category.addItem("全部", "")
        for name in EMOJI_CATEGORIES:
            self.category.addItem(name, name)
        header.addWidget(self.search, 1)
        header.addWidget(self.category)
        style_filter_controls(self.search, self.category)
        layout.addLayout(header)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.content = QWidget()
        self.grid = QGridLayout(self.content)
        self.grid.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.scroll.setWidget(self.content)
        layout.addWidget(self.scroll, 1)

        current_text = f"当前：{self.current_emoji}　" if self.current_emoji else ""
        hint = QLabel(current_text + "单击即可选择；组合表情会作为一个完整标签保存。")
        layout.addWidget(hint)
        buttons = QDialogButtonBox(QDialogButtonBox.Cancel)
        clear = buttons.addButton("清除标签", QDialogButtonBox.ResetRole)
        clear.clicked.connect(self._clear)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.search.textChanged.connect(self._rebuild)
        self.category.currentIndexChanged.connect(self._rebuild)
        self._rebuild()

    @staticmethod
    def _recent():
        value = QSettings().value("history/recent_emoji_tags", [])
        if isinstance(value, str):
            value = [value]
        return [str(item) for item in value or []]

    @staticmethod
    def _remember(value):
        recent = [value] + [item for item in EmojiPickerDialog._recent() if item != value]
        QSettings().setValue("history/recent_emoji_tags", recent[:12])

    def _entries(self):
        category = self.category.currentData()
        query = self.search.text().casefold().strip()
        entries = []
        if not category:
            entries.extend((emoji, "最近 " + emoji) for emoji in self._recent())
        for group, values in EMOJI_CATEGORIES.items():
            if category and group != category:
                continue
            for emoji, keywords in values:
                if query and query not in f"{emoji} {group} {keywords}".casefold():
                    continue
                if emoji not in [entry[0] for entry in entries]:
                    entries.append((emoji, f"{group} · {keywords}"))
        return entries

    def _rebuild(self):
        while self.grid.count():
            item = self.grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        font = QFont("Segoe UI Emoji", 18)
        entries = self._entries()
        for index, (emoji, tooltip) in enumerate(entries):
            button = QToolButton()
            button.setText(emoji)
            button.setFont(font)
            button.setFixedSize(48, 48)
            button.setToolTip(tooltip)
            button.clicked.connect(lambda _checked=False, value=emoji: self._choose(value))
            self.grid.addWidget(button, index // 7, index % 7)
        if not entries:
            self.grid.addWidget(QLabel("没有匹配的表情"), 0, 0)

    def _choose(self, value):
        self.selected_emoji = value
        self._remember(value)
        self.emoji_selected.emit(value)
        self.accept()

    def _clear(self):
        self.selected_emoji = ""
        self.accept()

    @classmethod
    def pick(cls, current="", parent=None):
        dialog = cls(current=current, parent=parent)
        if dialog.exec_() == QDialog.Accepted:
            return dialog.selected_emoji, True
        return current, False


class EmojiTagSlot(QToolButton):
    """Square tag slot that opens the in-app emoji picker."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._emoji = ""
        self.setFixedSize(54, 54)
        self.setFont(QFont("Segoe UI Emoji", 18))
        self.setToolTip("单击选择表情标签；在选择框中可搜索或清除")
        self.clicked.connect(self._pick)
        self.setText("")

    def text(self):
        return self._emoji

    def setText(self, value):
        self._emoji = str(value or "").strip()
        super().setText(self._emoji or "+")
        self.setProperty("hasEmoji", bool(self._emoji))
        self.style().unpolish(self)
        self.style().polish(self)

    def _pick(self):
        value, accepted = EmojiPickerDialog.pick(self._emoji, self)
        if accepted:
            self.setText(value)
