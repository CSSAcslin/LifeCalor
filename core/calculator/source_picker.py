from __future__ import annotations

from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QDialog, QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget,
)

from DataManager import Data, ProcessedData
from widget.DataTreeWidget import DataHistoryTreeWidget, DataTreeEntry


class CalculatorSourcePicker(QDialog):
    """Pick one calculator operand from the shared history tree."""

    def __init__(self, source_provider, parent=None):
        super().__init__(parent)
        self.source_provider = source_provider
        self.selected_entry = None
        self.setWindowTitle("添加运算数据")
        self.resize(980, 580)
        self._setup_ui()
        self.refresh_sources()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        header = QHBoxLayout()
        header.addWidget(QLabel("搜索"))
        self.search = QLineEdit()
        self.search.setClearButtonEnabled(True)
        self.search.setPlaceholderText("输入数据名称、类型、Shape 或 key")
        self.search.setToolTip("只过滤当前列表，不会删除或修改历史数据")
        self.search.textChanged.connect(self._filter)
        header.addWidget(self.search, 1)
        refresh = QPushButton("刷新")
        refresh.setToolTip("重新读取当前 Data 和 ProcessedData 历史")
        refresh.clicked.connect(self.refresh_sources)
        header.addWidget(refresh)
        layout.addLayout(header)

        self.tree = DataHistoryTreeWidget(self, self._action_factory, action_title="加入运算")
        self.tree.entry_double_clicked.connect(self._choose)
        layout.addWidget(self.tree, 1)

        footer = QHBoxLayout()
        self.hint = QLabel("展开 Other Results 可选择 out_processed 中的数组")
        self.hint.setObjectName("mutedLabel")
        footer.addWidget(self.hint, 1)
        cancel = QPushButton("取消")
        cancel.clicked.connect(self.reject)
        footer.addWidget(cancel)
        layout.addLayout(footer)

    def refresh_sources(self):
        sources = list(self.source_provider() or [])
        data = [source for source in sources if isinstance(source, Data)]
        processed = [source for source in sources if isinstance(source, ProcessedData)]
        self.tree.refresh_data(data, processed)
        self._filter(self.search.text())

    def _action_factory(self, _tree, _item, entry: DataTreeEntry):
        button = QPushButton(QIcon(':icons/icon_add.svg'), "添加")
        button.setToolTip(
            f"将“{entry.label}”加入计算器并分配下一个空闲别名；不会复制或修改源数据"
        )
        button.clicked.connect(lambda _checked=False, value=entry: self._choose(value))
        container = QWidget()
        row = QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(button)
        return container

    def _choose(self, entry):
        if not isinstance(entry, DataTreeEntry) or not entry.selectable:
            return
        self.selected_entry = entry
        self.accept()

    def _filter(self, text):
        self.tree.filter_text(text)

