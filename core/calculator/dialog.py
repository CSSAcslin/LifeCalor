from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import QRegularExpression, Qt, QTimer
from PyQt5.QtGui import QColor, QFont, QSyntaxHighlighter, QTextCharFormat
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ArrayCache import ArrayRef
from DataManager import Data, ProcessedData
from diagnostics import report_warning
from .engine import CalculationEngine
from .model import CalculationPlan, OperandSpec


@dataclass(frozen=True)
class _SourceEntry:
    source: object
    payload_key: str | None
    label: str
    kind: str
    shape: tuple[int, ...]
    dtype: str
    axes: str


class _ExpressionHighlighter(QSyntaxHighlighter):
    def __init__(self, document):
        super().__init__(document)
        self.aliases = set()
        self.function_format = QTextCharFormat()
        self.function_format.setForeground(QColor("#7c3aed"))
        self.function_format.setFontWeight(QFont.DemiBold)
        self.alias_format = QTextCharFormat()
        self.alias_format.setForeground(QColor("#0369a1"))
        self.alias_format.setFontWeight(QFont.Bold)
        self.number_format = QTextCharFormat()
        self.number_format.setForeground(QColor("#b45309"))

    def set_aliases(self, aliases):
        self.aliases = set(aliases)
        self.rehighlight()

    def highlightBlock(self, text):
        for name in CalculationEngine.FUNCTIONS:
            expression = QRegularExpression(rf"\b{name}(?=\s*\()")
            iterator = expression.globalMatch(text)
            while iterator.hasNext():
                match = iterator.next()
                self.setFormat(match.capturedStart(), match.capturedLength(), self.function_format)
        for alias in self.aliases:
            expression = QRegularExpression(rf"\b{QRegularExpression.escape(alias)}\b")
            iterator = expression.globalMatch(text)
            while iterator.hasNext():
                match = iterator.next()
                self.setFormat(match.capturedStart(), match.capturedLength(), self.alias_format)
        iterator = QRegularExpression(r"(?<![A-Za-z_])(?:\d+\.?\d*|\.\d+)").globalMatch(text)
        while iterator.hasNext():
            match = iterator.next()
            self.setFormat(match.capturedStart(), match.capturedLength(), self.number_format)


class DataCalculatorDialog(QDialog):
    COL_USE = 0
    COL_ALIAS = 1
    COL_NAME = 2
    COL_PAYLOAD = 3
    COL_SHAPE = 4
    COL_DTYPE = 5
    COL_AXES = 6
    COL_SLICE = 7

    def __init__(self, sources, parent=None):
        super().__init__(parent)
        self.setWindowTitle("多数据运算工作台")
        self.resize(1280, 760)
        self.entries = self._build_entries(sources)
        self.validation = None
        self._building_table = False
        self._setup_ui()
        self._populate_sources()
        self._connect_signals()
        self._schedule_validation()

    @staticmethod
    def _default_axes(shape):
        return {1: "T", 2: "YX", 3: "TYX", 4: "TYXC"}.get(len(shape), "?" * len(shape))

    @classmethod
    def _entry(cls, source, payload_key=None):
        spec = OperandSpec("A", source, payload_key)
        info = CalculationEngine.source_info(spec)
        parameters = getattr(source, "out_processed", None) if isinstance(source, ProcessedData) else getattr(source, "parameters", None)
        parameters = parameters or {}
        axes = str(parameters.get("scientific_axes") or parameters.get("source_axes") or cls._default_axes(info.shape))
        return _SourceEntry(
            source,
            payload_key,
            getattr(source, "name", "未命名数据"),
            source.__class__.__name__,
            info.shape,
            str(info.dtype),
            axes,
        )

    @classmethod
    def _build_entries(cls, sources):
        entries = []
        seen = set()
        for source in sources:
            identity = (source.__class__.__name__, getattr(source, "timestamp", None), getattr(source, "serial_number", id(source)), None)
            if identity not in seen:
                entries.append(cls._entry(source))
                seen.add(identity)
            for key, value in (getattr(source, "out_processed", None) or {}).items():
                if isinstance(value, (np.ndarray, ArrayRef)):
                    identity = (source.__class__.__name__, getattr(source, "timestamp", None), getattr(source, "serial_number", id(source)), key)
                    if identity not in seen:
                        entries.append(cls._entry(source, key))
                        seen.add(identity)
        return entries

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._build_source_panel())
        splitter.addWidget(self._build_expression_panel())
        splitter.addWidget(self._build_inspection_panel())
        splitter.setSizes([420, 430, 430])
        root.addWidget(splitter, 1)

        footer = QHBoxLayout()
        self.summary_label = QLabel("等待验证")
        self.summary_label.setMinimumWidth(420)
        footer.addWidget(self.summary_label, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.Cancel | QDialogButtonBox.Ok)
        self.execute_button = buttons.button(QDialogButtonBox.Ok)
        self.execute_button.setText("执行运算")
        self.execute_button.setObjectName("StressButton")
        buttons.button(QDialogButtonBox.Cancel).setText("取消")
        buttons.accepted.connect(self._accept_if_valid)
        buttons.rejected.connect(self.reject)
        footer.addWidget(buttons)
        root.addLayout(footer)

    def _panel(self, title):
        panel = QGroupBox(title)
        panel.setObjectName("calculatorPanel")
        return panel

    def _build_source_panel(self):
        panel = self._panel("输入数据")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 12, 8, 8)
        self.source_table = QTableWidget(0, 8)
        self.source_table.setHorizontalHeaderLabels(["使用", "别名", "数据", "字段", "Shape", "Dtype", "轴", "预处理切片"])
        self.source_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.source_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.source_table.setAlternatingRowColors(True)
        self.source_table.verticalHeader().hide()
        header = self.source_table.horizontalHeader()
        header.setSectionResizeMode(self.COL_USE, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(self.COL_ALIAS, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(self.COL_NAME, QHeaderView.Stretch)
        for column in (self.COL_PAYLOAD, self.COL_SHAPE, self.COL_DTYPE, self.COL_AXES, self.COL_SLICE):
            header.setSectionResizeMode(column, QHeaderView.ResizeToContents)
        layout.addWidget(self.source_table, 1)
        note = QLabel("切片示例：42, :, :　或　100:500, 20:400, :")
        note.setObjectName("mutedLabel")
        layout.addWidget(note)
        return panel

    def _build_expression_panel(self):
        panel = self._panel("运算表达式")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 12, 8, 8)
        self.expression = QPlainTextEdit()
        self.expression.setPlaceholderText("例如：A - B[42, :, :] 或 mean(A, axis=0)")
        self.expression.setMaximumHeight(110)
        font = QFont("Consolas")
        font.setPointSize(12)
        self.expression.setFont(font)
        self.highlighter = _ExpressionHighlighter(self.expression.document())
        layout.addWidget(self.expression)

        self.alias_bar = QHBoxLayout()
        self.alias_bar.addWidget(QLabel("数据"))
        self.alias_bar.addStretch(1)
        layout.addLayout(self.alias_bar)

        operators = [
            ["+", "-", "*", "/", "**", "(", ")", "[", "]", ":"],
            [">", ">=", "<", "<=", "==", "!="],
            ["mean()", "max()", "min()", "sum()", "std()"],
            ["abs()", "sqrt()", "log()", "clip()", "where()", "transpose()"],
        ]
        grid = QGridLayout()
        for row, group in enumerate(operators):
            for column, label in enumerate(group):
                button = QPushButton(label)
                button.setMinimumHeight(32)
                button.clicked.connect(lambda _checked=False, value=label: self._insert_token(value))
                grid.addWidget(button, row, column)
        layout.addLayout(grid)

        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        layout.addWidget(separator)
        self.result_name = QLineEdit()
        self.result_name.setPlaceholderText("结果名称（可选）")
        layout.addWidget(self.result_name)
        layout.addStretch(1)
        return panel

    def _build_inspection_panel(self):
        panel = self._panel("验证与预览")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 12, 8, 8)
        self.inspection_tabs = QTabWidget()

        validation_widget = QWidget()
        validation_layout = QVBoxLayout(validation_widget)
        validation_layout.setContentsMargins(0, 0, 0, 0)
        self.validation_state = QLabel("等待输入")
        self.validation_state.setMinimumHeight(34)
        self.validation_state.setAlignment(Qt.AlignCenter)
        validation_layout.addWidget(self.validation_state)
        self.trace_table = QTableWidget(0, 5)
        self.trace_table.setHorizontalHeaderLabels(["步骤", "输入 Shape", "输出 Shape", "Dtype", "状态"])
        self.trace_table.verticalHeader().hide()
        self.trace_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.trace_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.trace_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for column in range(1, 5):
            self.trace_table.horizontalHeader().setSectionResizeMode(column, QHeaderView.ResizeToContents)
        validation_layout.addWidget(self.trace_table, 1)
        self.detail_label = QLabel()
        self.detail_label.setWordWrap(True)
        validation_layout.addWidget(self.detail_label)
        self.inspection_tabs.addTab(validation_widget, "验证路径")

        preview_widget = QWidget()
        preview_layout = QVBoxLayout(preview_widget)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        self.preview = pg.ImageView()
        self.preview.ui.roiBtn.hide()
        self.preview.ui.menuBtn.hide()
        self.preview.ui.histogram.setMinimumWidth(90)
        preview_layout.addWidget(self.preview)
        self.preview_label = QLabel("选择左侧数据查看样本帧")
        self.preview_label.setObjectName("mutedLabel")
        preview_layout.addWidget(self.preview_label)
        self.inspection_tabs.addTab(preview_widget, "数据预览")
        layout.addWidget(self.inspection_tabs)
        return panel

    def _populate_sources(self):
        self._building_table = True
        try:
            self.source_table.setRowCount(len(self.entries))
            aliases = self._alias_sequence(len(self.entries))
            for row, (entry, alias) in enumerate(zip(self.entries, aliases)):
                use_item = QTableWidgetItem()
                use_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsUserCheckable)
                use_item.setCheckState(Qt.Checked if row == 0 else Qt.Unchecked)
                self.source_table.setItem(row, self.COL_USE, use_item)
                values = [alias, entry.label, entry.payload_key or "主数据", str(entry.shape), entry.dtype, entry.axes, ""]
                for offset, value in enumerate(values, start=1):
                    item = QTableWidgetItem(value)
                    if offset not in (self.COL_ALIAS, self.COL_SLICE):
                        item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                    self.source_table.setItem(row, offset, item)
            if self.entries:
                self.source_table.selectRow(0)
                self.expression.setPlainText("A")
        finally:
            self._building_table = False
        self._rebuild_alias_buttons()

    @staticmethod
    def _alias_sequence(count):
        aliases = []
        for index in range(count):
            aliases.append(chr(ord("A") + index) if index < 26 else f"D{index + 1}")
        return aliases

    def _connect_signals(self):
        self.validation_timer = QTimer(self)
        self.validation_timer.setSingleShot(True)
        self.validation_timer.setInterval(140)
        self.validation_timer.timeout.connect(self._validate)
        self.expression.textChanged.connect(self._schedule_validation)
        self.source_table.itemChanged.connect(self._source_changed)
        self.source_table.itemSelectionChanged.connect(self._update_preview)

    def _source_changed(self, item):
        if self._building_table:
            return
        if item.column() in (self.COL_USE, self.COL_ALIAS, self.COL_AXES, self.COL_SLICE):
            self._rebuild_alias_buttons()
            self._schedule_validation()

    def _selected_specs(self):
        specs = []
        for row, entry in enumerate(self.entries):
            if self.source_table.item(row, self.COL_USE).checkState() != Qt.Checked:
                continue
            specs.append(OperandSpec(
                self.source_table.item(row, self.COL_ALIAS).text().strip(),
                entry.source,
                entry.payload_key,
                self.source_table.item(row, self.COL_SLICE).text().strip(),
                self.source_table.item(row, self.COL_AXES).text().strip(),
            ))
        return specs

    def get_plan(self):
        return CalculationPlan(self.expression.toPlainText().strip(), self._selected_specs(), self.result_name.text().strip())

    def _schedule_validation(self):
        if hasattr(self, "validation_timer"):
            self.validation_timer.start()

    def _validate(self):
        self.validation = CalculationEngine.validate(self.get_plan())
        self.trace_table.setRowCount(len(self.validation.steps))
        for row, step in enumerate(self.validation.steps):
            inputs = " + ".join(str(shape) for shape in step.input_shapes) or "-"
            values = [step.expression, inputs, str(step.output_shape), step.dtype, "通过"]
            for column, value in enumerate(values):
                self.trace_table.setItem(row, column, QTableWidgetItem(value))
        if self.validation.valid:
            self.validation_state.setText("表达式有效")
            self.validation_state.setStyleSheet("background:#dcfce7;color:#166534;border:1px solid #86efac;font-weight:600;")
            memory = self._format_bytes(self.validation.estimated_bytes)
            warning = "　".join(self.validation.warnings)
            self.detail_label.setText(f"输出 {self.validation.output_shape}　{self.validation.output_dtype}　预计 {memory}" + (f"\n{warning}" if warning else ""))
            self.summary_label.setText(f"可执行：输出 {self.validation.output_shape}，{self.validation.output_dtype}")
            self.execute_button.setEnabled(True)
        else:
            self.validation_state.setText("表达式无效")
            self.validation_state.setStyleSheet("background:#fee2e2;color:#991b1b;border:1px solid #fca5a5;font-weight:600;")
            self.detail_label.setText(self.validation.error)
            self.summary_label.setText(self.validation.error or "等待有效表达式")
            self.execute_button.setEnabled(False)

    @staticmethod
    def _format_bytes(value):
        size = float(value)
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if size < 1024 or unit == "TB":
                return f"{size:.1f} {unit}"
            size /= 1024

    def _insert_token(self, token):
        cursor = self.expression.textCursor()
        if token.endswith("()"):
            cursor.insertText(token)
            cursor.movePosition(cursor.Left)
            self.expression.setTextCursor(cursor)
        else:
            cursor.insertText(token)
        self.expression.setFocus()

    def _rebuild_alias_buttons(self):
        while self.alias_bar.count() > 1:
            item = self.alias_bar.takeAt(1)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        aliases = [spec.alias for spec in self._selected_specs() if spec.alias]
        for alias in aliases:
            button = QPushButton(alias)
            button.setFixedSize(34, 28)
            button.clicked.connect(lambda _checked=False, value=alias: self._insert_token(value))
            self.alias_bar.addWidget(button)
        self.alias_bar.addStretch(1)
        self.highlighter.set_aliases(aliases)

    def _update_preview(self):
        rows = self.source_table.selectionModel().selectedRows()
        if not rows:
            return
        row = rows[0].row()
        entry = self.entries[row]
        try:
            if entry.payload_key:
                value = (getattr(entry.source, "out_processed", None) or {})[entry.payload_key]
            elif isinstance(entry.source, ProcessedData):
                value = entry.source.__dict__.get("_data_processed_storage")
                if value is None:
                    value = entry.source.data_processed
            elif isinstance(entry.source, Data):
                value = entry.source.__dict__.get("_data_origin_storage")
                if value is None:
                    value = entry.source.data_origin
            else:
                value = CalculationEngine.source_array(OperandSpec("A", entry.source, entry.payload_key))
            if isinstance(value, ArrayRef):
                value = value.load(mmap_mode="r")
            if value.ndim == 3:
                sample = value[0]
                description = f"首帧　{value.shape}　{value.dtype}"
            elif value.ndim == 2:
                sample = value
                description = f"二维数据　{value.shape}　{value.dtype}"
            else:
                self.preview.clear()
                self.preview_label.setText(f"{value.ndim} 维数据仅显示结构信息")
                return
            if np.iscomplexobj(sample):
                sample = np.abs(sample)
            self.preview.setImage(np.asarray(sample), autoLevels=True, autoRange=True)
            self.preview_label.setText(description)
        except Exception as exc:
            self.preview.clear()
            self.preview_label.setText(f"预览不可用：{exc}")

    def _accept_if_valid(self):
        self._validate()
        if not self.validation.valid:
            report_warning(self, "表达式无效", self.validation.error)
            return
        self.accept()
