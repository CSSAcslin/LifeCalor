from __future__ import annotations

from dataclasses import dataclass
import logging

import pyqtgraph as pg
from PyQt5.QtCore import QRegularExpression, Qt, QThreadPool, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QIcon, QSyntaxHighlighter, QTextCharFormat
from PyQt5.QtWidgets import (
    QAbstractItemView, QComboBox, QDialog, QDialogButtonBox, QFormLayout,
    QGridLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
    QPlainTextEdit, QPushButton, QSpinBox, QSplitter, QTabWidget,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from diagnostics import report_exception, report_warning, show_app_error
from widget.DataTreeWidget import DataTreeEntry
from .engine import CalculationEngine
from .metadata import CalculationMetadataPolicy
from .model import CalculationPlan, OperandSpec
from .preview import CalculatorPreviewTask
from .source_picker import CalculatorSourcePicker


@dataclass
class _OperandSlot:
    alias: str
    source: object
    payload_key: str | None
    label: str
    kind: str
    shape: tuple[int, ...]
    dtype: str
    axes: str
    slice_text: str = ""

    @property
    def identity(self):
        return (
            self.source.__class__.__name__,
            getattr(self.source, "timestamp", None),
            getattr(self.source, "serial_number", id(self.source)),
            self.payload_key,
        )


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
            iterator = QRegularExpression(rf"\b{name}(?=\s*\()").globalMatch(text)
            while iterator.hasNext():
                match = iterator.next()
                self.setFormat(match.capturedStart(), match.capturedLength(), self.function_format)
        for alias in self.aliases:
            iterator = QRegularExpression(rf"\b{QRegularExpression.escape(alias)}\b").globalMatch(text)
            while iterator.hasNext():
                match = iterator.next()
                self.setFormat(match.capturedStart(), match.capturedLength(), self.alias_format)
        iterator = QRegularExpression(r"(?<![A-Za-z_])(?:\d+\.?\d*|\.\d+)").globalMatch(text)
        while iterator.hasNext():
            match = iterator.next()
            self.setFormat(match.capturedStart(), match.capturedLength(), self.number_format)


class DataCalculatorDialog(QDialog):
    execute_requested = pyqtSignal(object)
    closed = pyqtSignal()

    COL_ALIAS, COL_NAME, COL_SLICE = range(3)

    HELP = {
        "+": "逐元素相加；Shape 必须相同或可以广播",
        "-": "逐元素相减；例如 A - B[42, :, :]",
        "*": "逐元素相乘，不执行矩阵乘法",
        "/": "逐元素相除，请注意除零值",
        "**": "幂运算，例如 A ** 2",
        ">": "逐元素大于比较，可作为 where 的条件",
        ">=": "逐元素大于等于比较",
        "<": "逐元素小于比较",
        "<=": "逐元素小于等于比较",
        "==": "逐元素相等比较",
        "!=": "逐元素不等比较",
        "mean()": "均值，例如 mean(A, axis=0)",
        "max()": "最大值，例如 max(A, axis=(1, 2))",
        "min()": "最小值，例如 min(A, axis=0)",
        "sum()": "求和，例如 sum(A, axis=0)",
        "std()": "标准差，例如 std(A, axis=0)",
        "abs()": "绝对值；complex 数据返回幅值",
        "sqrt()": "逐元素平方根",
        "log()": "逐元素自然对数",
        "clip()": "限制范围，例如 clip(A, 0, 1)",
        "where()": "条件选择，例如 where(A > 0, A, 0)",
        "transpose()": "调整轴顺序，例如 transpose(A, axes=(1, 2, 0))",
        "[": "开始输入切片或索引",
        "]": "结束切片或索引",
        ":": "完整维度或切片范围，例如 A[10, :, :]",
        "⌫": "删除光标前一个字符（Backspace）",
        "C": "清空整个表达式",
    }

    def __init__(
        self, initial_sources=None, parent=None, source_provider=None,
        import_callback=None, metadata_defaults=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("多数据运算工作台")
        self.resize(1280, 820)
        self.setMinimumSize(1040, 680)
        self.source_provider = source_provider or (lambda: list(initial_sources or []))
        self.import_callback = import_callback
        self.metadata_defaults = dict(metadata_defaults or {})
        self.slots: list[_OperandSlot] = []
        self.validation = None
        self._building_slots = False
        self._updating_metadata = False
        self._preview_array = None
        self._preview_request_id = 0
        self._preview_tasks = {}
        self._preview_labels = {}
        self._latest_result = None
        self._preview_mode = "input"
        self._execution_running = False
        self._awaiting_import = False
        self._closing = False
        self.thread_pool = QThreadPool.globalInstance()
        self._setup_ui()
        self._connect_signals()
        for source in list(initial_sources or []):
            self.add_source(source)
        self._schedule_validation()

    @staticmethod
    def _default_axes(shape):
        return {1: "T", 2: "HW", 3: "THW", 4: "THWC"}.get(len(shape), "?" * len(shape))

    @classmethod
    def _slot_from_source(cls, source, payload_key=None, label=None):
        info = CalculationEngine.source_info(OperandSpec("A", source, payload_key))
        metadata = CalculationMetadataPolicy.source_mapping(source)
        axes = str(
            metadata.get("scientific_axes")
            or metadata.get("source_axes")
            or cls._default_axes(info.shape)
        ).replace("Y", "H").replace("X", "W")
        kind = source.__class__.__name__ if payload_key is None else f"out_processed · {payload_key}"
        return _OperandSlot(
            "", source, payload_key, label or getattr(source, "name", "未命名数据"),
            kind, info.shape, str(info.dtype), axes,
        )

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)
        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.setChildrenCollapsible(False)
        self.main_splitter.addWidget(self._build_operand_panel())
        self.main_splitter.addWidget(self._build_workspace())
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setSizes([400, 860])
        root.addWidget(self.main_splitter, 1)

        footer = QHBoxLayout()
        self.summary_label = QLabel("请添加数据并输入表达式")
        footer.addWidget(self.summary_label, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.Cancel | QDialogButtonBox.Ok)
        self.execute_button = buttons.button(QDialogButtonBox.Ok)
        self.execute_button.setText("执行运算")
        self.execute_button.setObjectName("StressButton")
        self.execute_button.setToolTip("在后台任务线程中执行并生成 ProcessedData")
        buttons.button(QDialogButtonBox.Cancel).setText("关闭")
        buttons.accepted.connect(self._request_execution)
        buttons.rejected.connect(self.close)
        footer.addWidget(buttons)
        root.addLayout(footer)

    def _build_operand_panel(self):
        panel = QGroupBox("本次运算数据")
        panel.setMinimumWidth(340)
        panel.setMaximumWidth(460)
        layout = QVBoxLayout(panel)
        actions = QHBoxLayout()
        self.add_button = QPushButton(QIcon(':icons/icon_add.svg'), "添加数据")
        self.add_button.setToolTip("从历史树选择 Data、ProcessedData 或 out_processed 数组，依次分配 A/B/C…")
        self.add_button.clicked.connect(self._open_source_picker)
        actions.addWidget(self.add_button)
        self.import_button = QPushButton("从文件导入")
        self.import_button.setToolTip("复用主界面导入设置和后台任务；完成后自动加入下一个数据槽")
        self.import_button.setEnabled(self.import_callback is not None)
        self.import_button.clicked.connect(self._request_file_import)
        actions.addWidget(self.import_button)
        layout.addLayout(actions)

        self.slot_table = QTableWidget(0, 3)
        self.slot_table.setHorizontalHeaderLabels(["别名", "数据信息", "预切片"])
        self.slot_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.slot_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.slot_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.slot_table.setAlternatingRowColors(True)
        self.slot_table.verticalHeader().hide()
        header = self.slot_table.horizontalHeader()
        header.setSectionResizeMode(self.COL_ALIAS, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(self.COL_NAME, QHeaderView.Stretch)
        header.setSectionResizeMode(self.COL_SLICE, QHeaderView.ResizeToContents)
        layout.addWidget(self.slot_table, 1)

        row = QHBoxLayout()
        self.replace_button = QPushButton("替换")
        self.replace_button.setToolTip("替换选中槽的数据并保留别名，避免表达式被自动改写")
        self.replace_button.clicked.connect(lambda: self._open_source_picker(replace=True))
        row.addWidget(self.replace_button)
        self.remove_button = QPushButton("移除")
        self.remove_button.setToolTip("移除选中数据槽；其他槽的别名不变")
        self.remove_button.clicked.connect(self._remove_selected_slot)
        row.addWidget(self.remove_button)
        self.clear_slots_button = QPushButton("清空")
        self.clear_slots_button.setToolTip("清空本次运算槽，不删除历史或缓存")
        self.clear_slots_button.clicked.connect(self._clear_slots)
        row.addWidget(self.clear_slots_button)
        layout.addLayout(row)
        return panel

    def _build_workspace(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._build_expression_panel(), 0)
        layout.addWidget(self._build_tabs(), 1)
        return widget

    def _build_expression_panel(self):
        panel = QGroupBox("表达式")
        layout = QVBoxLayout(panel)
        self.expression = QPlainTextEdit()
        self.expression.setMaximumHeight(72)
        self.expression.setPlaceholderText("例如：A - B[42, :, :]　或　mean(A, axis=0)")
        self.expression.setToolTip("只支持提示中列出的数组运算、切片和函数")
        font = QFont("Consolas")
        font.setPointSize(12)
        self.expression.setFont(font)
        self.highlighter = _ExpressionHighlighter(self.expression.document())
        layout.addWidget(self.expression)

        self.alias_row = QHBoxLayout()
        self.alias_row.addWidget(QLabel("数据"))
        self.alias_row.addStretch(1)
        layout.addLayout(self.alias_row)

        rows = [
            ["7", "8", "9", "+", "-", "(", ")", "[", "]", "⌫"],
            ["4", "5", "6", "*", "/", ">", ">=", "<", "<=", "C"],
            ["1", "2", "3", "**", ":", "==", "!=", "mean()", "where()"],
            ["0", ".", ",", "max()", "min()", "sum()", "std()", "abs()", "transpose()"],
            ["sqrt()", "log()", "clip()"],
        ]
        grid = QGridLayout()
        for row_index, labels in enumerate(rows):
            for column, label in enumerate(labels):
                button = QPushButton(label)
                button.setFixedHeight(29)
                button.setToolTip(self.HELP.get(label, f"向表达式插入 {label}"))
                button.clicked.connect(lambda _checked=False, value=label: self._key_pressed(value))
                grid.addWidget(button, row_index, column)
        layout.addLayout(grid)

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("结果名称"))
        self.result_name = QLineEdit()
        self.result_name.setPlaceholderText("可选；留空时使用主数据名@math")
        self.result_name.setToolTip("设置生成的 ProcessedData 名称，不影响表达式")
        name_row.addWidget(self.result_name, 1)
        layout.addLayout(name_row)
        return panel

    def _build_tabs(self):
        self.inspection_tabs = QTabWidget()
        self.preview_tab = self._build_preview_tab()
        self.validation_tab = self._build_validation_tab()
        self.metadata_tab = self._build_metadata_tab()
        self.inspection_tabs.addTab(self.preview_tab, "数据预览")
        self.inspection_tabs.addTab(self.validation_tab, "验证路径")
        self.inspection_tabs.addTab(self.metadata_tab, "结果参数")
        return self.inspection_tabs

    def _build_preview_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        self.preview = pg.ImageView()
        self.preview.imageItem.setOpts(axisOrder="row-major")
        self.preview.ui.roiBtn.hide()
        self.preview.ui.menuBtn.hide()
        self.preview.ui.histogram.setMinimumWidth(110)
        layout.addWidget(self.preview, 1)
        row = QHBoxLayout()
        row.addWidget(QLabel("帧号"))
        self.frame_input = QSpinBox()
        self.frame_input.setRange(0, 0)
        self.frame_input.setToolTip("输入从 0 开始的帧索引；点击读取后直接访问当前数组或 mmap")
        row.addWidget(self.frame_input)
        self.read_frame_button = QPushButton("读取帧")
        self.read_frame_button.setToolTip("按帧号即时读取，不建立额外帧缓存")
        self.read_frame_button.clicked.connect(self._request_preview)
        row.addWidget(self.read_frame_button)
        self.preview_label = QLabel("请选择左侧数据槽")
        self.preview_label.setToolTip("显示帧号、Shape、dtype 和数值范围")
        row.addWidget(self.preview_label, 1)
        layout.addLayout(row)
        return widget

    def _build_validation_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        self.validation_state = QLabel("等待输入")
        self.validation_state.setMinimumHeight(32)
        self.validation_state.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.validation_state)
        self.trace_table = QTableWidget(0, 5)
        self.trace_table.setHorizontalHeaderLabels(["步骤", "输入 Shape", "输出 Shape", "Dtype", "状态"])
        self.trace_table.verticalHeader().hide()
        self.trace_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.trace_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        header = self.trace_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        for column in range(1, 5):
            header.setSectionResizeMode(column, QHeaderView.ResizeToContents)
        layout.addWidget(self.trace_table, 1)
        self.detail_label = QLabel()
        self.detail_label.setWordWrap(True)
        layout.addWidget(self.detail_label)
        return widget

    def _build_metadata_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        form = QFormLayout()
        self.metadata_source = QComboBox()
        self.metadata_source.setToolTip("选择基础参数继承来源，默认使用 A")
        form.addRow("继承来源", self.metadata_source)
        self.metadata_fields = {}
        labels = {
            "fps": "FPS", "time_step": "时间步长", "time_unit": "时间单位",
            "space_step": "空间步长", "space_unit": "空间单位",
            "value_unit": "数值单位", "scientific_axes": "结果轴",
        }
        for key, label in labels.items():
            editor = QLineEdit()
            editor.setToolTip(f"结果的{label}；默认继承，输入内容可覆盖")
            if key == "scientific_axes":
                editor.setToolTip("根据结果维度初始化为 T、HW、THW 或 THWC；transpose 等特殊运算后可手动修正")
            self.metadata_fields[key] = editor
            form.addRow(label, editor)
        layout.addLayout(form)
        note = QLabel("大型 out_processed 数组不会被复制；结果只保存来源、key、Shape 和切片记录。")
        note.setWordWrap(True)
        note.setObjectName("mutedLabel")
        layout.addWidget(note)
        layout.addStretch(1)
        return widget

    def _connect_signals(self):
        self.validation_timer = QTimer(self)
        self.validation_timer.setSingleShot(True)
        self.validation_timer.setInterval(140)
        self.validation_timer.timeout.connect(self._validate)
        self.expression.textChanged.connect(self._schedule_validation)
        self.slot_table.itemChanged.connect(self._slot_item_changed)
        self.slot_table.itemSelectionChanged.connect(self._slot_selection_changed)
        self.metadata_source.currentIndexChanged.connect(self._metadata_source_changed)

    def _next_alias(self):
        used = {slot.alias for slot in self.slots}
        for index in range(26):
            alias = chr(ord("A") + index)
            if alias not in used:
                return alias
        index = 1
        while f"D{index}" in used:
            index += 1
        return f"D{index}"

    def add_source(self, source, payload_key=None, label=None, replace_row=None):
        try:
            return self._add_source_impl(source, payload_key, label, replace_row)
        except Exception as exc:
            self.summary_label.setText(f"加入数据失败：{exc}")
            report_exception(
                self, "加入运算数据失败", str(exc), exc,
                stage="计算器加入数据", data=source, severity="error",
            )
            return False

    def _add_source_impl(self, source, payload_key=None, label=None, replace_row=None):
        if isinstance(source, DataTreeEntry):
            payload_key, label, source = source.payload_key, source.label, source.source
        slot = self._slot_from_source(source, payload_key, label)
        if replace_row is None and any(existing.identity == slot.identity for existing in self.slots):
            self.summary_label.setText("该数据已在本次运算中")
            return False
        if replace_row is None:
            slot.alias = self._next_alias()
            self.slots.append(slot)
            selected = len(self.slots) - 1
        else:
            slot.alias = self.slots[replace_row].alias
            self.slots[replace_row] = slot
            selected = replace_row
        self._refresh_slot_table(selected)
        if not self.expression.toPlainText().strip():
            self.expression.setPlainText(slot.alias)
        self.summary_label.setText(f"已加入 {slot.alias}: {slot.label}")
        return True

    def _refresh_slot_table(self, selected=None):
        self._building_slots = True
        try:
            self.slot_table.setRowCount(len(self.slots))
            for row, slot in enumerate(self.slots):
                values = [
                    slot.alias,
                    f"{slot.label}\n{slot.kind}\n{slot.shape} | {slot.axes} | {slot.dtype}",
                    slot.slice_text,
                ]
                self.slot_table.setRowHeight(row, 62)
                tooltip = (
                    f"别名: {slot.alias}\n名称: {slot.label}\n类型: {slot.kind}\n"
                    f"Shape: {slot.shape}\nDtype: {slot.dtype}\n轴: {slot.axes}\n"
                    f"预切片: {slot.slice_text or '无'}"
                )
                for column, value in enumerate(values):
                    item = QTableWidgetItem(str(value))
                    item.setToolTip(tooltip)
                    if column != self.COL_SLICE:
                        item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                    self.slot_table.setItem(row, column, item)
            if selected is not None and 0 <= selected < len(self.slots):
                self.slot_table.selectRow(selected)
        finally:
            self._building_slots = False
        if selected is not None and 0 <= selected < len(self.slots):
            self._slot_selection_changed()
        self._rebuild_alias_buttons()
        self._refresh_metadata_sources()
        self._schedule_validation()

    def _open_source_picker(self, replace=False):
        row = self.slot_table.currentRow() if replace else None
        if replace and row < 0:
            report_warning(self, "替换数据", "请先选中要替换的数据槽")
            return
        picker = CalculatorSourcePicker(self.source_provider, self)
        if picker.exec_() and picker.selected_entry is not None:
            self.add_source(picker.selected_entry, replace_row=row)

    def _request_file_import(self):
        if self.import_callback is not None and self.import_callback():
            self._awaiting_import = True
            self.summary_label.setText("文件正在后台导入，完成后将自动加入数据槽")

    def handle_import_finished(self, source):
        if not self._awaiting_import:
            return False
        self._awaiting_import = False
        return self.add_source(source)

    def _remove_selected_slot(self):
        row = self.slot_table.currentRow()
        if row >= 0:
            removed = self.slots.pop(row)
            self._refresh_slot_table(min(row, len(self.slots) - 1))
            self.summary_label.setText(f"已移除 {removed.alias}: {removed.label}")

    def _clear_slots(self):
        self.slots.clear()
        self._refresh_slot_table()
        self._preview_request_id += 1
        self._preview_array = None
        self.preview.clear()
        self.preview_label.setText("请选择左侧数据槽")

    def _slot_item_changed(self, item):
        if not self._building_slots and item.column() == self.COL_SLICE:
            self.slots[item.row()].slice_text = item.text().strip()
            self._schedule_validation()

    def _slot_selection_changed(self):
        if self._building_slots:
            return
        row = self.slot_table.currentRow()
        if row < 0 or row >= len(self.slots):
            return
        slot = self.slots[row]
        self._preview_request_id += 1
        self._preview_mode = "input"
        self._preview_array = None
        self.preview.clear()
        temporal = len(slot.shape) == 3 and slot.axes.startswith("T")
        self.frame_input.setEnabled(temporal)
        self.frame_input.setRange(0, max(0, slot.shape[0] - 1) if temporal else 0)
        self.read_frame_button.setEnabled(temporal or len(slot.shape) == 2)
        self.preview_label.setText(f"输入数据 · {slot.alias} · 输入帧号后点击“读取帧”")

    def _request_preview(self):
        if self._preview_mode == "result" and self._latest_result is not None:
            source = self._latest_result
            payload_key = None
            slice_text = ""
            shape = tuple(getattr(source, "datashape", ()) or ())
            metadata = CalculationMetadataPolicy.source_mapping(source)
            axes = str(
                metadata.get("scientific_axes")
                or metadata.get("source_axes")
                or self._default_axes(shape)
            ).replace("Y", "H").replace("X", "W")
            context_label = f"运算结果 · {getattr(source, 'name', '未命名结果')}"
        else:
            row = self.slot_table.currentRow()
            if row < 0 or row >= len(self.slots):
                return
            slot = self.slots[row]
            source = slot.source
            payload_key = slot.payload_key
            slice_text = slot.slice_text
            axes = slot.axes
            context_label = f"输入数据 · {slot.alias} · {slot.label}"

        self._preview_request_id += 1
        request_id = self._preview_request_id
        task = CalculatorPreviewTask(
            request_id, source, payload_key, slice_text,
            axes, self.frame_input.value(),
        )
        task.signals.completed.connect(self._preview_completed)
        task.signals.failed.connect(self._preview_failed)
        self._preview_tasks[request_id] = task
        self._preview_labels[request_id] = context_label
        self.read_frame_button.setEnabled(False)
        self.preview_label.setText(f"{context_label} · 正在后台读取预览帧...")
        self.thread_pool.start(task)

    def _preview_completed(self, request_id, result):
        self._release_preview_task_later(request_id)
        if request_id != self._preview_request_id:
            return
        try:
            self._preview_array = result.image
            self.preview.setImage(self._preview_array, autoLevels=True, autoRange=True)
            context_label = self._preview_labels.get(request_id, "数据")
            self.preview_label.setText(
                f"{context_label} · {result.description} · {result.image.shape} · "
                f"{result.image.dtype} · {result.minimum:.4g} ~ {result.maximum:.4g}"
            )
        except Exception as exc:
            self._preview_array = None
            self.preview.clear()
            self.preview_label.setText(f"预览显示失败：{exc}")
            report_exception(
                self, "预览显示失败", str(exc), exc,
                stage="计算器预览显示", severity="error",
            )
        finally:
            self.read_frame_button.setEnabled(True)

    def _preview_failed(self, request_id, error):
        self._release_preview_task_later(request_id)
        if request_id != self._preview_request_id:
            return
        self._preview_array = None
        self.preview.clear()
        context_label = self._preview_labels.get(request_id, "数据")
        self.preview_label.setText(f"{context_label} · 预览失败：{error.message}")
        self.read_frame_button.setEnabled(True)
        show_app_error(self, error)

    def _release_preview_task_later(self, request_id):
        QTimer.singleShot(0, lambda rid=request_id: self._release_preview_task(rid))

    def _release_preview_task(self, request_id):
        self._preview_tasks.pop(request_id, None)
        self._preview_labels.pop(request_id, None)
        if self._closing and not self._preview_tasks:
            self.deleteLater()

    def _selected_specs(self):
        return [OperandSpec(s.alias, s.source, s.payload_key, s.slice_text, s.axes) for s in self.slots]

    def _metadata_overrides(self):
        result = {}
        for key, editor in self.metadata_fields.items():
            if not editor.isModified():
                continue
            value = editor.text().strip()
            if not value:
                continue
            if key in {"fps", "time_step", "space_step"}:
                try:
                    value = float(value)
                except ValueError:
                    pass
            result[key] = value
        return result

    def get_plan(self):
        return CalculationPlan(
            self.expression.toPlainText().strip(), self._selected_specs(),
            self.result_name.text().strip(), self.metadata_source.currentData() or "",
            self._metadata_overrides(), dict(self.metadata_defaults),
        )

    def _schedule_validation(self):
        if hasattr(self, "validation_timer"):
            self.validation_timer.start()

    def _validate(self):
        self.validation = CalculationEngine.validate(self.get_plan())
        self.validation.warnings.extend(CalculationMetadataPolicy.warnings(self.get_plan()))
        self.trace_table.setRowCount(len(self.validation.steps))
        failed_row = None
        for row, step in enumerate(self.validation.steps):
            inputs = " + ".join(str(shape) for shape in step.input_shapes) or "-"
            state = "失败" if step.status == "invalid" else "警告" if step.status == "warning" else "通过"
            values = [step.expression, inputs, str(step.output_shape) if step.output_shape else "-", step.dtype or "-", state]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setToolTip(step.message or value)
                if step.status == "invalid":
                    item.setBackground(QColor("#fee2e2"))
                    item.setForeground(QColor("#991b1b"))
                    failed_row = row if failed_row is None else failed_row
                elif step.status == "warning":
                    item.setBackground(QColor("#fef3c7"))
                self.trace_table.setItem(row, column, item)

        if self.validation.valid:
            self.validation_state.setText("表达式有效")
            self.validation_state.setStyleSheet("background:#dcfce7;color:#166534;border:1px solid #86efac;font-weight:600;")
            memory = self._format_bytes(self.validation.estimated_bytes)
            self.detail_label.setText(
                f"输出 {self.validation.output_shape}　{self.validation.output_dtype}　预计 {memory}"
                + (f"\n{'　'.join(self.validation.warnings)}" if self.validation.warnings else "")
            )
            self.summary_label.setText(f"可执行：输出 {self.validation.output_shape}，{self.validation.output_dtype}")
            self.execute_button.setEnabled(not self._execution_running)
        else:
            self.validation_state.setText("表达式无效")
            self.validation_state.setStyleSheet("background:#fee2e2;color:#991b1b;border:1px solid #fca5a5;font-weight:600;")
            self.detail_label.setText(self.validation.error)
            self.summary_label.setText(self.validation.error or "等待有效表达式")
            self.execute_button.setEnabled(False)
            if failed_row is not None:
                self.trace_table.selectRow(failed_row)
                self.trace_table.scrollToItem(self.trace_table.item(failed_row, 0))
        self._refresh_metadata_fields(self.validation.output_shape if self.validation.valid else ())

    @staticmethod
    def _format_bytes(value):
        size = float(value)
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if size < 1024 or unit == "TB":
                return f"{size:.1f} {unit}"
            size /= 1024

    def _key_pressed(self, token):
        if token == "⌫":
            cursor = self.expression.textCursor()
            cursor.deletePreviousChar()
            self.expression.setTextCursor(cursor)
            return
        if token == "C":
            self.expression.clear()
            return
        cursor = self.expression.textCursor()
        cursor.insertText(token)
        if token.endswith("()"):
            cursor.movePosition(cursor.Left)
            self.expression.setTextCursor(cursor)
        self.expression.setFocus()

    def _rebuild_alias_buttons(self):
        while self.alias_row.count() > 1:
            item = self.alias_row.takeAt(1)
            if item.widget() is not None:
                item.widget().deleteLater()
        for slot in self.slots:
            button = QPushButton(slot.alias)
            button.setFixedSize(34, 28)
            button.setToolTip(f"插入 {slot.alias}：{slot.label}\nShape: {slot.shape} · {slot.axes}")
            button.clicked.connect(lambda _checked=False, value=slot.alias: self._key_pressed(value))
            self.alias_row.addWidget(button)
        self.alias_row.addStretch(1)
        self.highlighter.set_aliases(slot.alias for slot in self.slots)

    def _refresh_metadata_sources(self):
        current = self.metadata_source.currentData()
        self._updating_metadata = True
        self.metadata_source.clear()
        for slot in self.slots:
            self.metadata_source.addItem(f"{slot.alias} · {slot.label}", slot.alias)
        index = self.metadata_source.findData(current)
        if index >= 0:
            self.metadata_source.setCurrentIndex(index)
        self._updating_metadata = False
        self._refresh_metadata_fields(self.validation.output_shape if self.validation and self.validation.valid else ())

    def _metadata_source_changed(self):
        if not self._updating_metadata:
            self._refresh_metadata_fields(self.validation.output_shape if self.validation and self.validation.valid else ())

    def _refresh_metadata_fields(self, output_shape):
        if self._updating_metadata:
            return
        self._updating_metadata = True
        try:
            plan = CalculationPlan(
                self.expression.toPlainText().strip(), self._selected_specs(),
                self.result_name.text().strip(), self.metadata_source.currentData() or "",
                {}, dict(self.metadata_defaults),
            )
            values = CalculationMetadataPolicy.preview(plan, output_shape)
            for key, editor in self.metadata_fields.items():
                if editor.isModified():
                    continue
                editor.setText("" if values.get(key) is None else str(values.get(key, "")))
                editor.setModified(False)
        finally:
            self._updating_metadata = False

    def _request_execution(self):
        if self._execution_running:
            return
        self._validate()
        if not self.validation.valid:
            logging.warning(
                "计算器输入验证失败: expression=%s error=%s",
                self.expression.toPlainText().strip(), self.validation.error,
            )
            self.inspection_tabs.setCurrentWidget(self.validation_tab)
            return
        self._execution_running = True
        self.execute_button.setEnabled(False)
        self.summary_label.setText("运算任务已提交，主窗口和计算器可继续使用")
        self.execute_requested.emit(self.get_plan())

    def set_execution_finished(self, result=None):
        self._execution_running = False
        self.execute_button.setEnabled(bool(self.validation and self.validation.valid))
        name = getattr(result, "name", "运算结果")
        self.summary_label.setText(f"运算完成：{name}")
        if result is None:
            return
        self._latest_result = result
        self._preview_mode = "result"
        shape = tuple(getattr(result, "datashape", ()) or ())
        metadata = CalculationMetadataPolicy.source_mapping(result)
        axes = str(
            metadata.get("scientific_axes")
            or metadata.get("source_axes")
            or self._default_axes(shape)
        ).replace("Y", "H").replace("X", "W")
        temporal = len(shape) == 3 and axes.startswith("T")
        self.frame_input.setEnabled(temporal)
        self.frame_input.setRange(0, max(0, shape[0] - 1) if temporal else 0)
        self.frame_input.setValue(0)
        can_preview = temporal or len(shape) == 2
        self.read_frame_button.setEnabled(can_preview)
        self.inspection_tabs.setCurrentWidget(self.preview_tab)
        self._preview_array = None
        self.preview.clear()
        if can_preview:
            self._request_preview()
        else:
            self.preview_label.setText(f"运算结果 · {name} · 当前维度不提供图像预览")

    def set_execution_failed(self, error):
        self._execution_running = False
        self.execute_button.setEnabled(bool(self.validation and self.validation.valid))
        self.summary_label.setText(f"运算失败：{error.message}")

    def closeEvent(self, event):
        self._preview_request_id += 1
        if not self._closing:
            self._closing = True
            self.closed.emit()
        super().closeEvent(event)
        if not self._preview_tasks:
            QTimer.singleShot(0, self.deleteLater)
