from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import QObject, QRunnable, QThreadPool, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFormLayout, QHBoxLayout,
    QLabel, QPushButton, QSpinBox, QSplitter, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

from calculator.engine import CalculationEngine
from calculator.model import OperandSpec
from calculator.source_picker import CalculatorSourcePicker
from history.annotations import display_name_for
from tasks.model import CancellationToken, TaskStatus
from .engine import ConvolutionPlan, preview_convolution, validate_convolution


class _PreviewSignals(QObject):
    completed = pyqtSignal(int, object)
    failed = pyqtSignal(int, str)


class _PreviewTask(QRunnable):
    def __init__(self, request_id, owner_key, plan, frame, row, column):
        super().__init__()
        self.setAutoDelete(False)
        self.request_id = request_id
        self.owner_key = owner_key
        self.token = CancellationToken()
        self.plan = plan
        self.position = (frame, row, column)
        self.signals = _PreviewSignals()

    def run(self):
        try:
            self.token.raise_if_cancelled()
            value = preview_convolution(self.plan, *self.position)
            self.token.raise_if_cancelled()
            self.signals.completed.emit(self.request_id, value)
        except Exception as exc:
            self.signals.failed.emit(self.request_id, str(exc))


class ConvolutionPreviewPool(QThreadPool):
    """Own preview jobs beyond the lifetime of an individual dialog."""
    completed = pyqtSignal(int, object)
    failed = pyqtSignal(int, str)

    def __init__(self, parent):
        super().__init__(parent)
        self.setMaxThreadCount(1)
        self.tasks = {}
        self._next_id = 0
        self._closing = False

    def submit(self, owner_key, plan, frame, row, column):
        if self._closing:
            raise ValueError("应用正在关闭，无法提交预览")
        self._next_id += 1
        task = _PreviewTask(self._next_id, owner_key, plan, frame, row, column)
        self.tasks[task.request_id] = task
        task.signals.completed.connect(self._completed)
        task.signals.failed.connect(self._failed)
        self.start(task)
        return task.request_id

    def _completed(self, request_id, value):
        self.tasks.pop(request_id, None)
        self.completed.emit(request_id, value)

    def _failed(self, request_id, message):
        self.tasks.pop(request_id, None)
        self.failed.emit(request_id, message)

    def cancel_owner(self, owner_key):
        for task in self.tasks.values():
            if task.owner_key is owner_key:
                task.token.cancel()

    def prepare_shutdown(self):
        self._closing = True
        for task in self.tasks.values():
            task.token.cancel()

    def shutdown_pending(self):
        return bool(self.tasks) or self.activeThreadCount() > 0


class ConvolutionDialog(QDialog):
    execute_requested = pyqtSignal(object)
    cancel_requested = pyqtSignal()
    closed = pyqtSignal()

    def __init__(self, source_provider, parent=None):
        super().__init__(parent)
        self.source_provider = source_provider
        self.source = None
        self.payload_key = None
        self.task_id = None
        self._kernel = np.zeros((1, 3, 3), dtype=np.float64)
        self._layer = 0
        self._preview_id = 0
        owner = parent if parent is not None else QApplication.instance()
        pool = getattr(owner, "_convolution_preview_pool", None)
        if pool is None:
            pool = ConvolutionPreviewPool(owner)
            owner._convolution_preview_pool = pool
        self.preview_pool = pool
        self._preview_owner = object()
        self._preview_tasks = pool.tasks
        pool.completed.connect(self._preview_completed)
        pool.failed.connect(self._preview_failed)
        self._closing = False
        self._connections_closed = False
        self.setWindowTitle("时空卷积")
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.resize(1080, 720)
        self._setup_ui()
        self._generate_kernel()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        source_row = QHBoxLayout()
        self.source_label = QLabel("请选择二维 HW 或三维 THW 数据")
        source_row.addWidget(self.source_label, 1)
        choose = QPushButton("选择数据")
        choose.clicked.connect(self._choose_source)
        source_row.addWidget(choose)
        layout.addLayout(source_row)

        splitter = QSplitter(Qt.Horizontal)
        controls = QWidget()
        form = QFormLayout(controls)
        self.axes = QComboBox()
        self.axes.addItems(["HW", "THW"])
        self.axes.currentTextChanged.connect(self._axes_changed)
        form.addRow("数据轴", self.axes)
        self.preset = QComboBox()
        self.preset.addItems(["单位核", "均值核", "W 方向差分", "自定义"])
        form.addRow("预设", self.preset)
        size_row = QHBoxLayout()
        self.sizes = []
        self.origins = []
        for name in ("T", "H", "W"):
            size_row.addWidget(QLabel(name))
            spin = QSpinBox()
            spin.setRange(1, 15)
            spin.setValue(1 if name == "T" else 3)
            self.sizes.append(spin)
            size_row.addWidget(spin)
        self.sizes[0].setEnabled(False)
        form.addRow("核尺寸", size_row)
        generate = QPushButton("生成卷积核")
        generate.clicked.connect(self._generate_kernel)
        form.addRow(generate)
        self.normalize = QCheckBox("按权重和归一化")
        form.addRow(self.normalize)
        origin_row = QHBoxLayout()
        for name in ("T", "H", "W"):
            origin_row.addWidget(QLabel(name))
            spin = QSpinBox()
            self.origins.append(spin)
            origin_row.addWidget(spin)
        form.addRow("中心偏移", origin_row)
        self.boundary = QComboBox()
        for label, mode in (("对称 reflect", "reflect"), ("镜像 mirror", "mirror"),
                            ("最近值", "nearest"), ("周期 wrap", "wrap"), ("常值", "constant")):
            self.boundary.addItem(label, mode)
        form.addRow("全局边界", self.boundary)
        self.cval = QDoubleSpinBox()
        self.cval.setRange(-1e12, 1e12)
        self.cval.setDecimals(6)
        self.cval.setEnabled(False)
        self.boundary.currentIndexChanged.connect(
            lambda: self.cval.setEnabled(self.boundary.currentData() == "constant")
        )
        form.addRow("边界常值", self.cval)
        self.frame = QSpinBox()
        self.row = QSpinBox()
        self.column = QSpinBox()
        form.addRow("预览帧（从 0 起）", self.frame)
        form.addRow("ROI 起始行", self.row)
        form.addRow("ROI 起始列", self.column)
        form.addRow(QLabel("预览最大 48 × 48，使用完整卷积邻域。"))
        self.preview_button = QPushButton("读取并预览 ROI")
        self.preview_button.clicked.connect(self._request_preview)
        form.addRow(self.preview_button)
        backend = QLabel("计算后端 CPU")
        backend.setObjectName("mutedLabel")
        form.addRow(backend)
        splitter.addWidget(controls)

        editor = QWidget()
        editor_layout = QVBoxLayout(editor)
        layer_row = QHBoxLayout()
        layer_row.addWidget(QLabel("T 层（从 0 起）"))
        self.layer_selector = QSpinBox()
        self.layer_selector.valueChanged.connect(self._select_layer)
        layer_row.addWidget(self.layer_selector)
        layer_row.addStretch()
        editor_layout.addLayout(layer_row)
        self.weights_table = QTableWidget()
        self.weights_table.horizontalHeader().setDefaultSectionSize(68)
        editor_layout.addWidget(self.weights_table, 1)
        images = QHBoxLayout()
        self.original_preview = pg.ImageView()
        self.filtered_preview = pg.ImageView()
        for title, view in (("源 ROI", self.original_preview), ("卷积 ROI", self.filtered_preview)):
            column = QVBoxLayout()
            column.addWidget(QLabel(title))
            view.imageItem.setOpts(axisOrder="row-major")
            view.ui.roiBtn.hide()
            view.ui.menuBtn.hide()
            view.setMinimumHeight(180)
            column.addWidget(view)
            images.addLayout(column)
        editor_layout.addLayout(images, 1)
        splitter.addWidget(editor)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter, 1)

        self.status = QLabel("编辑各层权重后可先预览，再对完整数据执行。")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        buttons = QHBoxLayout()
        buttons.addStretch()
        self.run_button = QPushButton("执行卷积")
        self.run_button.clicked.connect(self._execute)
        self.cancel_button = QPushButton("取消任务")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_requested)
        close = QPushButton("关闭")
        close.clicked.connect(self.close)
        for button in (self.run_button, self.cancel_button, close):
            buttons.addWidget(button)
        layout.addLayout(buttons)

    def _choose_source(self):
        picker = CalculatorSourcePicker(self.source_provider, self)
        picker.setWindowTitle("选择卷积数据")
        if picker.exec_() == QDialog.Accepted:
            entry = picker.selected_entry
            self.set_source(entry.source, entry.payload_key)

    def set_source(self, source, payload_key=None):
        info = CalculationEngine.source_info(OperandSpec("A", source, payload_key))
        if len(info.shape) not in (2, 3) or not all(info.shape):
            self.status.setText("请选择二维 HW 或三维 THW 数值数组")
            return False
        self.source = source
        self.payload_key = payload_key
        self.axes.setCurrentText("HW" if len(info.shape) == 2 else "THW")
        self.frame.setRange(0, info.shape[0] - 1 if len(info.shape) == 3 else 0)
        self.row.setRange(0, info.shape[-2] - 1)
        self.column.setRange(0, info.shape[-1] - 1)
        self.source_label.setText(f"{display_name_for(source)}  {info.shape}  {info.dtype}")
        self._preview_id = -1
        self.preview_pool.cancel_owner(self._preview_owner)
        self.preview_button.setEnabled(True)
        return True

    def _axes_changed(self):
        temporal = self.axes.currentText() == "THW"
        self.sizes[0].setEnabled(temporal)
        if not temporal:
            self.sizes[0].setValue(1)
        self._generate_kernel()

    def _generate_kernel(self):
        shape = tuple(spin.value() for spin in self.sizes)
        weights = np.zeros(shape, dtype=np.float64)
        preset = self.preset.currentText()
        if preset == "均值核":
            weights.fill(1.0)
        elif preset == "W 方向差分":
            if shape[-1] < 2:
                self.status.setText("W 方向差分需要至少 2 列")
                return
            weights[shape[0] // 2, shape[1] // 2, 0] = -1
            weights[shape[0] // 2, shape[1] // 2, -1] = 1
        elif preset == "单位核":
            weights[tuple(size // 2 for size in shape)] = 1
        self._kernel = weights
        self._preview_id = -1
        self.preview_pool.cancel_owner(self._preview_owner)
        self.preview_button.setEnabled(True)
        self._generated_preset = preset
        self.normalize.setChecked(preset == "均值核")
        self._layer = 0
        self.layer_selector.blockSignals(True)
        self.layer_selector.setRange(0, shape[0] - 1)
        self.layer_selector.setValue(0)
        self.layer_selector.blockSignals(False)
        for spin, size in zip(self.origins, shape):
            spin.setRange(-(size // 2), (size - 1) // 2)
            spin.setValue(0)
        self.origins[0].setEnabled(self.axes.currentText() == "THW")
        self._show_layer()

    def _save_layer(self):
        layer = np.empty(self._kernel.shape[1:], dtype=np.float64)
        for row in range(layer.shape[0]):
            for col in range(layer.shape[1]):
                try:
                    layer[row, col] = float(self.weights_table.item(row, col).text())
                except (ValueError, AttributeError) as exc:
                    raise ValueError(f"第 {row + 1} 行、第 {col + 1} 列需要填写数值") from exc
        self._kernel[self._layer] = layer

    def _show_layer(self):
        layer = self._kernel[self._layer]
        self.weights_table.setRowCount(layer.shape[0])
        self.weights_table.setColumnCount(layer.shape[1])
        for row in range(layer.shape[0]):
            for col in range(layer.shape[1]):
                self.weights_table.setItem(row, col, QTableWidgetItem(f"{layer[row, col]:.12g}"))

    def _select_layer(self, index):
        try:
            self._save_layer()
        except ValueError as exc:
            self.status.setText(str(exc))
            self.layer_selector.blockSignals(True)
            self.layer_selector.setValue(self._layer)
            self.layer_selector.blockSignals(False)
            return
        self._layer = index
        self._show_layer()

    def build_plan(self):
        if self.source is None:
            raise ValueError("请先选择数据")
        if tuple(spin.value() for spin in self.sizes) != self._kernel.shape or self.preset.currentText() != self._generated_preset:
            raise ValueError("核尺寸或预设已变更，请先点击生成卷积核，再编辑各层权重")
        self._save_layer()
        temporal = self.axes.currentText() == "THW"
        kernel = self._kernel.copy() if temporal else self._kernel[0].copy()
        origin = tuple(spin.value() for spin in self.origins)
        plan = ConvolutionPlan(
            source=self.source, kernel=kernel, axes=self.axes.currentText(),
            payload_key=self.payload_key, normalize=self.normalize.isChecked(),
            boundary=self.boundary.currentData(), cval=self.cval.value(),
            origin=origin if temporal else origin[1:],
        )
        validate_convolution(plan)
        return plan

    def _execute(self):
        try:
            self.execute_requested.emit(self.build_plan())
        except (ValueError, MemoryError) as exc:
            self.status.setText(str(exc))

    def bind_task(self, task):
        self.task_id = task.task_id
        self.run_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.status.setText("卷积任务已提交，可在任务面板查看进度")

    def task_finished(self, task):
        if task.task_id != self.task_id:
            return
        self.run_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.task_id = None
        messages = {TaskStatus.COMPLETED: "卷积完成，结果可从历史数据中显示或导出",
                    TaskStatus.CANCELLED: "卷积已取消", TaskStatus.FAILED: f"卷积失败 {task.error}"}
        self.status.setText(messages.get(task.status, task.message))

    def _request_preview(self):
        try:
            plan = self.build_plan()
        except (ValueError, MemoryError) as exc:
            self.status.setText(str(exc))
            return
        try:
            self._preview_id = self.preview_pool.submit(
                self._preview_owner, plan, self.frame.value(), self.row.value(), self.column.value(),
            )
        except ValueError as exc:
            self.status.setText(str(exc))
            return
        self.preview_button.setEnabled(False)
        self.status.setText("正在读取 ROI 与卷积邻域")

    def _preview_completed(self, request_id, arrays):
        if self._closing or request_id != self._preview_id:
            return
        self.preview_button.setEnabled(True)
        for view, array in zip((self.original_preview, self.filtered_preview), arrays):
            view.setImage(np.abs(array) if np.iscomplexobj(array) else array, autoRange=True)
        self.status.setText("ROI 预览完成，显示复数数据时取幅值")

    def _preview_failed(self, request_id, message):
        if self._closing or request_id != self._preview_id:
            return
        self.preview_button.setEnabled(True)
        self.status.setText(message)

    def prepare_shutdown(self):
        self._closing = True
        self._preview_id = -1
        self.preview_pool.prepare_shutdown()

    def shutdown_pending(self):
        return self.preview_pool.shutdown_pending()

    def closeEvent(self, event):
        self._closing = True
        self._preview_id = -1
        self.preview_pool.cancel_owner(self._preview_owner)
        if not self._connections_closed:
            self._connections_closed = True
            self.preview_pool.completed.disconnect(self._preview_completed)
            self.preview_pool.failed.disconnect(self._preview_failed)
            self.closed.emit()
        super().closeEvent(event)
