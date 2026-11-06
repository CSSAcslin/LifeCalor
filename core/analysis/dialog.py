from __future__ import annotations

import numpy as np
from PyQt5.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QFormLayout, QLabel, QVBoxLayout,
)

from dataio.classification import describe_source, DataCategory
from diagnostics import report_warning
from history.annotations import display_name_for
from selection.policy import rect_mask_from_canvas


STATISTICS = (
    ("均值", "mean"), ("最大值", "max"), ("最小值", "min"),
    ("中位数", "median"), ("0.75 分位数", "quantile_075"),
    ("标准差", "std"), ("求和", "sum"), ("方差", "var"),
)


def validate_source(action, source, mode=1):
    info = describe_source(source)
    shape = info.shape
    metadata = getattr(source, "out_processed", {}) or {}
    try:
        dtype = np.dtype(info.dtype)
    except TypeError:
        return "需要具有明确 dtype 的数值数组。"
    if dtype.kind not in "biufc":
        return "需要数值数组。"
    accepts_complex = (
        action.action_id.startswith("transform.")
        or action.action_id == "statistics.distribution"
    )
    if dtype.kind == "c" and not accepts_complex:
        return "该算法需要实值输入，请先在数据计算器中明确提取幅值。"
    lifetime_map = (
        getattr(source, "type_processed", "") == "lifetime_distribution"
        and len(shape) == 2
    )
    if action.input_kind == "spectrum":
        spectrum = metadata.get("twoD_FFT")
        if (
            spectrum is None or len(spectrum.shape) not in (2, 3)
            or np.dtype(spectrum.dtype).kind != "c"
        ):
            return "请选择包含 twoD_FFT 完整复频谱的傅里叶变换结果。"
    elif action.input_kind == "image_or_video":
        if len(shape) != 2 and info.category is not DataCategory.VIDEO:
            return "需要二维图像或具有时间轴的三维序列。"
    elif not (action.input_kind == "video_or_lifetime" and lifetime_map):
        if len(shape) != 3 or info.category is not DataCategory.VIDEO:
            return "需要具有明确时间轴的 THW 三维序列。"
    if len(shape) == 3 and info.axes and info.axes[0] != "T":
        return "当前算法要求时间轴位于首维，请先在数据计算器中转置。"
    if len(shape) == 3 and action.input_kind != "spectrum":
        times = getattr(source, "time_point", None)
        if times is None or len(times) != shape[0]:
            return "需要与 THW 首维匹配的时间轴。"
    if action.processed_types and mode == 1:
        if getattr(source, "type_processed", "") not in action.processed_types:
            return "默认模式需要前序处理结果：" + " / ".join(action.processed_types)
    if action.action_id.startswith("lifetime.") or action.action_id.startswith("physics."):
        times = getattr(source, "time_point", None)
        if not lifetime_map and (
            times is None or len(times) != shape[0]
            or not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0)
        ):
            return "需要与帧数匹配、有限且递增的物理时间轴。"
    return ""


class AnalysisInputDialog(QDialog):
    """Confirm the exact data object and optional matching canvas ROI once."""

    def __init__(self, window, action, sources, preferred=None):
        super().__init__(window)
        self.window = window
        self.action = action
        self.sources = sources
        self.setWindowTitle(action.title.rstrip("…") + " — 输入数据")
        self.setMinimumWidth(470)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.source_combo = QComboBox()
        self.source_combo.setMinimumContentsLength(25)
        self.source_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        for source in sources:
            info = describe_source(source)
            kind = getattr(source, "type_processed", "原始数据")
            self.source_combo.addItem(
                f"{display_name_for(source)} · {kind} · {info.shape}", source
            )
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        self.roi_combo = QComboBox()
        self.method_combo = QComboBox()
        for label, method in STATISTICS:
            self.method_combo.addItem(label, method)
        form.addRow("输入数据", self.source_combo)
        form.addRow(self.summary)
        if action.roi:
            form.addRow("匹配画布的选区", self.roi_combo)
        if action.action_id == "statistics.roi":
            form.addRow("统计方法", self.method_combo)
        layout.addLayout(form)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.button(QDialogButtonBox.Ok).setText("使用此输入")
        self.buttons.accepted.connect(self._accept_input)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self.source_combo.currentIndexChanged.connect(self._update_source)
        index = next((i for i, source in enumerate(sources) if source is preferred), 0)
        self.source_combo.setCurrentIndex(index)
        self._update_source()

    @property
    def source(self):
        return self.source_combo.currentData()

    @property
    def mask(self):
        return self.roi_combo.currentData() if self.action.roi else None

    @property
    def method(self):
        return self.method_combo.currentData()

    def _update_source(self):
        source = self.source
        self.roi_combo.clear()
        if source is None:
            self.buttons.button(QDialogButtonBox.Ok).setEnabled(False)
            return
        info = describe_source(source)
        details = (
            f"原始名称 {source.name}\nShape {info.shape} · dtype {info.dtype}"
            f" · 轴 {''.join(info.axes) or '由时间轴识别'}"
        )
        metadata = {
            **(getattr(source, "parameters", {}) or {}),
            **(getattr(source, "out_processed", {}) or {}),
        }
        scales = []
        for key in ("fps", "time_step", "time_unit", "space_step", "space_unit"):
            value = metadata.get(key)
            if isinstance(value, (str, int, float)):
                scales.append(f"{key}={value}")
        if scales:
            details += "\n" + " · ".join(scales)
        if self.action.action_id == "physics.heat":
            details += "\n传热换算使用主寿命图，双指数模型的主字段为 tau1。"
        self.summary.setText(details)
        for canvas in self.window.image_display.display_canvas:
            if getattr(canvas.data, "timestamp_inherited", None) != source.timestamp:
                continue
            _, mask = self.window.image_display.get_draw_roi(canvas.id)
            options = [("绘制 ROI", mask), ("Anchor", getattr(canvas, "anchor_mask", None))]
            if getattr(canvas, "v_rect_roi", None) is not None:
                options.append(("矩形选区", rect_mask_from_canvas(canvas)))
            for label, mask in options:
                if mask is not None and np.asarray(mask).shape == tuple(source.framesize):
                    if np.any(mask):
                        self.roi_combo.addItem(f"画布 {canvas.id} · {label}", mask)
        error = validate_source(self.action, source, self.window.mode)
        if self.action.roi and self.roi_combo.count() == 0:
            error = "请在该数据对应的画布上绘制并选择非空区域。"
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(not error)
        self.buttons.button(QDialogButtonBox.Ok).setToolTip(error)
        if error:
            self.summary.setText(details + "\n" + error)

    def _accept_input(self):
        error = validate_source(self.action, self.source, self.window.mode)
        if self.action.roi and self.mask is None:
            error = "请选择与数据匹配的非空 ROI。"
        if error:
            report_warning(self, "输入数据", error)
            return
        self.accept()
