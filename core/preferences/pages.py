from __future__ import annotations

import os
from pathlib import Path

from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtGui import QColor, QDesktopServices, QDoubleValidator, QIcon
from PyQt5.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QFileDialog,
    QFrame,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from appearance import get_theme_manager
from compute.dialog import ComputeSettingsPage
from compute.model import PrecisionPolicy


def _form_group(title):
    group = QGroupBox(title)
    form = QFormLayout(group)
    form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
    return group, form


def _scientific_edit(value, minimum=0.0, maximum=1e100):
    edit = QLineEdit(f"{float(value):.8g}")
    validator = QDoubleValidator(float(minimum), float(maximum), 12, edit)
    validator.setNotation(QDoubleValidator.ScientificNotation)
    edit.setValidator(validator)
    return edit


class ColorButton(QPushButton):
    def __init__(self, color, parent=None):
        super().__init__(parent)
        self._color = str(color)
        self.setMinimumWidth(90)
        self.clicked.connect(self._choose)
        self._refresh()

    def _choose(self):
        color = QColorDialog.getColor(QColor(self._color), self)
        if color.isValid():
            self.set_color(color.name())

    def set_color(self, color):
        self._color = str(color)
        self._refresh()

    def color(self):
        return self._color

    def _refresh(self):
        self.setText(self._color)
        foreground = "#111111" if QColor(self._color).lightness() > 150 else "#ffffff"
        self.setStyleSheet(
            f"QPushButton {{ background: {self._color}; color: {foreground}; }}"
        )


TOOL_PRESENTATION = {
    "Pen": (":/icons/icon_pen.svg", "画笔"),
    "Line": (":/icons/icon_line.svg", "直线"),
    "Rect": (":/icons/icon_rect.svg", "矩形"),
    "Ellipse": (":/icons/icon_ellipse.svg", "椭圆"),
    "Eraser": (":/icons/icon_eraser.svg", "橡皮擦"),
    "Fill": (":/icons/icon_fill.svg", "填充"),
    "V-line": (":/icons/icon_v-line.svg", "矢量线"),
    "V-rect": (":/icons/icon_v-rect.svg", "矢量框"),
    "Anchor": (":/icons/icon_anchor.svg", "Anchor"),
}


class ToolSettingsSection(QWidget):
    """Compact, unframed settings section with explicit tool scope."""

    def __init__(self, title, icon_path, description, tools, parent=None):
        super().__init__(parent)
        self.setObjectName("preferencesToolSection")
        layout = QGridLayout(self)
        layout.setContentsMargins(8, 8, 12, 8)
        layout.setHorizontalSpacing(22)
        layout.setVerticalSpacing(7)
        layout.setColumnStretch(0, 5)
        layout.setColumnStretch(1, 4)

        header = QHBoxLayout()
        header.setSpacing(9)
        icon = QLabel()
        icon.setFixedSize(28, 28)
        icon.setPixmap(QIcon(icon_path).pixmap(24, 24))
        icon.setAlignment(Qt.AlignCenter)
        text = QVBoxLayout()
        text.setSpacing(1)
        title_label = QLabel(title)
        title_label.setObjectName("preferencesToolTitle")
        description_label = QLabel(description)
        description_label.setObjectName("preferencesToolDescription")
        description_label.setWordWrap(True)
        text.addWidget(title_label)
        text.addWidget(description_label)
        header.addWidget(icon)
        header.addLayout(text, 1)
        layout.addLayout(header, 0, 0)

        scope = QGridLayout()
        scope.setContentsMargins(36, 0, 0, 0)
        scope.setHorizontalSpacing(10)
        scope.setVerticalSpacing(5)
        scope_title = QLabel("影响工具")
        scope_title.setObjectName("preferencesToolScopeTitle")
        scope.addWidget(scope_title, 0, 0, 1, 3)
        for index, tool in enumerate(tools):
            tool_icon, label = TOOL_PRESENTATION[tool]
            indicator = QWidget()
            indicator.setObjectName("preferencesToolScopeItem")
            indicator.setToolTip(f"本组设置会应用到 {label} 工具")
            indicator_layout = QHBoxLayout(indicator)
            indicator_layout.setContentsMargins(0, 0, 0, 0)
            indicator_layout.setSpacing(4)
            icon_label = QLabel()
            icon_label.setPixmap(QIcon(tool_icon).pixmap(16, 16))
            icon_label.setFixedSize(18, 18)
            icon_label.setToolTip(label)
            text_label = QLabel(label)
            text_label.setObjectName("preferencesToolScopeText")
            text_label.setToolTip(f"这些设置会应用到 {label} 工具")
            indicator_layout.addWidget(icon_label)
            indicator_layout.addWidget(text_label)
            indicator_layout.addStretch()
            row, column = divmod(index, 3)
            scope.addWidget(indicator, row + 1, column)
        for column in range(3):
            scope.setColumnStretch(column, 1)
        layout.addLayout(scope, 1, 0)

        self.form = QFormLayout()
        self.form.setContentsMargins(0, 4, 0, 0)
        self.form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.form.setHorizontalSpacing(12)
        self.form.setVerticalSpacing(7)
        layout.addLayout(self.form, 0, 1, 2, 1)

        divider = QFrame()
        divider.setObjectName("preferencesToolDivider")
        divider.setFrameShape(QFrame.HLine)
        layout.addWidget(divider, 2, 0, 1, 2)


class PreferencesPage(QWidget):
    title = ""

    def __init__(self, window, parent=None):
        super().__init__(parent)
        self.window = window
        self._baseline = {}

    def values(self):
        raise NotImplementedError

    def current_values(self):
        raise NotImplementedError

    def changed_values(self):
        values = self.values()
        return {
            key: value for key, value in values.items()
            if self._baseline.get(key) != value
        }

    def conflicts(self):
        current = self.current_values()
        return [
            key for key in self.changed_values()
            if current.get(key) != self._baseline.get(key)
        ]

    def has_changes(self):
        return bool(self.changed_values())

    def validate_page(self):
        return ""

    def apply_page(self):
        raise NotImplementedError

    def mark_applied(self):
        self._baseline = self.current_values()

    def reset_defaults(self):
        raise NotImplementedError


class GeneralPage(PreferencesPage):
    title = "常规与更新"

    def __init__(self, window, parent=None):
        super().__init__(window, parent)
        layout = QVBoxLayout(self)
        group, form = _form_group("启动与更新")
        self.auto_update = QCheckBox("启动后按既有检查间隔自动检查更新")
        form.addRow("自动检查更新", self.auto_update)
        form.addRow("当前版本", QLabel(str(window.current_version)))
        self.log_path = QLineEdit(str(window.get_log_path()))
        self.log_path.setReadOnly(True)
        open_log = QPushButton("打开日志目录")
        open_log.clicked.connect(self._open_log_directory)
        row = QHBoxLayout()
        row.addWidget(self.log_path, 1)
        row.addWidget(open_log)
        form.addRow("日志文件", row)
        layout.addWidget(group)
        layout.addStretch()
        self.auto_update.setChecked(
            window.settings.value("should_check", True, type=bool)
        )
        self._baseline = self.current_values()

    def _open_log_directory(self):
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(self.log_path.text()).parent)))

    def values(self):
        return {"should_check": self.auto_update.isChecked()}

    def current_values(self):
        return {
            "should_check": self.window.settings.value(
                "should_check", True, type=bool
            )
        }

    def apply_page(self):
        for key, value in self.changed_values().items():
            self.window.settings.setValue(key, value)
        self.window.settings.sync()
        self.mark_applied()

    def reset_defaults(self):
        self.auto_update.setChecked(True)


class AppearancePage(PreferencesPage):
    title = "外观与画布"
    TOOL_DEFAULTS = {
        "pen_size": 2,
        "pen_color": "#008000",
        "fill_color": "#006400",
        "vector_color": "#FFFF00",
        "vector_width": 2,
        "auto_fill": False,
        "anchor_select": False,
        "anchor_shape": "square",
        "anchor_size": 5,
        "anchor_method": "mean",
    }
    METHODS = (
        ("平均值", "mean"), ("最大值", "max"), ("最小值", "min"),
        ("中位值", "median"), ("0.75 分位数", "quantile_075"),
        ("标准差", "std"), ("求和", "sum"), ("方差", "var"),
        ("值分布统计", "value_distribution"),
    )

    def __init__(self, window, parent=None):
        super().__init__(window, parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 12)
        layout.setSpacing(12)
        appearance, appearance_form = _form_group("界面外观")
        self.theme = QComboBox()
        self.theme.addItem("清爽浅色", "light")
        self.theme.addItem("石墨深色", "dark")
        appearance_form.addRow("界面主题", self.theme)
        theme_note = QLabel("应用后立即更新所有窗口；取消不会保留未应用的主题选择。")
        theme_note.setObjectName("preferencesToolDescription")
        theme_note.setWordWrap(True)
        appearance_form.addRow("", theme_note)
        layout.addWidget(appearance)

        self.pen_size = QSpinBox()
        self.pen_size.setRange(1, 100)
        self.vector_width = QSpinBox()
        self.vector_width.setRange(1, 100)
        self.pen_color = ColorButton("#008000")
        self.fill_color = ColorButton("#006400")
        self.vector_color = ColorButton("#FFFF00")
        self.auto_fill = QCheckBox("新绘制区域默认填充")
        self.anchor_select = QCheckBox("新画布默认启用快速提取")
        self.anchor_shape = QComboBox()
        self.anchor_shape.addItem("正方形", "square")
        self.anchor_shape.addItem("圆形", "circle")
        self.anchor_size = QSpinBox()
        self.anchor_size.setRange(1, 1000)
        self.anchor_method = QComboBox()
        for label, value in self.METHODS:
            self.anchor_method.addItem(label, value)

        tool_grid = QGridLayout()
        tool_grid.setContentsMargins(0, 0, 0, 0)
        tool_grid.setHorizontalSpacing(0)
        tool_grid.setVerticalSpacing(2)
        tool_grid.setColumnStretch(0, 1)

        drawing = ToolSettingsSection(
            "绘制与擦除",
            ":/icons/icon_pen.svg",
            "线宽用于画笔、直线、矩形、椭圆和橡皮擦；描边颜色不影响橡皮擦。",
            ("Pen", "Line", "Rect", "Ellipse", "Eraser"),
        )
        drawing.form.addRow("线宽 / 橡皮擦大小", self.pen_size)
        drawing.form.addRow("描边颜色", self.pen_color)

        filling = ToolSettingsSection(
            "区域填充",
            ":/icons/icon_fill.svg",
            "填充颜色用于填充工具；开启自动填充后也用于新绘制的矩形和椭圆。",
            ("Rect", "Ellipse", "Fill"),
        )
        filling.form.addRow("填充颜色", self.fill_color)
        filling.form.addRow("矩形 / 椭圆自动填充", self.auto_fill)

        vector = ToolSettingsSection(
            "矢量选区",
            ":/icons/icon_v-line.svg",
            "颜色用于矢量线、矢量框和 Anchor；选区宽度只控制矢量线。",
            ("V-line", "V-rect", "Anchor"),
        )
        vector.form.addRow("选区颜色", self.vector_color)
        vector.form.addRow("矢量线宽度", self.vector_width)

        anchor = ToolSettingsSection(
            "Anchor 快速提取",
            ":/icons/icon_anchor.svg",
            "控制 Anchor 的默认选区和统计方法；显示颜色沿用“矢量选区”的颜色。",
            ("Anchor",),
        )
        anchor.form.addRow("默认启用快速提取", self.anchor_select)
        anchor.form.addRow("选区形状", self.anchor_shape)
        anchor.form.addRow("选区大小", self.anchor_size)
        anchor.form.addRow("统计方法", self.anchor_method)

        tool_grid.addWidget(drawing, 0, 0)
        tool_grid.addWidget(filling, 1, 0)
        tool_grid.addWidget(vector, 2, 0)
        tool_grid.addWidget(anchor, 3, 0)
        layout.addLayout(tool_grid)
        scope_note = QLabel(
            "应用后，这些参数用于现有画布接下来的绘制以及之后新建的画布；"
            "不会改写已经绘制完成的 ROI、矢量选区或 Anchor 结果。"
        )
        scope_note.setObjectName("preferencesToolDescription")
        scope_note.setWordWrap(True)
        layout.addWidget(scope_note)
        layout.addStretch()
        self._load(self.current_values())
        self._baseline = self.current_values()

    @staticmethod
    def _select(combo, value):
        index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)

    def _load(self, values):
        self._select(self.theme, values["theme"])
        self.pen_size.setValue(int(values["pen_size"]))
        self.pen_color.set_color(values["pen_color"])
        self.fill_color.set_color(values["fill_color"])
        self.vector_color.set_color(values["vector_color"])
        self.vector_width.setValue(int(values["vector_width"]))
        self.auto_fill.setChecked(bool(values["auto_fill"]))
        self.anchor_select.setChecked(bool(values["anchor_select"]))
        self._select(self.anchor_shape, values["anchor_shape"])
        self.anchor_size.setValue(int(values["anchor_size"]))
        self._select(self.anchor_method, values["anchor_method"])

    def values(self):
        return {
            "theme": str(self.theme.currentData()),
            "pen_size": self.pen_size.value(),
            "pen_color": self.pen_color.color(),
            "fill_color": self.fill_color.color(),
            "vector_color": self.vector_color.color(),
            "vector_width": self.vector_width.value(),
            "auto_fill": self.auto_fill.isChecked(),
            "anchor_select": self.anchor_select.isChecked(),
            "anchor_shape": str(self.anchor_shape.currentData()),
            "anchor_size": self.anchor_size.value(),
            "anchor_method": str(self.anchor_method.currentData()),
        }

    def current_values(self):
        manager = get_theme_manager()
        values = {
            "theme": manager.current_theme if manager is not None else "light"
        }
        values.update({
            key: self.window.tool_params.get(key, default)
            for key, default in self.TOOL_DEFAULTS.items()
        })
        return values

    def apply_page(self):
        patch = self.changed_values()
        theme = patch.pop("theme", None)
        if theme is not None and not self.window._set_interface_theme(theme):
            raise RuntimeError("界面主题未能应用，已保留原主题")
        if patch:
            self.window.tool_params.update(patch)
            self.window._save_param_group("tool", self.window.tool_params)
            image_display = getattr(self.window, "image_display", None)
            if image_display is not None:
                image_display.tool_parameters.update(self.window.tool_params)
                for canvas in tuple(image_display.display_canvas):
                    canvas.set_toolset(
                        image_display.tool_parameters,
                        update_display_style=False,
                    )
        self.window.settings.sync()
        self.mark_applied()

    def reset_defaults(self):
        values = {"theme": "light", **self.TOOL_DEFAULTS}
        self._load(values)


class PlotPage(PreferencesPage):
    title = "绘图默认值"
    DEFAULTS = {
        "line_style": "--", "line_width": 2, "marker_style": "s",
        "marker_size": 6, "color": "#1f77b4", "show_grid": False,
        "heatmap_cmap": "jet", "contour_levels": 10, "set_axis": True,
    }

    def __init__(self, window, parent=None):
        super().__init__(window, parent)
        layout = QVBoxLayout(self)
        group, form = _form_group("结果图默认样式")
        self.color = ColorButton("#1f77b4")
        self.line_style = QComboBox()
        for label, value in (("实线", "-"), ("虚线", "--"), ("点线", ":"), ("点划线", "-.")):
            self.line_style.addItem(label, value)
        self.line_width = QSpinBox()
        self.line_width.setRange(1, 10)
        self.marker = QComboBox()
        for label, value in (("无", ""), ("圆形", "o"), ("方形", "s"), ("三角形", "^"), ("星号", "*")):
            self.marker.addItem(label, value)
        self.marker_size = QSpinBox()
        self.marker_size.setRange(1, 20)
        self.grid = QCheckBox()
        self.axis = QCheckBox()
        self.cmap = QComboBox()
        for name in ("jet", "plasma", "inferno", "magma", "viridis"):
            self.cmap.addItem(name.title(), name)
        self.contours = QSpinBox()
        self.contours.setRange(0, 50)
        self.contours.setSpecialValueText("无等高线")
        self.apply_current = QCheckBox("同时应用到当前结果图（不会重新计算）")
        form.addRow("线条颜色", self.color)
        form.addRow("线条样式", self.line_style)
        form.addRow("线条宽度", self.line_width)
        form.addRow("标记", self.marker)
        form.addRow("标记大小", self.marker_size)
        form.addRow("显示网格", self.grid)
        form.addRow("设置轴范围", self.axis)
        form.addRow("热图颜色映射", self.cmap)
        form.addRow("等高线级别", self.contours)
        form.addRow("", self.apply_current)
        layout.addWidget(group)
        layout.addStretch()
        self._load(self.current_values())
        self._baseline = self.current_values()

    @staticmethod
    def _select(combo, value):
        index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)

    def _load(self, values):
        self.color.set_color(values["color"])
        self._select(self.line_style, values["line_style"])
        self.line_width.setValue(int(values["line_width"]))
        self._select(self.marker, values["marker_style"])
        self.marker_size.setValue(int(values["marker_size"]))
        self.grid.setChecked(bool(values["show_grid"]))
        self.axis.setChecked(bool(values["set_axis"]))
        self._select(self.cmap, str(values["heatmap_cmap"]).lower())
        self.contours.setValue(int(values["contour_levels"]))

    def values(self):
        return {
            "color": self.color.color(),
            "line_style": str(self.line_style.currentData()),
            "line_width": self.line_width.value(),
            "marker_style": str(self.marker.currentData()),
            "marker_size": self.marker_size.value(),
            "show_grid": self.grid.isChecked(),
            "set_axis": self.axis.isChecked(),
            "heatmap_cmap": str(self.cmap.currentData()),
            "contour_levels": self.contours.value(),
        }

    def current_values(self):
        return {
            key: self.window.plot_params.get(key, default)
            for key, default in self.DEFAULTS.items()
        }

    def apply_page(self):
        patch = self.changed_values()
        if patch:
            self.window.plot_params.update(patch)
            self.window._save_param_group("plot", self.window.plot_params)
        if patch or self.apply_current.isChecked():
            self.window.result_display.update_plot_settings(
                self.window.plot_params,
                update=self.apply_current.isChecked(),
            )
        self.apply_current.setChecked(False)
        self.window.settings.sync()
        self.mark_applied()

    def has_changes(self):
        return super().has_changes() or self.apply_current.isChecked()

    def conflicts(self):
        if not super().has_changes():
            return []
        return super().conflicts()

    def reset_defaults(self):
        self._load(self.DEFAULTS)


class LifetimePage(PreferencesPage):
    title = "寿命拟合默认值"
    DEFAULTS = {
        "from_start_cal": False, "r_squared_min": 0.4,
        "peak_min": 0, "peak_max": 50, "tau_min": 1e-3, "tau_max": 1e3,
    }

    def __init__(self, window, parent=None):
        super().__init__(window, parent)
        layout = QVBoxLayout(self)
        group, form = _form_group("指数拟合默认值")
        self.from_start = QCheckBox("从序列起点拟合；关闭时从信号峰值拟合")
        self.peak_min = QSpinBox()
        self.peak_min.setRange(0, 2_000_000_000)
        self.peak_max = QSpinBox()
        self.peak_max.setRange(0, 2_000_000_000)
        self.r2 = _scientific_edit(0.4, -1.0, 1.0)
        self.tau_min = _scientific_edit(1e-3, 0.0)
        self.tau_max = _scientific_edit(1e3, 0.0)
        form.addRow("拟合起点", self.from_start)
        form.addRow("峰位最小帧", self.peak_min)
        form.addRow("峰位最大帧", self.peak_max)
        form.addRow("R² 最小值", self.r2)
        form.addRow("τ 最小值（当前数据时间单位）", self.tau_min)
        form.addRow("τ 最大值（当前数据时间单位）", self.tau_max)
        layout.addWidget(group)
        note = QLabel("这些值只作为新任务默认值，不会改写当前数据的单位。")
        note.setWordWrap(True)
        layout.addWidget(note)
        layout.addStretch()
        self._load(self.current_values())
        self._baseline = self.current_values()

    def _load(self, values):
        self.from_start.setChecked(bool(values["from_start_cal"]))
        self.peak_min.setValue(int(values["peak_min"]))
        self.peak_max.setValue(int(values["peak_max"]))
        self.r2.setText(f'{float(values["r_squared_min"]):.8g}')
        self.tau_min.setText(f'{float(values["tau_min"]):.8g}')
        self.tau_max.setText(f'{float(values["tau_max"]):.8g}')

    def values(self):
        return {
            "from_start_cal": self.from_start.isChecked(),
            "peak_min": self.peak_min.value(),
            "peak_max": self.peak_max.value(),
            "r_squared_min": float(self.r2.text()),
            "tau_min": float(self.tau_min.text()),
            "tau_max": float(self.tau_max.text()),
        }

    def current_values(self):
        return {
            key: self.window.cal_set_params.get(key, default)
            for key, default in self.DEFAULTS.items()
        }

    def validate_page(self):
        try:
            values = self.values()
        except ValueError:
            return "请输入有效的拟合数值。"
        if values["peak_min"] > values["peak_max"]:
            return "峰位最小帧不能大于最大帧。"
        if values["tau_min"] <= 0 or values["tau_min"] > values["tau_max"]:
            return "τ 范围必须为正数，且最小值不能大于最大值。"
        if not -1.0 <= values["r_squared_min"] <= 1.0:
            return "R² 最小值必须位于 -1 到 1 之间。"
        return ""

    def apply_page(self):
        from LifetimeCalculator import LifetimeCalculator

        patch = self.changed_values()
        if patch:
            self.window.cal_set_params.update(patch)
            self.window._save_param_group("cal_set", self.window.cal_set_params)
            LifetimeCalculator.set_cal_parameters(self.window.cal_set_params)
            self.window.plot_params["_from_start_cal"] = self.window.cal_set_params["from_start_cal"]
            self.window.result_display.update_plot_settings(
                self.window.plot_params, update=False
            )
        self.window.settings.sync()
        self.mark_applied()

    def reset_defaults(self):
        self._load(self.DEFAULTS)


class CachePage(PreferencesPage):
    title = "缓存与存储"
    DEFAULTS = {
        "cache_threshold_mb": 512,
        "memory_budget_mb": 4096,
        "cache_cleanup_startup": True,
    }

    def __init__(self, window, parent=None):
        super().__init__(window, parent)
        layout = QVBoxLayout(self)
        group, form = _form_group("大数组缓存")
        self.directory = QLineEdit()
        self.directory.setReadOnly(True)
        browse = QPushButton("浏览")
        browse.clicked.connect(self._browse)
        open_directory = QPushButton("打开目录")
        open_directory.clicked.connect(self._open_directory)
        directory_row = QHBoxLayout()
        directory_row.addWidget(self.directory, 1)
        directory_row.addWidget(browse)
        directory_row.addWidget(open_directory)
        self.threshold = QSpinBox()
        self.threshold.setRange(1, 1024 * 1024)
        self.threshold.setSuffix(" MB")
        self.memory = QSpinBox()
        self.memory.setRange(256, 1024 * 1024)
        self.memory.setSuffix(" MB")
        self.cleanup = QCheckBox("启动时清理未被历史索引保留的临时缓存")
        form.addRow("缓存目录", directory_row)
        form.addRow("数组写入阈值", self.threshold)
        form.addRow("交互数据内存预算", self.memory)
        form.addRow("启动清理", self.cleanup)
        layout.addWidget(group)
        note = QLabel(
            "切换目录后，新缓存写入新目录；旧历史与旧文件不会自动搬移或删除。"
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        layout.addStretch()
        self._load(self.current_values())
        self._baseline = self.current_values()

    def _load(self, values):
        self.directory.setText(str(values["cache_directory"]))
        self.threshold.setValue(int(values["cache_threshold_mb"]))
        self.memory.setValue(int(values["memory_budget_mb"]))
        self.cleanup.setChecked(bool(values["cache_cleanup_startup"]))

    def _browse(self):
        directory = QFileDialog.getExistingDirectory(
            self, "选择缓存目录", self.directory.text(), QFileDialog.ShowDirsOnly
        )
        if directory:
            self.directory.setText(directory)

    def _open_directory(self):
        directory = self.directory.text().strip()
        if directory:
            QDesktopServices.openUrl(QUrl.fromLocalFile(directory))

    def values(self):
        return {
            "cache_directory": self.directory.text().strip(),
            "cache_threshold_mb": self.threshold.value(),
            "memory_budget_mb": self.memory.value(),
            "cache_cleanup_startup": self.cleanup.isChecked(),
        }

    def current_values(self):
        values = {
            "cache_directory": self.window.tool_params.get("cache_directory")
            or self.window.default_cache_directory()
        }
        values.update({
            key: self.window.tool_params.get(key, default)
            for key, default in self.DEFAULTS.items()
        })
        return values

    def validate_page(self):
        raw_directory = self.directory.text().strip()
        if not raw_directory:
            return "请选择缓存目录。"
        directory = Path(raw_directory)
        candidate = directory if directory.exists() else directory.parent
        if not candidate.exists() or not os.access(str(candidate), os.W_OK):
            return "缓存目录不存在，且其上级目录不可写。"
        return ""

    def apply_page(self):
        patch = self.changed_values()
        old_directory = self.window.tool_params.get(
            "cache_directory"
        ) or self.window.default_cache_directory()
        if patch:
            self.window.tool_params.update(patch)
            self.window._save_param_group("tool", self.window.tool_params)
            self.window.apply_cache_settings()
            new_directory = self.window.tool_params["cache_directory"]
            if Path(old_directory) != Path(new_directory):
                self.window.history_controller.handle_cache_directory_change(
                    old_directory, new_directory
                )
        self.window.settings.sync()
        self.mark_applied()

    def reset_defaults(self):
        values = {
            "cache_directory": self.window.default_cache_directory(),
            **self.DEFAULTS,
        }
        self._load(values)


class ComputePageAdapter(ComputeSettingsPage):
    title = "计算与加速"

    def __init__(self, window, parent=None):
        self.window = window
        super().__init__(window.settings, parent)

    def validate_page(self):
        global_precision = PrecisionPolicy(self.precision_combo.currentData())
        if global_precision is PrecisionPolicy.SINGLE:
            for algorithm in ("lifetime_single", "lifetime_double"):
                selected = str(
                    self.algorithm_precision_combos[algorithm].currentData() or ""
                )
                if selected in {"", PrecisionPolicy.SINGLE.value}:
                    return (
                        "寿命拟合单精度尚未通过科学验收。"
                        "请将单/双指数寿命的精度覆盖设为“双精度”，"
                        "或修改全局精度。"
                    )
        return ""

    def apply_page(self):
        if self.has_changes():
            self.save_preferences()
            self.window._sync_lifetime_compute_controls()

    def mark_applied(self):
        pass
