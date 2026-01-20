import logging
import os
import sys
import time
import psutil
from math import ceil
from typing import List

import numpy as np
from PyQt5.QtGui import QColor, QIntValidator, QFont
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QGroupBox,
                             QRadioButton, QSpinBox, QLineEdit, QPushButton,
                             QLabel, QMessageBox, QFormLayout, QDoubleSpinBox, QColorDialog, QComboBox, QCheckBox,
                             QFileDialog, QWhatsThis, QTextBrowser, QTableWidget, QDialogButtonBox, QTableWidgetItem,
                             QHeaderView, QAbstractItemView, QTabWidget, QWidget, QListWidget, QListWidgetItem,
                             QSizePolicy, QTreeWidget, QTreeWidgetItem, QTextEdit)
from PyQt5.QtCore import Qt, QEvent, QTimer, QModelIndex, pyqtSignal, QSize
from fontTools.merge import layoutPreMerge

from DataManager import Data,ProcessedData
import re

class ToolBucket:
    @staticmethod
    def available_cpu_count(threshold=20.0, check_interval=1.0):
        """
        获取可用（闲置）CPU核心数（放在整理意味着目前这条命令只需要在弹窗级别使用）

        参数:
        :param threshold: CPU使用率阈值，低于此值认为核心可用
        :param check_interval: 检查CPU使用率的时间间隔

        返回:
        可用CPU核心数
        """
        cpu_percentages = psutil.cpu_percent(interval=check_interval, percpu=True)

        total_cpus = len(cpu_percentages)
        # 统计使用率低于阈值的核心
        available_cpus = sum(1 for usage in cpu_percentages if usage < threshold)

        return total_cpus, available_cpus

    @staticmethod
    def parse_frame_input(input:str, max_frame, parse_type="frame", freq=None, parent = None):
        """
        解析用户输入的帧数或频率范围，输出集合或索引数组。
        支持格式：
            - 单点: 5
            - 范围: 10-20
            - 带步长的范围: 10-50-2 (在10到50之间每隔1个取一个，即步长为2)
        :param input: 输入文字
        :param max_frame: 最大帧数
        :param parse_type: 处理模式（分为帧模式和频率模式）
        :param freq: 频率模式需要输入的频率表
        :param parent: QMessageBox 用到的
        :return: 排序后不重复的所有解包值
            """
        try:
            text = input
            max_frame = int(max_frame)
            if parse_type == 'frame':
                if not text:
                    raise ValueError("输入为空,请输入有效的帧数选择")
            elif parse_type == 'freq':
                if not text:
                    raise ValueError("请输入有效的频率范围")

            # 全选
            if text == 'all':
                if parse_type == 'frame':
                    return list(range(max_frame + 1))
                else:
                    return freq

            # 替换所有分隔符为逗号
            text = text.replace(';', ',').replace('，', ',')
            frames = set()
            target_idx = []
            parts = text.split(',')

            for part in parts:
                part = part.strip()  # 去除前后空格
                if not part:
                    continue

                # 处理范围输入 (e.g., "5-10", "20-25")
                if '-' in part:
                    range_parts = part.split('-')
                    length = len(range_parts)

                    if length not in [2, 3]:   # 3是带步长
                        raise ValueError(f"无效的范围格式: {part} (应为 '起始-结束' 或 '起始-结束-步长')")

                    try:
                        start = int(range_parts[0])
                        end = int(range_parts[1])
                        step = 1  # 默认步长
                        if length == 3:
                            step = int(range_parts[2])
                    except ValueError:
                        raise ValueError(f"范围包含非数字字符: {part}")

                    if start > end:
                        raise ValueError(f"起始帧({start})不能大于结束帧({end})")

                    if step < 1:
                        raise ValueError(f"步长({step})必须为正整数")

                    if parse_type == 'frame':
                        # 确保范围在有效区间内
                        if start < 0 or end > max_frame:
                            raise ValueError(f"范围 {part} 超出有效帧范围 (0-{max_frame})")
                        frames.update(range(start, end + 1, step))

                    elif parse_type == 'freq':
                        if freq is None:
                            raise ValueError("内部错误: 频率数组未提供")

                        if start < min(freq) or end > max(freq):
                            raise ValueError(f"范围 {part} 超出有效频率范围 ({min(freq)}-{max(freq)})")
                        indices_in_range = np.where((freq >= start) & (freq <= end))[0]

                        if len(indices_in_range) == 0:
                            continue  # 该范围内没有对应的频率点

                        stepped_indices = indices_in_range[::step]
                        target_idx.extend(stepped_indices)

                # 处理单帧数字
                else:
                    if parse_type == 'frame':
                        frame = int(part)
                        if frame < 0 or frame > max_frame:
                            raise ValueError(f"帧号 {frame} 超出有效范围 (0-{max_frame})")
                        frames.add(frame)
                    elif parse_type == 'freq':
                        val = float(part)  # 频率可能是浮点数输入
                        target_idx.append(np.argmin(np.abs(freq - val)))

            return sorted(frames) if parse_type == 'frame' else np.unique(target_idx)

        except ValueError as e:
            error_msg = str(e)
            if parse_type == 'frame':
                msg = (
                    f"无效输入: {error_msg}\n\n正确格式示例:\n"
                    "• 单帧: 5\n"
                    "• 序列: 1,3,5\n"
                    "• 范围: 10-15\n"
                    "• 间隔取值: 10-50-5 (10到50每5帧取1帧)\n"
                    "• 混合: 1, 10-20-2, 30-35\n"
                    f"• 所有帧: all\n\n当前有效范围: 0-{max_frame}"
                )
                if parent:
                    QMessageBox.warning(parent, "输入错误", msg)
                logging.warning(f"帧数输入错误: {error_msg} - 原文: {text}")

            if parse_type == 'freq':
                freq_min = min(freq) if freq is not None else 0
                freq_max = max(freq) if freq is not None else 0
                msg = (
                    f"无效输入: {error_msg}\n\n正确格式示例:\n"
                    "• 单频率: 30 (取最近似值)\n"
                    "• 范围: 10-20 (取范围内所有点)\n"
                    "• 间隔取值: 10-100-2 (10-100Hz范围内每隔1个点取值)\n"
                    "• 混合: 10, 30-50-5, 80\n"
                    f"• 全频率: all ({freq_min:.1f}-{freq_max:.1f})"
                )
                if parent:
                    QMessageBox.warning(parent, "输入错误", msg)
                logging.warning(f"频率输入错误: {error_msg} - 原文: {text}")

            return None

# 坏帧处理对话框
class BadFrameDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("坏帧处理")
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
        self.setMinimumWidth(400)

        self.init_ui()
        self.bad_frames = []

    def init_ui(self):
        layout = QVBoxLayout()

        # 检测方法选择
        method_group = QGroupBox("坏帧检测方法")
        method_layout = QHBoxLayout()

        self.auto_radio = QRadioButton("自动检测")
        self.auto_radio.setChecked(True)
        self.manual_radio = QRadioButton("手动输入")

        method_layout.addWidget(self.auto_radio)
        method_layout.addWidget(self.manual_radio)
        method_group.setLayout(method_layout)

        # 自动检测参数
        auto_group = QGroupBox("自动检测参数")
        auto_layout = QVBoxLayout()

        self.threshold_spin = QSpinBox()
        self.threshold_spin.setRange(1, 10)
        self.threshold_spin.setValue(3)
        self.threshold_spin.setSuffix("σ")

        auto_layout.addWidget(QLabel("敏感度 (标准差倍数):"))
        auto_layout.addWidget(self.threshold_spin)
        auto_group.setLayout(auto_layout)

        # 手动输入
        manual_group = QGroupBox("手动选择坏帧")
        manual_layout = QVBoxLayout()

        self.frame_input = QLineEdit()
        self.frame_input.setPlaceholderText("输入坏帧位置，用逗号分隔 (如: 12,25,30)")

        manual_layout.addWidget(QLabel("坏帧位置:"))
        manual_layout.addWidget(self.frame_input)
        manual_group.setLayout(manual_layout)

        # 处理方法选择
        process_group = QGroupBox("坏帧处理方法")

        # 帧平均参数
        avg_group = QGroupBox("修复参数")
        avg_layout = QVBoxLayout()

        self.n_frames_spin = QSpinBox()
        self.n_frames_spin.setRange(1, 10)
        self.n_frames_spin.setValue(2)
        self.n_frames_spin.setSuffix("帧")

        avg_layout.addWidget(QLabel("前后平均帧数:"))
        avg_layout.addWidget(self.n_frames_spin)
        avg_group.setLayout(avg_layout)

        # 按钮
        button_layout = QHBoxLayout()
        self.apply_btn = QPushButton("应用修复")
        self.apply_btn.clicked.connect(self.apply_fix)
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.clicked.connect(self.reject)

        button_layout.addWidget(self.apply_btn)
        button_layout.addWidget(self.cancel_btn)

        # 组装布局
        layout.addWidget(method_group)
        layout.addWidget(auto_group)
        layout.addWidget(manual_group)
        layout.addWidget(avg_group)
        layout.addLayout(button_layout)

        self.setLayout(layout)

        # 初始状态
        self.auto_radio.toggled.connect(self.update_ui_state)
        self.update_ui_state()

    def event(self, event):
        if event.type() == QEvent.EnterWhatsThisMode:
            QWhatsThis.leaveWhatsThisMode()
            self.show_custom_help()  # 调用自定义弹窗方法
            return True
        return super().event(event)

    def show_custom_help(self):
        """显示自定义非模态帮助对话框"""
        help_title = "功能帮助说明"
        help_content = """
        <h3>主要功能说明</h3>
        <ul>
            <li><b>功能1</b>: 描述内容...</li>
            <li><b>功能2</b>: 描述内容...</li>
            <li><b>高级选项</b>: 点击<a href="https://example.com">这里</a>查看详情</li>
        </ul>
        <p><i>注：本帮助窗口不会阻塞主界面操作</i></p>
        """

        # 创建并显示自定义对话框
        self.help_dialog = CustomHelpDialog(help_title, help_content, self)
        self.help_dialog.show()  # 非阻塞显示

    def update_ui_state(self):
        """根据选择的方法更新UI状态"""
        auto_selected = self.auto_radio.isChecked()
        self.threshold_spin.setEnabled(auto_selected)
        self.frame_input.setEnabled(not auto_selected)

    def get_bad_frames(self) -> List[int] :
        """获取用户选择的坏帧列表"""
        if self.auto_radio.isChecked():
            return self.parent().detect_bad_frames_auto_signal.emit(
                self.parent().data.origin,
                self.threshold_spin.value()
            )
        else:
            try:
                return [int(x.strip()) for x in self.frame_input.text().split(",") if x.strip()]
            except ValueError:
                QMessageBox.warning(self, "输入错误", "请输入有效的帧号，用逗号分隔")
                return []

    def apply_fix(self):
        """应用修复并关闭对话框"""
        self.bad_frames = self.get_bad_frames()
        if not self.bad_frames:
            QMessageBox.information(self, "无坏帧", "未检测到需要修复的坏帧")
            return

        n_frames = self.n_frames_spin.value()
        aim_data = self.parent().data

        # 修复数据
        self.parent().fix_bad_frames_signal.emit(
            aim_data,
            self.bad_frames,
            n_frames
        )

        self.accept()
        QMessageBox.information(self, "修复完成", f"已修复 {len(self.bad_frames)} 个坏帧")
        logging.info(f"已修复 {len(self.bad_frames)} 个坏帧")

# 计算设置对话框
class CalculationSetDialog(QDialog):
    def __init__(self,params, parent=None):
        super().__init__(parent)
        self.setWindowTitle("计算设置")
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
        self.setMinimumWidth(350)

        # 默认参数
        self.params = params

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # 计算设置组
        cal_set_group = QGroupBox('计算方法修改')
        cal_set_layout = QFormLayout()
        self.from_start_cal = QCheckBox()
        self.from_start_cal.setChecked(self.params['from_start_cal'])
        cal_set_layout.addRow(QLabel('从头拟合(默认为从最大值拟合)'),self.from_start_cal)
        cal_set_group.setLayout(cal_set_layout)

        # R方设置组
        r2_group = QGroupBox("拟合质量筛选")
        r2_layout = QFormLayout()

        self.r2_spin = QDoubleSpinBox()
        self.r2_spin.setRange(0.0, 1.0)
        self.r2_spin.setValue(self.params['r_squared_min'])
        self.r2_spin.setSingleStep(0.01)
        self.r2_spin.setDecimals(3)

        r2_layout.addRow(QLabel("R²最小值:"), self.r2_spin)
        r2_group.setLayout(r2_layout)

        # 信号范围组
        peak_group = QGroupBox("信号幅值范围")
        peak_layout = QFormLayout()

        self.peak_min_spin = QDoubleSpinBox()
        self.peak_min_spin.setRange(-1e8, 1e2)
        self.peak_min_spin.setValue(self.params['peak_min'])
        self.peak_min_spin.setSingleStep(0.1)

        self.peak_max_spin = QDoubleSpinBox()
        self.peak_max_spin.setRange(-1e2, 1e8)
        self.peak_max_spin.setValue(self.params['peak_max'])
        self.peak_max_spin.setSingleStep(0.1)

        peak_layout.addRow(QLabel("最小值:"), self.peak_min_spin)
        peak_layout.addRow(QLabel("最大值:"), self.peak_max_spin)
        peak_group.setLayout(peak_layout)

        # 寿命范围组
        tau_group = QGroupBox("寿命τ值范围 (ps)")
        tau_layout = QFormLayout()

        self.tau_min_spin = QDoubleSpinBox()
        self.tau_min_spin.setRange(0e-6, 1e6)
        self.tau_min_spin.setValue(self.params['tau_min'])
        self.tau_min_spin.setSingleStep(1e-3)
        self.tau_min_spin.setDecimals(6)

        self.tau_max_spin = QDoubleSpinBox()
        self.tau_max_spin.setRange(1e-6, 1e10)
        self.tau_max_spin.setValue(self.params['tau_max'])
        self.tau_max_spin.setSingleStep(1e2)
        self.tau_max_spin.setDecimals(6)

        tau_layout.addRow(QLabel("最小值:"), self.tau_min_spin)
        tau_layout.addRow(QLabel("最大值:"), self.tau_max_spin)
        tau_group.setLayout(tau_layout)

        # 按钮组
        button_layout = QHBoxLayout()
        self.apply_btn = QPushButton("应用")
        self.apply_btn.clicked.connect(self.apply_settings)
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.clicked.connect(self.reject)

        button_layout.addWidget(self.apply_btn)
        button_layout.addWidget(self.cancel_btn)

        # 组装布局
        layout.addWidget(cal_set_group)
        layout.addWidget(r2_group)
        layout.addWidget(peak_group)
        layout.addWidget(tau_group)
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def apply_settings(self):
        """收集参数并关闭对话框"""
        self.params = {
            'from_start_cal': self.from_start_cal.isChecked(),
            'r_squared_min': self.r2_spin.value(),
            'peak_min': self.peak_min_spin.value(),
            'peak_max': self.peak_max_spin.value(),
            'tau_min': self.tau_min_spin.value(),
            'tau_max': self.tau_max_spin.value()
        }
        self.accept()

# 绘图设置对话框
class PltSettingsDialog(QDialog):
    def __init__(self, params,parent=None):
        super().__init__(parent)
        self.setWindowTitle("绘图设置")
        self.setMinimumWidth(400)
        self.setMinimumHeight(500)

        # 默认参数
        self.params = params
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # 绘图类型选择
        type_group = QGroupBox("绘图类型")
        type_layout = QHBoxLayout()

        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems(["寿命热图", "区域寿命曲线"])
        self.plot_type_combo.currentIndexChanged.connect(self.update_ui)

        type_layout.addWidget(QLabel("绘图模式:"))
        type_layout.addWidget(self.plot_type_combo)
        type_layout.addStretch()
        type_group.setLayout(type_layout)

        # 通用绘图设置
        common_group = QGroupBox("通用设置")
        common_layout = QFormLayout()

        self.color_btn = QPushButton()
        self.color_btn.setStyleSheet(f"background-color: {self.params['color']}")
        self.color_btn.clicked.connect(self.choose_color)

        self.grid_check = QCheckBox()
        self.grid_check.setChecked(self.params['show_grid'])

        self.axis_set = QCheckBox()
        self.axis_set.setChecked(self.params['set_axis'])

        common_layout.addRow(QLabel("线条颜色:"), self.color_btn)
        common_layout.addRow(QLabel("显示网格:"), self.grid_check)
        common_layout.addRow(QLabel('设置轴范围'), self.axis_set)
        common_group.setLayout(common_layout)

        # 曲线图特有设置
        self.curve_group = QGroupBox("曲线图设置")
        curve_layout = QFormLayout()

        self.line_style_combo = QComboBox()
        self.line_style_combo.addItems(["实线 -", "虚线 --", "点线 :", "点划线 -."])
        self.line_style_combo.setCurrentIndex(1)

        self.line_width_spin = QSpinBox()
        self.line_width_spin.setRange(1, 10)
        self.line_width_spin.setValue(self.params['line_width'])

        self.marker_combo = QComboBox()
        self.marker_combo.addItems(["无", "圆形 o", "方形 s", "三角形 ^", "星号 *"])
        self.marker_combo.setCurrentIndex(2)

        self.marker_size_spin = QSpinBox()
        self.marker_size_spin.setRange(1, 20)
        self.marker_size_spin.setValue(self.params['marker_size'])

        curve_layout.addRow(QLabel("线条样式:"), self.line_style_combo)
        curve_layout.addRow(QLabel("线条宽度:"), self.line_width_spin)
        curve_layout.addRow(QLabel("标记样式:"), self.marker_combo)
        curve_layout.addRow(QLabel("标记大小:"), self.marker_size_spin)
        self.curve_group.setLayout(curve_layout)

        # 热图特有设置
        self.heatmap_group = QGroupBox("热图设置")
        heatmap_layout = QFormLayout()

        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(["Jet", "Plasma", "Inferno", "Magma", "Viridis"])

        self.contour_spin = QSpinBox()
        self.contour_spin.setRange(0, 50)
        self.contour_spin.setValue(self.params['contour_levels'])
        self.contour_spin.setSpecialValueText("无等高线")

        heatmap_layout.addRow(QLabel("颜色映射:"), self.cmap_combo)
        heatmap_layout.addRow(QLabel("等高线级别:"), self.contour_spin)
        self.heatmap_group.setLayout(heatmap_layout)

        # 按钮组
        button_layout = QHBoxLayout()
        self.apply_btn = QPushButton("应用")
        self.apply_btn.clicked.connect(self.apply_settings)
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.clicked.connect(self.reject)

        button_layout.addWidget(self.apply_btn)
        button_layout.addWidget(self.cancel_btn)

        # 组装布局
        layout.addWidget(type_group)
        layout.addWidget(common_group)
        layout.addWidget(self.curve_group)
        layout.addWidget(self.heatmap_group)
        layout.addLayout(button_layout)

        self.setLayout(layout)
        self.update_ui()  # 初始化UI状态

    def update_ui(self):
        """根据绘图类型显示/隐藏相关设置"""
        is_curve = self.plot_type_combo.currentText() == "区域寿命曲线"
        self.curve_group.setVisible(is_curve)
        self.heatmap_group.setVisible(not is_curve)

    def choose_color(self):
        """选择颜色"""
        color = QColorDialog.getColor(QColor(self.params['color']), self)
        if color.isValid():
            self.params['color'] = color.name()
            self.color_btn.setStyleSheet(f"background-color: {self.params['color']}")

    def apply_settings(self):
        """收集参数并关闭对话框"""
        self.params = {
            'current_mode': 'curve' if self.plot_type_combo.currentText() == "区域寿命曲线" else 'heatmap',
            'line_style': ['-', '--', ':', '-.'][self.line_style_combo.currentIndex()],
            'line_width': self.line_width_spin.value(),
            'marker_style': ['', 'o', 's', '^', '*'][self.marker_combo.currentIndex()],
            'marker_size': self.marker_size_spin.value(),
            'color': self.params['color'],
            'show_grid': self.grid_check.isChecked(),
            'heatmap_cmap': ['jet', 'plasma', 'inferno', 'magma', 'viridis'][self.cmap_combo.currentIndex()],
            'contour_levels': self.contour_spin.value(),
            'set_axis':self.axis_set.isChecked()
        }
        self.accept()

# 数据保存弹窗
class DataSavingPop(QDialog):
    def __init__(self,parent = None):
        super().__init__(parent)
        self.setWindowTitle("数据保存")
        self.setMinimumWidth(300)
        self.setMinimumHeight(200)

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # 创建水平布局容器来放置标签和复选框
        def create_checkbox_row(label_text, checkbox):
            row_layout = QHBoxLayout()
            label = QLabel(label_text)
            row_layout.addWidget(label)
            row_layout.addStretch()  # 添加弹性空间使复选框靠右
            row_layout.addWidget(checkbox)
            return row_layout

        # 是否拟合
        self.fitting_check = QCheckBox()
        self.fitting_check.setChecked(True)
        # 是否加标题
        self.index_check = QCheckBox()
        self.index_check.setChecked(False)
        # 是否显示额外信息（未完成）
        self.extra_check = QCheckBox()
        self.extra_check.setChecked(False)

        # 添加各个复选框行
        layout.addLayout(create_checkbox_row("是否导出拟合数据:", self.fitting_check))
        layout.addLayout(create_checkbox_row("是否导出标题:(utf-8编码)", self.index_check))
        layout.addLayout(create_checkbox_row("是否显示额外信息（未完成）:", self.extra_check))

        button_layout = QHBoxLayout()
        self.apply_btn = QPushButton("导出")
        self.apply_btn.clicked.connect(self.accept)
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.apply_btn)
        button_layout.addWidget(self.cancel_btn)
        layout.addLayout(button_layout)

        self.setLayout(layout)

# 计算stft参数弹窗
class STFTComputePop(QDialog):
    def __init__(self,params,case,parent = None,time_length = None):
        super().__init__(parent)
        self.setWindowTitle("短时傅里叶变换")
        self.setMinimumWidth(300)
        self.setMinimumHeight(200)
        self.params = params
        self.help_dialog = None
        self.case = case
        self.time_length = time_length
        self.init_ui()

    def init_ui(self):
        layout = QFormLayout()
        row = 6
        self.target_freq_input = QDoubleSpinBox()
        self.target_freq_input.setRange(0.1, 10000)
        self.target_freq_input.setValue(self.params['target_freq'])
        self.target_freq_input.setSuffix(" Hz")

        self.fps_input = QSpinBox()
        self.fps_input.setRange(10,99999)
        self.fps_input.setValue(self.params['EM_fps'])

        self.scale_range_input = QSpinBox()
        self.scale_range_input.setRange(0,99999)
        self.scale_range_input.setValue(self.params['stft_scale_range'])

        self.window_size_input = QSpinBox()
        self.window_size_input.setRange(1, 65536)
        self.window_size_input.setValue(self.params['stft_window_size'])

        self.noverlap_input = QSpinBox()
        self.noverlap_input.setRange(0, 65536)
        self.noverlap_input.setValue(self.params['stft_noverlap'])

        self.custom_nfft_input = QSpinBox()
        self.custom_nfft_input.setRange(0, 65536)
        self.custom_nfft_input.setValue(self.params['custom_nfft'])

        layout.addRow(QLabel("目标频率"),self.target_freq_input)
        layout.addRow(QLabel("平均范围"),self.scale_range_input)
        layout.addRow(QLabel("采样帧率"),self.fps_input)
        layout.addRow(QLabel("窗口大小"),self.window_size_input)
        layout.addRow(QLabel("窗口重叠"),self.noverlap_input)
        layout.addRow(QLabel("变换长度"),self.custom_nfft_input)

        if self.case == 'process':
            row = 10
            self.multiprocess_check = QCheckBox()
            layout.addRow(QLabel("启用加速"),self.multiprocess_check)
            self.hint_label = QLabel("建议启用时关闭无用程序")
            layout.addRow(QLabel("启用警告："),self.hint_label)
            self.multiprocess_check.toggled.connect(self.multiprocess_handle)

            self.batch_size_input = QSpinBox()
            self.batch_size_input.setRange(0,10000)
            self.batch_size_input.setValue(0)
            layout.addRow(QLabel("批处理大小"),self.batch_size_input)
            self.noverlap_input.valueChanged.connect(self._batch_size_cal)
            self.custom_nfft_input.valueChanged.connect(self._batch_size_cal)
            self.window_size_input.valueChanged.connect(self._batch_size_cal)
            self.batch_size_input.setEnabled(False)

            self.cpu_use_input = QSpinBox()
            self.cpu_use_input.setRange(0,100)
            self.cpu_use_input.setValue(0)
            layout.addRow(QLabel("加速核数"),self.cpu_use_input)
            self.cpu_use_input.setEnabled(False)

        button_layout = QHBoxLayout()
        self.apply_btn = QPushButton("执行STFT")
        self.apply_btn.clicked.connect(self.accept)
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.apply_btn)
        button_layout.addWidget(self.cancel_btn)
        layout.setLayout(row,QFormLayout.FieldRole,button_layout)

        self.setLayout(layout)

    def event(self, event):
        if event.type() == QEvent.EnterWhatsThisMode:
            # QWhatsThis.leaveWhatsThisMode()
            QTimer.singleShot(0, self.show_custom_help)
            return True
        return super().event(event)

    def show_custom_help(self):
        """显示自定义非模态帮助对话框"""
        QWhatsThis.leaveWhatsThisMode()
        help_title = "STFT 帮助说明"

        # 创建并显示自定义对话框
        self.help_dialog = CustomHelpDialog(help_title, ['stft'])
        self.help_dialog.setWindowModality(Qt.NonModal)
        self.help_dialog.show()  # 非阻塞显示
        self.help_dialog.activateWindow()
        self.help_dialog.raise_()

    def multiprocess_handle(self):
        """多进程加速启用"""
        multipro = self.multiprocess_check.isChecked()
        if multipro:
            self.batch_size_input.setEnabled(True)
            self.hint_label.setText("启用后其他软件卡顿属正常现象")
            self.cpu_use_input.setEnabled(True)
            self._batch_size_cal()
            self.cpu_use_input.setValue(ToolBucket.available_cpu_count()[1])
            self.cpu_use_input.setSuffix(f"/{ToolBucket.available_cpu_count()[0]}")
        else:
            self.hint_label.setText("建议启用时关闭无用程序")
            self.batch_size_input.setEnabled(False)
            self.cpu_use_input.setEnabled(False)

    def _batch_size_cal(self):
        """动态计算批处理大小Size"""
        custom_nfft =  self.custom_nfft_input.value()
        window_size = self.window_size_input.value()
        noverlap = self.noverlap_input.value()

        # 1. 计算频率轴长度 (Freq Bins)
        n_freqs = custom_nfft // 2 + 1

        # 2. 计算步长 (Hop Size)
        hop_size = window_size - noverlap
        if hop_size < 1:
            self.noverlap_input.setValue(window_size - 1)

        # 3. 估算时间轴长度 (Time Steps)
        # 加上 padding 带来的额外几帧，这里多算一点作为安全冗余 (+5)
        n_time_steps = ceil(self.time_length / hop_size) + 5

        # 4. 计算单个像素 STFT 结果占用的字节数 (Complex64 = 8 bytes)
        bytes_per_pixel = n_freqs * n_time_steps * 8

        # 5. 设定内存安全阈值 (例如 800 MB)
        mem_info = psutil.virtual_memory()
        available_ram = mem_info.available
        SAFE_MEMORY_LIMIT = min(available_ram * 0.5, 2 * 1024 * 1024 * 1024)
        # 意味着我们每次循环处理产生的数据量控制在 800MB 以内，这对大多数电脑都很轻松
        TARGET_BLOCK_SIZE = 8 * 1024 * 1024  # 8 MB

        # 3. 计算最佳 Batch Size
        optimal_batch_size = int(TARGET_BLOCK_SIZE / bytes_per_pixel)
        # 6. 算出 Batch Size
        # 兜底：至少处理1个像素，如果连1个都存不下，那是硬件问题了
        batch_size = max(1, int(SAFE_MEMORY_LIMIT / bytes_per_pixel))
        batch_size = max(1, optimal_batch_size)
        self.batch_size_input.setValue(batch_size)

# 计算cwt参数弹窗
class CWTComputePop(QDialog):
    def __init__(self,params,case='quality',parent = None):
        super().__init__(parent)
        self.setWindowTitle("小波变换")
        self.setMinimumWidth(300)
        self.setMinimumHeight(200)
        self.params = params
        self.case = case
        self.init_ui()

    def init_ui(self):
        layout = QFormLayout()

        self.target_freq_input = QDoubleSpinBox()
        self.target_freq_input.setRange(0.1, 1000)
        self.target_freq_input.setValue(self.params['target_freq'])
        self.target_freq_input.setSuffix(" Hz")

        self.fps_input = QSpinBox()
        self.fps_input.setRange(100,99999)
        self.fps_input.setValue(self.params['EM_fps'])

        self.cwt_size_input = QSpinBox()
        self.cwt_size_input.setRange(0, 65536)
        if self.case == 'quality':
            self.cwt_size_input.setValue(256)
        else:
            self.cwt_size_input.setValue(1)

        self.wavelet = QComboBox()
        self.wavelet.addItems(['cmor1-1.0','cmor1.5-1.0','cmor3-3','cmor8-3 ','cgau8','mexh','morl'])
        self.wavelet.setCurrentText(self.params['cwt_type'])

        layout.addRow(QLabel("目标频率"),self.target_freq_input)
        layout.addRow(QLabel("小波类型"),self.wavelet)
        layout.addRow(QLabel("采样帧率"),self.fps_input)
        layout.addRow(QLabel("计算尺度"),self.cwt_size_input)

        self.cwt_scale_range = QDoubleSpinBox()
        self.cwt_scale_range.setRange(0, 10000)
        self.cwt_scale_range.setValue(self.params['cwt_scale_range'])
        self.cwt_scale_range.setSuffix(" Hz")
        layout.addRow(QLabel("处理跨度"), self.cwt_scale_range)

        if self.case == 'signal':
            self.apply_btn = QPushButton("执行CWT")
        else:
            self.apply_btn = QPushButton("执行质量评价")

        button_layout = QHBoxLayout()
        self.apply_btn.clicked.connect(self.accept)
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.apply_btn)
        button_layout.addWidget(self.cancel_btn)
        layout.setLayout(5,QFormLayout.FieldRole,button_layout)

        self.setLayout(layout)

# 单通道信号参数弹窗
class SCSComputePop(QDialog):
    def __init__(self,params,parent = None):
        super().__init__(parent)
        self.setWindowTitle("单通道模式")
        self.setMinimumWidth(300)
        self.setMinimumHeight(120)
        self.params = params
        self.init_ui()

    def init_ui(self):
        layout = QFormLayout()

        self.thr_known_check = QCheckBox()
        self.thr_known_check.setChecked(self.params['thr_known'])

        self.thr_input = QDoubleSpinBox()
        self.thr_input.setRange(0.1, 1000)
        self.thr_input.setValue(self.params['scs_thr'])

        self.zoom_input = QSpinBox()
        self.zoom_input.setRange(0,100)
        self.zoom_input.setValue(self.params['scs_zoom'])


        layout.addRow(QLabel("阈值是否已知"), self.thr_known_check)
        layout.addRow(QLabel("阈值设置"),self.thr_input)
        layout.addRow(QLabel("插值倍数"),self.zoom_input)
        layout.setSpacing(10)


        button_layout = QHBoxLayout()
        self.apply_btn = QPushButton("执行")
        self.apply_btn.clicked.connect(self.accept)
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.apply_btn)
        button_layout.addWidget(self.cancel_btn)
        layout.setLayout(3,QFormLayout.FieldRole,button_layout)

        self.setLayout(layout)

        self.thr_known_check.toggled.connect(self.update_thr_state)
        self.update_thr_state()

    def update_thr_state(self):
        thr_known = self.thr_known_check.isChecked()
        self.thr_input.setEnabled(thr_known)

# 视频与彩色图像导出
class DataExportDialog(QDialog):
    def __init__(self, parent=None, datatypes=None,export_type='EM',canvas_info=None,is_temporal = None):
        super().__init__(parent)
        self.directory = None
        if export_type == 'EM':
            self.setWindowTitle("数据导出(EM模式)")
        else:
            self.setWindowTitle("数据导出(画布模式)")
        self.setMinimumWidth(300)  # 加宽以适应新控件
        self.setMinimumHeight(450)
        # self.datatypes = datatypes if datatypes else []
        self.is_temporal = is_temporal
        self.datatypes = ['tif','avi','gif','png','plt']
        self.canvas_info = canvas_info
        self.export_type = export_type
        self.current_type = 'tif'

        self.init_ui()
        # self.setStyleSheet(self.style_sheet())  # 设置样式表

    def style_sheet(self):
        """返回美化界面的样式表"""
        return """
            QLineEdit, QComboBox {
                background-color: white;
                border: 1px solid #ced4da;
                border-radius: 4px;
                padding: 5px;
                min-height: 25px;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 3px;
                padding: 4px 6px;
                font-weight: 450;
                min-width: 50px;
            }
            QPushButton#cancel {
                background-color: #6c757d;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton#cancel:hover {
                background-color: #5a6268;
            }
        """

    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # 0. 画布选择（画布导出模式）
        form_layout = QFormLayout()
        if self.export_type == 'canvas':
            self.canvas_selector = QComboBox()
            if self.canvas_info:
                for item in self.canvas_info:
                    self.canvas_selector.addItem(item)
            form_layout.addRow(QLabel("画布选择"),self.canvas_selector)
            self.canvas_selector.currentIndexChanged.connect(self.datatypes_change)
        else:
            pass

        # 1. 路径选择组件

        path_label = QLabel("保存路径:")
        self.path_edit = QLineEdit()
        form_layout.addRow(path_label,self.path_edit)
        self.path_edit.setPlaceholderText("选择或输入文件保存路径")
        browse_btn = QPushButton("浏览文件夹")
        browse_btn.clicked.connect(self.browse_directory)
        form_layout.addRow(QLabel(""),browse_btn)

        # 2. 文本输入框
        text_label = QLabel("文件名称:")
        self.text_edit = QLineEdit()
        self.text_edit.setPlaceholderText("请输入文件名（前缀）")
        form_layout.addRow(text_label,self.text_edit)

        # 3. 动态数据类型选择器
        type_label = QLabel("数据类型:")
        self.type_combo = QComboBox()
        self.update_datatype(self.datatypes)  # 初始化下拉菜单
        form_layout.addRow(type_label, self.type_combo)
        self.type_combo.currentIndexChanged.connect(self._update_type)

        # 4. 其他参数
        self.duration_label = QLabel("视频时长：")
        self.duration_input = QSpinBox()
        self.duration_input.setRange(0,10000)
        self.duration_input.setValue(60)
        form_layout.addRow(self.duration_label, self.duration_input)
        self.duration_label.setVisible(False)
        self.duration_input.setVisible(False)
        self.title_input = QLineEdit()
        self.title_input.setPlaceholderText("导出图的title，可留空")
        self.title_label = QLabel("导图标题：")
        form_layout.addRow(self.title_label , self.title_input)
        self.title_input.setVisible(False)
        self.title_label.setVisible(False)
        self.colorbar_label_label = QLabel("彩棒标签：")
        self.colorbar_label_input = QLineEdit()
        self.colorbar_label_input.setPlaceholderText("导出图的右侧的标签，可留空")
        form_layout.addRow(self.colorbar_label_label , self.colorbar_label_input)
        self.colorbar_label_label.setVisible(False)
        self.colorbar_label_input.setVisible(False)

        self.info_label = QLabel('注意：使用avi/gif/png，会对原始数据有压缩！')
        main_layout.addLayout(form_layout)
        main_layout.addWidget(self.info_label)

        # 5. 确认/取消按钮
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        cancel_btn = QPushButton("取消")
        cancel_btn.setObjectName("cancel")
        cancel_btn.clicked.connect(self.reject)
        confirm_btn = QPushButton("确认导出")
        confirm_btn.clicked.connect(self.accept)

        btn_layout.addWidget(confirm_btn)
        btn_layout.addWidget(cancel_btn)

        main_layout.addLayout(btn_layout)
        self.setLayout(main_layout)

    def datatypes_change(self):
        if self.is_temporal[self.canvas_selector.currentIndex()]:
            self.datatypes = ['tif','avi','gif','png','plt']
            self.update_datatype(self.datatypes)
        if not self.is_temporal[self.canvas_selector.currentIndex()]:
            self.datatypes = ['tif', 'png', 'plt']
            self.update_datatype(self.datatypes)

    def update_datatype(self, datatypes):
        """动态更新数据类型下拉菜单"""
        self.type_combo.clear()
        if datatypes:
            self.type_combo.addItems(datatypes)
        else:
            self.type_combo.addItem("无可用格式")
            self.type_combo.setEnabled(False)

    def _update_type(self):
        """选择到处类型后发生什么"""
        self.current_type = self.type_combo.currentText()
        if self.current_type == 'gif' or self.current_type == 'avi':
            self.duration_label.setVisible(True)
            self.duration_input.setVisible(True)
            self.title_input.setVisible(False)
            self.title_label.setVisible(False)
            self.colorbar_label_label.setVisible(False)
            self.colorbar_label_input.setVisible(False)
            self.info_label.setText('注意：使用avi/gif/png，会对原始数据有压缩！')
        elif self.current_type == 'plt':
            self.duration_label.setVisible(False)
            self.duration_input.setVisible(False)
            self.title_input.setVisible(True)
            self.title_label.setVisible(True)
            self.colorbar_label_label.setVisible(True)
            self.colorbar_label_input.setVisible(True)
            self.info_label.setText("本方法是用于导出带colorbar结果的tif图")
        else:
            self.duration_label.setVisible(False)
            self.duration_input.setVisible(False)
            self.title_input.setVisible(False)
            self.title_label.setVisible(False)
            self.colorbar_label_label.setVisible(False)
            self.colorbar_label_input.setVisible(False)
            self.info_label.setText('注意：使用avi/gif/png，会对原始数据有压缩！')
            return

    def browse_directory(self):
        """打开文件夹选择对话框"""
        self.directory = QFileDialog.getExistingDirectory(
            self,
            "选择保存文件夹",
            options=QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        if self.directory:
            self.path_edit.setText(self.directory)

    def get_values(self):
        """获取用户输入的值"""
        return {
            # 'canvas': self.canvas_selector.currentIndex() if self.export_type == 'canvas' else None,
            # 'path': self.path_edit.text().strip(),
            # 'filename': self.text_edit.text().strip(),
            # 'filetype': self.type_combo.currentText(),
            'duration': self.duration_input.value() if self.current_type in ['gif','avi'] else None,
            'title': self.title_input.text() if self.current_type in ['plt'] else None,
            'colorbar_label': self.colorbar_label_input.text() if self.current_type in ['plt'] else None,
        }

# 数据选择查看视窗
class DataViewAndSelectPop(QDialog):
    def __init__(self, parent=None, datadict=None, processed_datadict=None, add_canvas=False):
        super().__init__(parent)
        self.datadict = datadict or []
        self.processed_datadict = processed_datadict or []
        self.add_canvas = add_canvas

        self.selected_timestamp = None
        self.selected_index = -1
        self.selected_name = ""
        self.selected_table = None  # 记录选择来自哪个表格

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('数据选择')
        self.setMinimumSize(800, 500)  # 增加对话框尺寸以适应多个表格

        # 创建主布局
        main_layout = QVBoxLayout(self)

        # 创建选项卡容器
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget, 3)  # 选项卡占据大部分空间

        # 根据传入的数据创建表格
        self.tables = []

        if self.datadict != []:
            self.create_table_tab(self.datadict, "原始数据")

        if self.processed_datadict != []:
            self.create_table_tab(self.processed_datadict, "处理后数据")

        # 如果没有数据，显示提示信息
        if not self.tables:
            no_data_label = QLabel("没有可显示的数据")
            no_data_label.setAlignment(Qt.AlignCenter)
            self.tab_widget.addTab(no_data_label, "无数据")

        # 创建底部状态显示区域
        bottom_layout = QHBoxLayout()
        self.status_label = QLabel("目前选择的数据：")
        self.selected_data_label = QLabel("暂无")
        bottom_layout.addWidget(self.status_label)
        bottom_layout.addWidget(self.selected_data_label)
        bottom_layout.addStretch()

        main_layout.addLayout(bottom_layout)

        # 创建按钮框
        self.button_box = QDialogButtonBox(QDialogButtonBox.Cancel, Qt.Horizontal, self)

        # 根据add_canvas设置按钮状态(现在不要这个按钮了)
        # if self.add_canvas:
        #     self.button_box.button(QDialogButtonBox.Ok).setEnabled(False)
        #     self.button_box.button(QDialogButtonBox.Ok).setText("确定")
        # else:
        #     self.button_box.button(QDialogButtonBox.Ok).setEnabled(True)
        #     self.button_box.button(QDialogButtonBox.Ok).setText("选择")

        main_layout.addWidget(self.button_box)

        # 连接按钮信号
        # self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

    def create_table_tab(self, data_list, tab_name):
        """创建表格并添加到选项卡"""
        tab = QWidget()
        tab_layout = QVBoxLayout(tab)

        table = QTableWidget()
        tab_layout.addWidget(table)

        # 配置表格
        self.setup_table(table, data_list)

        # 添加到选项卡
        self.tab_widget.addTab(tab, tab_name)
        self.tables.append(table)

        # 排序功能
        table.setSortingEnabled(True)
        table.horizontalHeader().setSortIndicatorShown(True)

        return table

    def setup_table(self, table, data_list):
        """设置表格内容和按钮"""
        num_rows = len(data_list)
        if num_rows > 0:
            all_keys = list(data_list[0].keys())
            keys = [key for key in all_keys if key not in ['timestamp']]
            num_cols = len(keys) + 1  # 增加一列用于放置按钮
        else:
            keys = []
            num_cols = 0

        # 设置表格行数、列数和表头
        table.setRowCount(num_rows)
        table.setColumnCount(num_cols)
        if num_rows > 0:
            column_headers = keys + ['操作']  # 添加一个"操作"列
            table.setHorizontalHeaderLabels(column_headers)

        # 填充数据并插入按钮
        for row_idx, data_dict in enumerate(data_list):
            for col_idx, key in enumerate(keys):
                value = str(data_dict.get(key, ''))
                item = QTableWidgetItem(value)
                item.setToolTip(value)
                item.setTextAlignment(Qt.AlignCenter)
                table.setItem(row_idx, col_idx, item)

            # 在最后一列创建并设置按钮
            button_text = "显示选择" if self.add_canvas else "设为当前"
            button = QPushButton(button_text)

            # 使用lambda表达式捕获当前行索引和表格
            button.clicked.connect(lambda checked, r=row_idx, t=table: self.on_row_button_clicked(r, t))
            table.setCellWidget(row_idx, num_cols - 1, button)

        # 调整列宽以自适应内容
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.setSelectionBehavior(QTableWidget.SelectRows)  # 单选整行

        # 连接单元格点击事件
        table.cellClicked.connect(lambda row, col, t=table: self.on_cell_clicked(row, col, t))

    def on_row_button_clicked(self, row_index, table):
        """处理行按钮点击事件"""
        # 确定数据来自哪个表格
        table_index = self.tables.index(table)
        if table_index == 0 and self.datadict != []:
            data_list = self.datadict
            self.selected_table = 'data'
        else:
            data_list = self.processed_datadict
            self.selected_table = 'processed_data'

        self.selected_index = row_index
        selected_data = data_list[row_index]

        # 获取名称和时间戳
        name = selected_data.get('name')
        self.selected_name = str(name) if name else f"数据{row_index + 1}"
        self.selected_timestamp = selected_data.get('timestamp')

        # 更新状态显示
        self.selected_data_label.setText(self.selected_name)

        self.accept()

    def on_cell_clicked(self, row_index, col_index, table):
        """处理单元格点击事件"""
        # 只处理数据列的点击，忽略按钮列的点击
        if col_index < table.columnCount() - 1:
            self.on_row_button_clicked(row_index, table)

    def on_show_selected(self):
        """当add_canvas为True时，显示选择的画布（留空）"""
        # 这里可以添加显示画布的逻辑
        pass

    def get_selected_timestamp(self):
        """获取选择的数据信息,(timestamp,selected_type(data or processed_data))"""
        return self.selected_timestamp, self.selected_table

# 帮助dialog
class CustomHelpDialog(QDialog):
    """自定义非模态帮助对话框"""
    ALL_TOPICS = ["general","canvas", "stft", "cwt", "lifetime","whole","single"]
    def __init__(self, title, topics=None, content = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowFlags(self.windowFlags() | Qt.WindowMinMaxButtonsHint | Qt.WindowCloseButtonHint)
        self.setAttribute(Qt.WA_DeleteOnClose)  # 关闭时自动释放
        self.htmlpath = self.get_html_path()
        self.HELP_CONTENT = {
    "general": {
        "title": "程序使用指南",
        "html": "main_help"
        },
    "canvas":{
        "title": "成像系统指南",
        "html": "canvas_help",
    },
    "stft": {
        "title": "STFT(短时傅里叶变换)分析",
        "html": "stft_help",
        },
    "cwt": {
        "title": "CWT(连续小波变换)分析",
        "html": 'cwt_help',
        },
    "lifetime": {
        "title": "指数型寿命计算（没写完）",
        "html": "lifetime_help",
        },
    "whole": {
        "title":"全细胞分析（没写完）",
        "html": "whole_help",
        },
    "single": {
        "title":"单通道分析（没写完）",
        "html": "single_help",
        },
    "data_view":{
        "title":"数据查看帮助",
        "html": "data_view_help",
    }
    }
        self.content = content
        # 创建布局和控件
        layout = QVBoxLayout()

        self.tab_widget = QTabWidget()

        # 添加帮助主题
        self.add_help_tabs(topics)

        layout.addWidget(self.tab_widget)

        # 关闭按钮
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)

        self.setLayout(layout)
        self.resize(500, 700)  # 设置初始大小

    @staticmethod
    def get_html_path() -> str:
        """
        获取资源的绝对路径，兼容开发和打包环境
        """
        try:
            # PyInstaller 创建的单文件临时文件夹
            base_path = sys._MEIPASS
        except AttributeError:
            # 开发环境，返回当前工作目录
            base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            base_path = os.path.join(base_path,'core')
        return os.path.join(base_path , 'HTML')

    def add_help_tabs(self, topics):
        """添加指定的帮助主题标签页"""
        # 如果没有指定主题或列表为空，则显示所有主题
        if not topics:
            topics = self.ALL_TOPICS
        elif topics == "custom":
            self.add_tab(topics,self.content)

        # 添加每个主题的标签页
        for topic in topics:
            if topic in self.ALL_TOPICS:
                self.add_tab(topic)
            elif topic in self.HELP_CONTENT:
                self.add_tab(topic)

    def add_tab(self, topic_key, html_text = None):
        """添加单个帮助主题标签页"""
        if topic_key == 'custom':
            text = html_text
            title = '方法帮助'
        elif topic_key not in self.HELP_CONTENT:
            logging.error("impossible fault from help")
            return
        else:
            topic = self.HELP_CONTENT[topic_key]
            html_path = os.path.join(self.htmlpath, topic["html"]+'.html')
            with open(html_path, 'r', encoding='utf-8') as file:
                text = file.read()
            title = topic["title"]
        # 创建文本浏览器
        browser = QTextBrowser()
        browser.setOpenExternalLinks(True)
        browser.setHtml(text)

        # 添加到标签页
        self.tab_widget.addTab(browser, title)

    # def show_topic(self, topic_key):
    #     """显示特定主题并使其成为当前标签页"""
    #     if topic_key not in HelpContentHTML.HELP_CONTENT:
    #         return
    #
    #     # 查找标签页索引
    #     for i in range(self.tab_widget.count()):
    #         if self.tab_widget.tabText(i) == HelpContentHTML.HELP_CONTENT[topic_key]["title"]:
    #             self.tab_widget.setCurrentIndex(i)
    #             return
    #
    #     # 如果没找到，添加新标签页
    #     self.add_tab(topic_key)
    #     self.tab_widget.setCurrentIndex(self.tab_widget.count() - 1)

    # def _render_markdown(self, md_content):
    #     """将Markdown转换为HTML"""
    #     # 添加扩展支持代码高亮和围栏代码块
    #     extensions = [
    #         CodeHiliteExtension(noclasses=True),
    #         FencedCodeExtension()
    #     ]
    #     return markdown.markdown(md_content, extensions=extensions)

# 画布及roi查看和选择
class ROIInfoDialog(QDialog):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.canvas_id = None
        self.roi_type = None
        self.setWindowTitle("图像与ROI信息（双击选择）")
        self.setMinimumSize(600, 400)

        self.parent_window = parent
        self.init_ui()
        self.load_data()

    def init_ui(self):
        layout = QVBoxLayout()

        # 创建表格
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(["画布ID", "图像名称", "图像尺寸", "ROI类型", "ROI详情",'操作'])
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.doubleClicked.connect(self.handle_click)

        layout.addWidget(self.table)

        # 添加按钮
        button_box = QDialogButtonBox(QDialogButtonBox.Cancel)
        # button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setLayout(layout)

    def load_data(self):
        """加载所有画布的信息到表格"""
        self.canvas_info = self.parent_window.get_all_canvas_info()
        self.table.setRowCount(0)

        for info in self.canvas_info:
            # 添加画布基本信息行
            row_position = self.table.rowCount()
            self.table.insertRow(row_position)
            # 创建并设置所有单元格
            items_data = [
                str(info['canvas_id']),
                info['image_name'],
                f"{info['image_size'][1]}×{info['image_size'][0]}",
                "这是画布本身",
                ""
            ]

            for col, text in enumerate(items_data):
                item = QTableWidgetItem(text)
                item.setToolTip(text)  # 始终设置ToolTip
                self.table.setItem(row_position, col, item)

            # 添加ROI信息行
            for roi in info['ROIs']:
                row_position = self.table.rowCount()
                self.table.insertRow(row_position)

                # ROI详情
                if roi['type'] == 'v_rect':
                    x, y = roi['position']
                    w, h = roi['size']
                    details = f"位置: ({x}, {y}), 尺寸: {w}×{h}"
                elif roi['type'] == 'v_line':
                    x1, y1 = roi['start']
                    x2, y2 = roi['end']
                    details = f"起点: ({x1}, {y1}), 终点: ({x2}, {y2}), 宽度: {roi['width']}"
                elif roi['type'] == 'anchor':
                    x, y = roi['position']
                    details = f"位置: ({x}, {y})"
                elif roi['type'] == 'pixel_roi':
                    n = roi['counts']
                    details = f"选中{n}个像素"
                else:
                    details = "没有ROI"

                roi_items_data = [
                    str(info['canvas_id']),
                    info['image_name'],
                    "",
                    roi['type'],
                    details
                ]
                for col, text in enumerate(roi_items_data):
                    item = QTableWidgetItem(text)
                    item.setToolTip(text)  # 始终设置ToolTip
                    self.table.setItem(row_position, col, item)
                # self.table.setItem(row_position, 4, QTableWidgetItem(details))
                button_text = "ROI设置"
                button = QPushButton(button_text)

                # 使用lambda表达式捕获当前行索引和表格
                button.clicked.connect(lambda checked, r=row_position: self.handle_click(r))
                self.table.setCellWidget(row_position, 5, button)

        # 调整列宽
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)

    def handle_click(self, index: QModelIndex|int):
        """双击行时选择对应的画布"""
        if isinstance(index, QModelIndex):
            row = index.row()
        else:
            row = index
        canvas_id_item = self.table.item(row, 0)
        roi_type_item = self.table.item(row, 3)
        self.roi_type = self.table.item(row, 3).text()
        if not canvas_id_item:
            return None
        self.canvas_id = int(canvas_id_item.text())
        # 查找对应的画布信息
        canvas_info = None
        for info in self.canvas_info:
            if info['canvas_id'] == self.canvas_id:
                canvas_info = info
                break

        # 判断是画布行还是ROI行
        if roi_type_item and roi_type_item.text() != "这是画布本身":
            # 这是ROI行
            roi_type = roi_type_item.text()

            # 查找对应的ROI信息
            roi_info = None
            for roi in canvas_info['ROIs']:
                if roi['type'] == roi_type:
                    # 根据类型和详细信息匹配ROI
                    roi_info = roi
                    break

            if roi_info:
                # 返回ROI和画布信息
                self.selected_roi_info = {
                    'type': roi_type,
                    'canvas_info': {
                        'canvas_id': canvas_info['canvas_id'],
                        'image_name': canvas_info['image_name'],
                        'image_size': canvas_info['image_size']
                    },
                    'roi_info': roi_info
                }
        # self.parent_window.set_cursor_id(canvas_id) # 忘记这个有没有用了

        self.accept()

# ROI编辑处理对话框
class ROIProcessedDialog(QDialog):
    def __init__(self, draw_type, canvas_id, roi, roi_info,data_type = None, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.setWindowTitle("ROI设置对话框")
        self.setMinimumSize(250, 400)
        self.draw_type = draw_type
        self.canvas_id = canvas_id
        self.roi_info = roi_info
        self.data_type = data_type

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.addWidget(QLabel("所选ROI信息："))
        self.infolist = QListWidget()
        self.infolist.setAlternatingRowColors(True)  # 交替行颜色
        self.infolist.setSelectionMode(QListWidget.NoSelection)  # 不可选择
        self.infolist.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        layout.addWidget(self.infolist)

        form_layout = QFormLayout()
        self.crop_check = QCheckBox()
        self.inverse_check = QCheckBox()
        self.reset_value = QDoubleSpinBox()
        self.reset_value.setRange(-1000.0,9999.0)
        self.reset_value.setValue(1)
        self.zoom_check = QCheckBox()
        self.zoom_check.setText("插值放大|倍数")
        self.zoom_check.setEnabled(False)
        self.zoom_factor = QDoubleSpinBox()
        self.zoom_factor.setValue(1)
        self.zoom_factor.setEnabled(False)
        self.fast_check = QCheckBox()
        self.fast_check.setEnabled(False)
        self.crop_check.toggled.connect(lambda checked: self.zoom_check.setEnabled(checked))
        self.zoom_check.toggled.connect(lambda checked: self.zoom_factor.setEnabled(checked))
        form_layout.addRow(QLabel("截取数据:"),self.crop_check)
        form_layout.addRow(QLabel("注释："), QLabel('选择后选区数据\n会被裁剪提取出来'))
        form_layout.addRow(QLabel("选区反转:"),self.inverse_check)
        form_layout.addRow(QLabel("注释："), QLabel('默认操作都是针对选区的，\n该选项会反转选区'))
        form_layout.addRow(QLabel("选区赋值:"),self.reset_value)
        form_layout.addRow(QLabel("注释："), QLabel('为原值的倍数，\n填0相当于去掉原值'))
        form_layout.addRow(QLabel("选区放大："),QLabel("是否进行插值放大以及放大的倍数"))
        form_layout.addRow(self.zoom_check, self.zoom_factor)
        form_layout.addRow(QLabel("便捷操作:"),self.fast_check)
        layout.addLayout(form_layout)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        self.update_list()
        self.setLayout(layout)


    def update_list(self):
        fast_text = ''
        if self.draw_type == 'v_rect':
            type_text = "ROI类型：矢量矩形选区"
            # x, y = self.roi_info['ROIs']['position']
            w, h = self.roi_info['roi_info']['size']
            text1 = f"该ROI尺寸为：{w}×{h} pixels"
            text2 = f"该ROI所属数据：{self.roi_info['canvas_info']['image_name']}"
            crop_text = f"是否支持截取数据：是（注意是否有便捷操作）"
            inverse_text = f"是否支持选区反转：否"
            reset_text = f"是否支持选区赋值：是"
            zoom_text = f"是否支持选区插值放大：是（需截取数据）"
            self.inverse_check.setEnabled(False)
            if self.data_type == 'Accumulated_time_amplitude_map':
                fast_text = f'支持的便捷操作: 单通道计算快速实现（母函数截取数据）'
                self.fast_check.setEnabled(True)
                self.fast_check.setChecked(True)
        elif self.draw_type == 'v_line':
            type_text = "ROI类型：矢量直线选区"
            x1, y1 = self.roi_info['roi_info']['start']
            x2, y2 = self.roi_info['roi_info']['end']
            text1 = f"该ROI起点: ({x1}, {y1}), 终点: ({x2}, {y2}), 宽度: {self.roi_info['roi_info']['width']}"
            text2 = f"该ROI所属数据：{self.roi_info['canvas_info']['image_name']}"
            crop_text = f"是否支持截取数据：否"
            inverse_text = f"是否支持选区反转：否"
            reset_text = f"是否支持选区赋值：否"
            zoom_text = f"是否支持选区插值放大：否"
            fast_text = f'矢量直线目前不支持高级设置，请直接点ok'
            self.reset_value.setEnabled(False)
            self.crop_check.setEnabled(False)
            self.inverse_check.setEnabled(False)
            self.zoom_check.setEnabled(False)
            self.zoom_factor.setEnabled(False)
        else : #self.draw_type == 'pixel_roi'
            type_text = "ROI类型：像素绘制选区"
            # x, y = self.roi_info['ROIs']['position']
            n = self.roi_info['roi_info']['counts']
            text1 = f"该ROI共覆盖：{n}个 pixels"
            text2 = f"该ROI所属数据：{self.roi_info['canvas_info']['image_name']}"
            crop_text = f"是否支持截取数据：是"
            inverse_text = f"是否支持选区反转：是"
            reset_text = f"是否支持选区赋值：是"
            zoom_text = f"是否支持选区插值放大：是"
        self.infolist.addItem(QListWidgetItem(type_text))
        self.infolist.addItem(QListWidgetItem(text1))
        self.infolist.addItem(QListWidgetItem(text2))
        self.infolist.addItem(QListWidgetItem(crop_text))
        self.infolist.addItem(QListWidgetItem(inverse_text))
        self.infolist.addItem(QListWidgetItem(reset_text))
        self.infolist.addItem(QListWidgetItem(zoom_text))
        if fast_text:
            self.infolist.addItem(QListWidgetItem(fast_text))
        self.infolist.adjustSize()

# 伪色彩管理弹窗
class ColorMapDialog(QDialog):
    def __init__(self, parent = None, colormap_list = None, canvas_info=None,params = None):
        super().__init__(parent)
        if canvas_info is None:
            canvas_info = []
        self.params = params
        self.setWindowTitle("伪色彩管理器")
        self.setMinimumWidth(300)
        self.setMinimumHeight(300)
        self.colormap_list = colormap_list
        self.canvas_info = canvas_info
        self.canvas_index = -1
        self.parent_window = parent
        self.imagemin = params['min_value'] if params['min_value'] is not None else 0
        self.imagemax = params['max_value'] if params['min_value'] is not None else 255
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        self.colormap_control_layout = QFormLayout()

        # 区域选择器
        self.canvas_selector = QComboBox()
        self.canvas_selector.addItems(["所有区域"])
        if self.canvas_info:
            for item in self.canvas_info:
                self.canvas_selector.addItem(item)
        self.canvas_selector.currentIndexChanged.connect(self._handle_canvas_change)

        # 伪彩色开关
        self.colormap_toggle = QCheckBox()
        self.colormap_toggle.stateChanged.connect(self._handle_colormap_toggle)

        # 伪彩色方案选择
        self.colormap_selector = QComboBox()
        self.colormap_selector.addItems(self.colormap_list)

        # 边界设置
        self.boundary_set = QComboBox()
        self.boundary_set.addItems(['自动设置','手动设置'])
        self.boundary_set.currentIndexChanged.connect(self._handle_boundary_set)

        self.up_boundary_set = QDoubleSpinBox()
        self.up_boundary_set.setRange(-999999,999999)
        self.up_boundary_set.setDecimals(3)
        self.low_boundary_set = QDoubleSpinBox()
        self.low_boundary_set.setRange(-999999,999999)
        self.low_boundary_set.setDecimals(3)

        # 添加到布局
        self.colormap_control_layout.addRow(QLabel("应用区域:"),self.canvas_selector)
        self.colormap_control_layout.addRow(QLabel("伪彩显示:"),self.colormap_toggle)
        self.colormap_control_layout.addRow(QLabel("配色方案:"),self.colormap_selector)
        self.colormap_control_layout.addRow(QLabel("边界设置:"),self.boundary_set)
        self.colormap_control_layout.addRow(QLabel("上界设置:"),self.up_boundary_set)
        self.colormap_control_layout.addRow(QLabel('下界设置:'),self.low_boundary_set)

        self.colormap_control_layout.setSpacing(10)

        layout.addLayout(self.colormap_control_layout)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setLayout(layout)

        self._handle_colormap_toggle()
        self._handle_boundary_set()

    def _handle_colormap_toggle(self):
        if self.colormap_toggle.isChecked():
            self.colormap_selector.setEnabled(True)
            # self.canvas_selector.setEnabled(True)
            self.boundary_set.setEnabled(True)
        else:
            self.colormap_selector.setEnabled(False)
            # self.canvas_selector.setEnabled(False)
            self.boundary_set.setEnabled(False)

    def _handle_canvas_change(self):
        if self.canvas_selector.currentIndex() >= 1:
            canvas_id = self.canvas_selector.currentIndex()-1
            canvas = self.parent_window.display_canvas[canvas_id]
            self.imagemin = self.parent_window.display_canvas[canvas_id].min_value
            self.imagemax = self.parent_window.display_canvas[canvas_id].max_value
            self.colormap_toggle.setChecked(canvas.use_colormap)
            self.colormap_selector.setCurrentText(canvas.colormap)
            self.up_boundary_set.setValue(self.imagemax)
            self.low_boundary_set.setValue(self.imagemin)
            self.canvas_index = self.canvas_selector.currentIndex() - 1
            if not canvas.auto_boundary_set:
                self.boundary_set.setCurrentIndex(1)
            else:
                self.boundary_set.setCurrentIndex(0)
        else:
            self.canvas_index = -1
            self.colormap_toggle.setChecked(self.params['use_colormap'])
            self.colormap_selector.setCurrentText(self.params['colormap'])
            self.imagemin = self.params['min_value'] if self.params['min_value'] is not None else 0
            self.imagemax = self.params['max_value'] if self.params['min_value'] is not None else 255
            self.up_boundary_set.setValue(self.imagemax)
            self.low_boundary_set.setValue(self.imagemin)
            if not self.params['auto_boundary_set']:
                self.boundary_set.setCurrentIndex(1)
            else:
                self.boundary_set.setCurrentIndex(0)

    def _handle_boundary_set(self):
        if self.boundary_set.currentIndex() == 0:
            self.auto_boundary_set = True
            self.up_boundary_set.setEnabled(False)
            self.low_boundary_set.setEnabled(False)
        else:
            self.auto_boundary_set = False
            self.up_boundary_set.setEnabled(True)
            self.low_boundary_set.setEnabled(True)
            self.up_boundary_set.setValue(self.imagemax)
            self.low_boundary_set.setValue(self.imagemin)

    def get_value(self):
        return {'colormap':self.colormap_selector.currentText() if self.colormap_toggle.isChecked() else None,
            'use_colormap':self.colormap_toggle.isChecked(),
            'auto_boundary_set':self.auto_boundary_set,
            'min_value':self.low_boundary_set.value() if not self.auto_boundary_set else None,
            'max_value':self.up_boundary_set.value() if not self.auto_boundary_set else None,}

# 数据树结构显示
class DataTreeViewDialog(QDialog):
    """.12.2版本从原来专为plot使用的，现在加入多功能。改名为树结构显示，原来叫DataPlotSelectDialog"""
    sig_plot_request = pyqtSignal(np.ndarray,str, object)
    sig_canvas_signal = pyqtSignal(object, str)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("数据流管理与导出")
        self.resize(1100, 500)
        self.setModal(False)  # 设为非模态，方便一边看数据一边操作主界面
        self.setWindowFlags(Qt.Dialog |
                            Qt.WindowCloseButtonHint |
                            Qt.WindowContextHelpButtonHint)
        self.help_window = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # 顶部说明
        header_layout = QHBoxLayout()
        header_layout.addWidget(QLabel("数据层级结构：原始数据 (Data) -> 处理数据 (ProcessedData) -> ..."))

        refresh_btn = QPushButton("刷新列表（收起列表）")
        refresh_btn.clicked.connect(self.refresh_data)
        expand_all_btn = QPushButton("展开列表")
        expand_all_btn.clicked.connect(self.expand_except_parameters_results)
        header_layout.addWidget(refresh_btn)
        header_layout.addWidget(expand_all_btn)

        layout.addLayout(header_layout)

        # 核心控件：QTreeWidget
        self.tree = QTreeWidget()
        self.tree.setColumnCount(6)
        self.tree.setHeaderLabels(["名称 / Key", "类型", "尺寸 & 大小", "数值范围", "创建时 / 值", "操作"])

        # 调整列宽
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.tree.header().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.tree.header().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.tree.header().setSectionResizeMode(4, QHeaderView.ResizeToContents)

        self.tree.setAlternatingRowColors(True)   # 开启交替行颜色（可选，看起来更像表格）
        self.tree.setAnimated(True)              # 开启展开收起的动画
        self.tree.setIndentation(20)            # 设置缩进宽度

        layout.addWidget(self.tree)

    def refresh_data(self):
        self.tree.clear()

        # 1. 获取所有数据
        data_history = Data.get_history_list()
        processed_data_history = ProcessedData.get_history_list()

        # 2. 建立节点映射表 { timestamp_float: QTreeWidgetItem }
        # 用于通过 timestamp 快速（或遍历）找到父节点的 TreeItem
        self.node_map = {}

        # --- 第一步：加载所有原始 Data (作为根节点) ---
        # 倒序显示，让最新的在最上面（符合直觉），但在构建Map时要注意顺序
        # 为了逻辑顺畅，我们先建立好所有的Data节点
        for data_obj in reversed(data_history):
            root_item = QTreeWidgetItem(self.tree)
            self._setup_data_item(root_item, data_obj)

            # 添加 Parameters
            if data_obj.parameters:
                param_node = QTreeWidgetItem(root_item)
                param_node.setText(0, "⚙️ Parameters")
                self._fill_dict_items(param_node, data_obj.parameters, data_obj)

            # 记录到 Map 中，供后续 ProcessedData 查找父节点
            self.node_map[data_obj.timestamp] = root_item

        # --- 第二步：加载 ProcessedData (支持多层嵌套) ---

        # 关键点：必须按【创建时间正序】排序。
        # 这样保证在处理 "子ProcessedData" 时，它的 "父ProcessedData" 已经被创建并加入到 self.node_map 中了。
        sorted_processed = sorted(processed_data_history, key=lambda x: getattr(x, 'timestamp', 0))

        orphan_processed = []  # 记录找不到爹的孤儿数据

        for proc_obj in sorted_processed:
            # 1. 寻找父节点
            parent_ts = proc_obj.timestamp_inherited
            parent_item = None

            # 由于浮点数精度问题，不能直接 dict.get(float)，需要模糊匹配
            # 优化：如果数据量极大，建议将 timestamp 格式化字符串作为 key
            # 这里采用遍历匹配 (对于UI显示的数据量级通常没问题)
            for ts, item in self.node_map.items():
                if abs(ts - parent_ts) < 1e-6:
                    parent_item = item
                    break

            if parent_item:
                # 2. 找到了父节点（可能是 Data，也可能是之前添加的 ProcessedData）
                proc_item = QTreeWidgetItem(parent_item)
                self._setup_processed_item(proc_item, proc_obj)

                # 添加 Out Processed Results
                if proc_obj.out_processed:
                    out_node = QTreeWidgetItem(proc_item)
                    out_node.setText(0, "⚙️ Other Results")
                    self._fill_dict_items(out_node, proc_obj.out_processed, proc_obj)

                # 3. 重要：将当前 ProcessedData 也加入 Map
                # 这样后续的数据如果是基于它的，就可以把它当做父节点
                if hasattr(proc_obj, 'timestamp'):
                    self.node_map[proc_obj.timestamp] = proc_item
            else:
                # 没找到父节点，暂时放入孤儿列表
                orphan_processed.append(proc_obj)

        # --- 第三步：处理孤儿数据 (原始数据已被删除或丢失) ---
        if orphan_processed:
            orphan_root = QTreeWidgetItem(self.tree)
            orphan_root.setText(0, "历史处理记录 (无关联源数据)")
            # 设置颜色提示
            # orphan_root.setForeground(0, QBrush(Qt.GlobalColor.gray))
            orphan_root.setExpanded(True)

            for proc_obj in orphan_processed:
                # 注意：这里孤儿内部如果也有嵌套关系，上面的逻辑因为找不到第一级父节点，
                # 后续子节点也会掉入 orphan_processed。
                # 在孤儿区简单平铺显示，或者也可以再做一次递归，视需求而定。
                # 这里做简单平铺处理：
                proc_item = QTreeWidgetItem(orphan_root)
                self._setup_processed_item(proc_item, proc_obj)

                if proc_obj.out_processed:
                    out_node = QTreeWidgetItem(proc_item)
                    out_node.setText(0, "⚙️ Out Processed Results")
                    self._fill_dict_items(out_node, proc_obj.out_processed, proc_obj)

        self.tree.expandToDepth(0)

    def expand_except_parameters_results(self):
        def process_item(item):
            if item.text(0) in ["⚙️ Parameters", "⚙️ Other Results"]:
                item.setExpanded(False)
            else:
                item.setExpanded(True)
                for i in range(item.childCount()):
                    process_item(item.child(i))

        root = self.tree.invisibleRootItem()
        for i in range(root.childCount()):
            process_item(root.child(i))

    def _setup_data_item(self, item: QTreeWidgetItem, data_obj: Data):
        """配置 Data 类型的行显示"""
        item.setText(0, f"📦 {data_obj.name}")
        item.setText(1, f"原始 ({data_obj.format_import})")
        item.setText(2, self._shape_to_str(data_obj.datashape)+'\n'+self._format_array_size(data_obj.data_origin))
        item.setText(3, f"{data_obj.datamin:.2f} ~ {data_obj.datamax:.2f}")
        # 将时间戳格式化
        time_str = time.strftime('%y/%m/%d %H:%M:%S', time.localtime(data_obj.timestamp))
        item.setText(4, time_str)

        # 检查是否线性数据并添加按钮
        self._check_and_add_button(item, data_obj.data_origin, data_obj.name, data_obj)

    def _setup_processed_item(self, item: QTreeWidgetItem, proc_obj: ProcessedData):
        """配置 ProcessedData 类型的行显示"""
        item.setText(0, f"🔎 {re.sub(r'[^@]+@', '...@', proc_obj.name)}") # 类似输出: ...@...@r_stft
        item.setText(1, f"🏷️ {proc_obj.type_processed}")
        if proc_obj.data_processed is not None:
            item.setText(2, self._shape_to_str(proc_obj.datashape)+'\n'+self._format_array_size(proc_obj.data_processed))
            item.setText(3, f"{proc_obj.datamin:.2f} ~ {proc_obj.datamax:.2f}")
        else:
            item.setText(2, "None")
        time_str = time.strftime('%y/%m/%d %H:%M:%S', time.localtime(proc_obj.timestamp))
        item.setText(4, time_str)

        # 检查是否线性数据并添加按钮
        if proc_obj.data_processed is not None:
            self._check_and_add_button(item, proc_obj.data_processed, proc_obj.name, proc_obj)

    def _fill_dict_items(self, parent_item: QTreeWidgetItem, data_dict: dict, original_obj):
        """递归填充字典数据"""
        for k, v in data_dict.items():
            child = QTreeWidgetItem(parent_item)
            child.setText(0, str(k))

            # 如果值是 numpy 数组，显示其摘要
            if isinstance(v, np.ndarray):
                child.setText(1, "ndarray")
                child.setText(2, self._shape_to_str(v.shape))
                child.setText(3, f'{v.min():.2f} ~ {v.max():.2f}')
                child.setText(4, "Array Data")
                # 如果是一维数组，也允许导出
                self._check_and_add_button(child, v, str(k), original_obj, False)
            elif isinstance(v, dict):
                child.setText(1, "dict")
                self._fill_dict_items(child, v, original_obj)  # 递归
            elif isinstance(v, list):
                child.setText(1, "list")
                child.setText(2, self._shape_to_str(len(v)))
                try:
                    child.setText(3, f'{min(v):.2f} ~ {max(v):.2f}')
                except:
                    child.setText(3, f'{v[0]} ~ {v[-1]}')
                child.setText(4, "List Data")
            elif isinstance(v, float):
                child.setText(1, "float")
                child.setText(4,f'{v:.4f}')
            else:
                child.setText(1, type(v).__name__)
                child.setText(4, str(v))

    def _check_and_add_button(self, item: QTreeWidgetItem, data_array: np.ndarray, name: str, original_obj, is_father = True):
        """
        判断数据是否为线性（1D），如果是，在最后一列添加按钮
        判断是否是图像，如果是，加导出成像按钮
        """
        if not isinstance(data_array, np.ndarray):
            return

        is_linear = False
        is_image = False
        # 判断逻辑：一维数组，或者二维数组中有一维是1 (例如 (1000, 1))
        if data_array.ndim == 1:
            is_linear = True
        elif data_array.ndim == 2:
            if data_array.shape[0] == 1 or data_array.shape[1] == 1:
                is_linear = True
            else:
                is_image = True
        elif data_array.ndim == 3:
            is_image = True

        if is_linear:
            linear_layout = QHBoxLayout()
            new_name = QLineEdit()
            new_name.setPlaceholderText("请为数据重命名（默认为原名，建议改名）")
            new_name.setToolTip("重命名数据仅在可视化窗口内应用")
            linear_btn = QPushButton("     数据可视化(线性)    ")
            linear_layout.addWidget(new_name)
            linear_layout.addWidget(linear_btn)
            linear_widget = QWidget()
            linear_widget.setLayout(linear_layout)
            # 使用 lambda 捕获数据
            # 注意：lambda 中的变量绑定问题，需要默认参数
            linear_btn.clicked.connect(lambda _, d=data_array, o=original_obj: self.emit_plot_signal(d, new_name.text() or name, o))
            linear_btn.setStyleSheet("padding: 0px;")

            # 因为 QTreeWidget 是 ItemView，需要用 setItemWidget 将 Widget 放入单元格
            self.tree.setItemWidget(item, 5, linear_widget)

        elif is_image and not is_father: # 不要父节点那些，只要参数字典里面的
            image_layout = QHBoxLayout()
            image_btn = QPushButton("     数据可视化(图像)    ")
            image_layout.addWidget(image_btn)
            image_btn.setToolTip("点击会创建新画布呈现选中的数据，同时新增ProcessedData")
            image_widget = QWidget()
            image_widget.setLayout(image_layout)
            image_btn.clicked.connect(lambda _, data = original_obj, key = name: self.emit_canvas_signal(data, key))
            image_btn.setStyleSheet("padding: 0px;")
            image_btn.setMinimumHeight(14)

            self.tree.setItemWidget(item, 5, image_widget)

    def emit_plot_signal(self, data, name, obj):
        """发射信号"""
        data = np.squeeze(data)
        logging.info(f"要呈现的数据: {name}, 大小: {data.shape}")
        self.sig_plot_request.emit(data, name, obj)

    def emit_canvas_signal(self, data, key):
        logging.info(f"要呈现的数据: {data.name}-{key}")
        self.sig_canvas_signal.emit(data, key)

    def event(self, e):
        """拦截 ContextHelp 事件（即点击了标题栏的 ? 号）"""
        if e.type() == QEvent.EnterWhatsThisMode:
            self.open_help_window()
            e.accept()
            return True

        return super().event(e)

    def open_help_window(self):
        """打开帮助界面"""
        QWhatsThis.leaveWhatsThisMode()
        self.help_window = CustomHelpDialog("数据查看功能使用说明",["data_view"],)
        self.help_window.show()
        self.help_window.raise_()
        self.help_window.activateWindow()

    @staticmethod
    def _format_array_size(array):
        """格式化numpy数组大小"""
        if array is None:
            return " 0 bytes"
        size_bytes = array.nbytes
        for unit in ['bytes', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f" {size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f" {size_bytes:.1f} TB"

    @staticmethod
    def _shape_to_str(shape):
        """将形状转换为 t×h×w 格式的字符串 """
        if hasattr(shape, 'shape'):
            # 如果传入的是numpy数组对象
            shape = shape.shape

        # 确保shape是可迭代的
        if not hasattr(shape, '__iter__'):
            shape = (shape,)

        # 将每个维度转换为字符串并用乘号连接
        return '×'.join(str(dim) for dim in shape)

# 心跳分析模式弹窗
class HeartBeatFrameSelectDialog(QDialog):
    def __init__(self, data, parent=None):
        super().__init__()
        self.setWindowTitle("心肌细胞分析设置")
        self.setGeometry(1000, 650, 300, 20)
        self.setModal(False)
        self.setWindowFlags(Qt.Dialog |
                            Qt.WindowCloseButtonHint |
                            Qt.WindowContextHelpButtonHint)
        self.help_window = None
        self.data = data
        self.parent = parent
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.mode_map = {0: 'video', 1: 'images', 2: 'both'}
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        step_layout = QHBoxLayout()
        step_layout.addWidget(QLabel("采样间距（不宜太短）："))
        self.step_input = QSpinBox()
        self.step_input.setRange(1,min(self.data.framesize))
        self.step_input.setValue(5)
        step_layout.addWidget(self.step_input)
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("处理模式:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["比较模式","连续分析"])
        self.mode_combo.currentIndexChanged.connect(self.mode_changed)
        mode_layout.addWidget(self.mode_combo)
        layout.addLayout(step_layout)
        layout.addLayout(mode_layout)
        self.base_label = QLabel("请输入基准帧:")
        layout.addWidget(self.base_label)
        self.base_frame_input = QSpinBox()
        self.base_frame_input.setRange(-1, self.data.timelength)
        self.base_frame_input.setValue(0)
        layout.addWidget(self.base_frame_input)
        self.compare_label = QLabel("请输入所有后续需要比较的帧")
        layout.addWidget(self.compare_label)
        self.motion_frame_input = QTextEdit()
        self.motion_frame_input.setPlaceholderText("输入帧位（图像第一帧是0）; 以逗号或分号分隔，范围用-; 间隔取值: 10-50-5 (10到50每5帧取1帧); 输入all选取全部帧")
        layout.addWidget(self.motion_frame_input)

        save_layout = QHBoxLayout()
        save_layout.addWidget(QLabel("快速保存："))
        self.save_check = QCheckBox()
        save_layout.addWidget(self.save_check)
        self.save_check.clicked.connect(self.save_changed)
        save_layout.addWidget(QLabel("保存格式："))
        self.save_mode = QComboBox()
        self.save_mode.addItems(['视频','图片','我都要！'])
        save_layout.addWidget(self.save_mode)
        self.save_mode.setEnabled(False)
        save_layout2 = QHBoxLayout()
        self.path_input = QLineEdit()
        self.path_input.setPlaceholderText("请选择或输入文件夹路径")
        self.browse_btn = QPushButton("浏览...")
        self.browse_btn.clicked.connect(self.browse_folder)
        save_layout2.addWidget(self.path_input,stretch=1)
        save_layout2.addWidget(self.browse_btn)
        self.path_input.setEnabled(False)
        self.browse_btn.setEnabled(False)
        layout.addLayout(save_layout)
        layout.addLayout(save_layout2)

        button_layout = QHBoxLayout()
        button_layout.addStretch()
        self.apply_btn = QPushButton("开始处理")
        self.apply_btn.clicked.connect(self.start_process)
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.apply_btn)
        button_layout.addWidget(self.cancel_btn)

        layout.addLayout(button_layout)

    def start_process(self):
        motion_frames = ToolBucket.parse_frame_input(self.motion_frame_input.toPlainText(),self.data.timelength)
        if motion_frames:
            if not self.parent.avi_thread.isRunning():
                self.parent.avi_thread.start()
            self.parent.heartbeat_signal.emit(self.data, self.step_input.value(),
                                              self.base_frame_input.value(),
                                              motion_frames, self.get_path(),
                                              self.mode_map[self.save_mode.currentIndex()])
            logging.info("完成帧数选择，开始心肌细胞运动分析")
            self.accept()

    def save_changed(self):
        """选择保存"""
        if self.save_check.isChecked():
            self.path_input.setEnabled(True)
            self.browse_btn.setEnabled(True)
            self.save_mode.setEnabled(True)
        else:
            self.path_input.setEnabled(False)
            self.browse_btn.setEnabled(False)
            self.save_mode.setEnabled(False)

    def mode_changed(self):
        """模式改变"""
        if self.mode_combo.currentIndex() == 0:
            self.base_frame_input.setEnabled(True)
            self.base_label.setText("请输入基准帧:")
            self.compare_label = QLabel("请输入所有后续需要比较的帧")
            self.base_frame_input.setValue(0)
        elif self.mode_combo.currentIndex() == 1:
            self.base_frame_input.setEnabled(False)
            self.base_label.setText("无需输入基准帧:")
            self.compare_label = QLabel("请输入所有参与处理的帧数")
            self.base_frame_input.setValue(-1)

    def browse_folder(self):
        """打开文件夹选择对话框"""
        folder = QFileDialog.getExistingDirectory(
            self,  # 父窗口
            "选择文件夹",  # 对话框标题
            "",  # 默认路径（空表示当前目录）
            QFileDialog.ShowDirsOnly  # 只显示文件夹
        )

        if folder:  # 如果用户选择了文件夹
            self.path_input.setText(folder)

    def get_path(self):
        """获取当前选择的路径"""
        if self.save_check.isChecked():
            if not self.path_input.text():
                QMessageBox.warning(self,"保存错误","请输入或选择要保存的文件夹")
                raise ValueError
            return self.path_input.text()
        else:
            return ""