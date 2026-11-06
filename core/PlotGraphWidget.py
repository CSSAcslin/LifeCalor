import sys
import logging
import numpy as np
import pyqtgraph as pg
from PyQt5.QtGui import QColor, QCursor
from PyQt5.QtWidgets import (QTabWidget, QWidget, QVBoxLayout, QApplication,
                             QPushButton, QHBoxLayout, QCheckBox, QLabel, QAction, QMenu, QActionGroup, QInputDialog)
from PyQt5.QtCore import Qt, pyqtSignal
from scipy import signal

from appearance import get_theme_manager, theme_tokens


class PlotGraphWidget(QWidget):
    '''基于PyQtGraph实现的强大数据显示及分析控件'''

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.plot_widget = pg.PlotWidget()
        layout.addWidget(self.plot_widget)
        self.legend = self.plot_widget.addLegend()
        self.legend.anchor((1, 0), (1, 0), offset=(-10, 10))
        self.color_cycle = [
            '#E59473', '#A18EE0', '#3BEAE0', '#D5F088',
            '#D67D7C', '#89B5FF', '#51EFAA', '#EBD881',
            '#C88ACD', '#63D2FF', '#7EF986', '#F4C476'


        ]
        # 记录单位对应的 10的幂次 (相对于秒)
        self.unit_powers = {
            's': 0,
            'ms': -3,
            'us': -6, 'μs': -6,  # 支持微秒的两种写法
            'ns': -9,
            'ps': -12,
            'fs': -15,
        }
        self.base_unit = None  # 用于存储第一个添加的数据的单位
        self.data_items = []
        self.data_cache = {}
        self.curve_time_units = {}
        self._analysis_validation_cache = {}
        self.precomputed_histogram_items = set()

        # 悬浮取值标签 (替代原来的十字光标)
        self.hover_label = QLabel(self)
        self.hover_label.hide()

        self.hover_point = pg.ScatterPlotItem(size=12, pen=pg.mkPen('w', width=2), brush=pg.mkBrush('r'))
        self.hover_point.setZValue(20)  # 置于最顶层
        self.hover_point.hide()
        self.plot_widget.addItem(self.hover_point)

        self.plot_widget.scene().sigMouseMoved.connect(self._on_mouse_moved)

        # 寻峰工具相关变量
        self.peak_region = pg.LinearRegionItem(
            brush=pg.mkBrush(0, 0, 255, 30),
            pen=pg.mkPen('b', width=2, style=Qt.DashLine),
            hoverPen=pg.mkPen('r', width=4)
                                               )
        self.peak_region.setZValue(10)
        self.peak_region.hide()
        self.plot_widget.addItem(self.peak_region)

        self.peak_marker = pg.ScatterPlotItem(size=18, pen=pg.mkPen('r', width=2), brush=pg.mkBrush('y'), symbol='star')
        self.peak_marker.setZValue(20)
        self.peak_marker.hide()
        self.plot_widget.addItem(self.peak_marker)

        self.peak_info_label = QLabel(self)
        self.peak_info_label.hide()

        self.peak_region.sigRegionChanged.connect(self._update_peak_analysis)
        self.init_custom_context_menu()
        self._theme_manager = get_theme_manager()
        if self._theme_manager is not None:
            self._theme_manager.register_adapter(self)
        else:
            app = QApplication.instance()
            theme_id = app.property("appearanceTheme") if app is not None else "light"
            self.apply_theme(theme_tokens(theme_id))

    def apply_theme(self, tokens):
        """Update plot chrome without recalculating or replacing plotted data."""
        colors = tokens.colors
        self.plot_widget.setBackground(colors["panel_bg"])
        for axis_name in ("left", "bottom", "right", "top"):
            axis = self.plot_widget.getAxis(axis_name)
            axis.setPen(pg.mkPen(colors["border"]))
            axis.setTextPen(pg.mkPen(colors["secondary"]))

        self.legend.setBrush(pg.mkBrush(QColor(colors["raised_bg"])))
        self.legend.setPen(pg.mkPen(colors["border"]))
        self._apply_legend_theme(colors["text"])

        self.hover_label.setStyleSheet(self._floating_label_style(tokens, "accent"))
        self.peak_info_label.setStyleSheet(self._floating_label_style(tokens, "info"))

    def _apply_legend_theme(self, text_color):
        for _sample, label in self.legend.items:
            text_item = getattr(label, "item", None)
            if text_item is not None:
                text_item.setDefaultTextColor(QColor(text_color))

    @staticmethod
    def _floating_label_style(tokens, accent_key):
        colors = tokens.colors
        background = QColor(colors["raised_bg"])
        return (
            "QLabel {"
            f"background-color: rgba({background.red()}, {background.green()}, "
            f"{background.blue()}, 235);"
            f"border: 1px solid {colors[accent_key]};"
            "border-radius: 4px; padding: 4px;"
            f"color: {colors['text']};"
            "}"
        )

    def resizeEvent(self, event):
        """确保悬浮标签始终在右下角"""
        super().resizeEvent(event)
        self._update_labels_position()

    def _update_labels_position(self):
        """动态排列右下角的悬浮标签（自动堆叠，避免重叠）"""
        margin_right = 20
        current_bottom = self.height() - 20  # 从最底部开始

        # 1. 如果光标取值框可见，先放置它在最底部
        if self.hover_label.isVisible():
            self.hover_label.move(self.width() - self.hover_label.width() - margin_right,
                                  current_bottom - self.hover_label.height())
            # 更新当前可用底部高度（往上抬，预留10px间距）
            current_bottom -= (self.hover_label.height() + 10)

            # 2. 如果寻峰框可见，放置在剩下的底部位置（取值框的上方，或沉底）
        if self.peak_info_label.isVisible():
            self.peak_info_label.move(self.width() - self.peak_info_label.width() - margin_right,
                                      current_bottom - self.peak_info_label.height())

    """悬浮取值功能"""
    def _on_mouse_moved(self, pos):
        """鼠标移动事件：计算距离最近的曲线点"""
        if not self.plot_widget.sceneBoundingRect().contains(pos):
            self.hover_label.hide()
            return

        mouse_point = self.plot_widget.plotItem.vb.mapSceneToView(pos)
        mouse_x, mouse_y = mouse_point.x(), mouse_point.y()

        closest_item = None
        closest_dist = float('inf')
        closest_val = (0, 0)

        # 遍历所有可见的曲线
        for item in self.data_items:
            if not item.isVisible() or item not in self.data_cache:
                continue

            x_data, y_data = self.data_cache[item]
            if len(x_data) == 0: continue

            # 使用二分查找快速找到X轴最近的点 (假设X是递增的，这在时间序列中成立)
            idx = np.searchsorted(x_data, mouse_x)
            if idx == 0:
                idx = 0
            elif idx == len(x_data):
                idx = len(x_data) - 1
            else:
                # 比较左右两个点哪个更近
                if abs(mouse_x - x_data[idx - 1]) < abs(mouse_x - x_data[idx]):
                    idx = idx - 1

            data_x, data_y = x_data[idx], y_data[idx]

            # 简单的距离判断：如果X距离和Y距离在合理范围内 (相当于屏幕上的吸附)
            # 这里为了简单，我们计算归一化距离
            view_rect = self.plot_widget.viewRect()
            dx = abs(data_x - mouse_x) / view_rect.width()
            dy = abs(data_y - mouse_y) / view_rect.height()
            dist = dx ** 2 + dy ** 2

            if dist < closest_dist and dist < 0.05:  # 0.05 是吸附阈值
                closest_dist = dist
                closest_item = item
                closest_val = (data_x, data_y, idx)

        if closest_item:
            name = closest_item.opts['name']
            self.hover_label.setText(f"◆ 取点 {name}\n  X: {closest_val[0]:.4g}\n  Y: {closest_val[1]:.4g}\n  帧位：{closest_val[2]}")
            self.hover_label.adjustSize()
            self.hover_label.show()

            if not self.hover_label.isVisible():
                self.hover_label.show()

            self.hover_point.setData([closest_val[0]], [closest_val[1]]) # 高亮点
            self.hover_point.show()
            self._update_labels_position()  # 内容变化，重新排版
        else:
            if self.hover_label.isVisible():
                self.hover_label.hide()
                self._update_labels_position()  # 隐藏后重新排版
            self.hover_point.hide()

    """绘图功能"""
    def handle_plot_signal(self, data, dict):
        """很简单的一个解包函数"""
        self.plot_data(data,**dict)

    def handle_from_image(self,data, name):
        """从图像直接获取的数据"""
        self.plot_data(data, name = name)

    def plot_data(self, array:np.ndarray, **kwargs):
        """
        添加或更新数据
        :param data_id: 数据的唯一标识 (str)
        :param x, y: 数据数组
        :param kwargs: 绘图参数
               - mode: 'line' 或 'scatter'
               - color: 线条/点颜色
               - width: 线宽
               - symbol: 散点形状 ('o', 's', 't', 'd', '+')
               - name: 图例名称
        """

        # 提取样式参数
        mode = kwargs.get('mode', 'line')
        color = kwargs.get('color', self.color_cycle[len(self.data_items) % len(self.color_cycle)])
        width = kwargs.get('width', 2)
        name = kwargs.get('name', f'data_{len(self.data_items)}')
        symbol = kwargs.get('symbol', 'o')
        symbol_size = kwargs.get('symbol_size', 8)
        time_unit = kwargs.get('time_unit', None)
        analysis_mode = kwargs.get('analysis_mode', None)
        precomputed_hist = analysis_mode in ('hist_precomputed', 'histogram_distribution')

        if array.ndim != 2 or array.shape[1] < 2:
            raise ValueError(f"PlotGraph 需要二维两列数据，实际 shape={array.shape}")

        x_data = array[:, 0].copy()  # 复制一份X轴数据，避免修改原数组
        y_data = array[:, 1]

        # Cache a curve's unit after converting its X values to the plot's time basis.
        curve_unit = time_unit or self.base_unit or 's'
        if not precomputed_hist and self.base_unit is None:
            self.base_unit = curve_unit
        if not precomputed_hist and curve_unit in self.unit_powers:
            if self.base_unit in self.unit_powers:
                power_diff = self.unit_powers[curve_unit] - self.unit_powers[self.base_unit]
                x_data = x_data * (10 ** power_diff)
                curve_unit = self.base_unit

        # 构造 Pen
        pen = pg.mkPen(color=color, width=width)

        # 如果是散点模式，pen 设为 None (只画点)
        plot_pen = pen if mode == 'line' else None
        plot_symbol = symbol if mode == 'scatter' else None

        # === 创建新数据 ===
        item = self.plot_widget.plot(
            x_data, y_data,
            pen=plot_pen,
            symbol=plot_symbol,
            symbolBrush=color,
            symbolSize=symbol_size,
            name=name
        )
        self.data_items.append(item)
        # 加入缓存
        self.data_cache[item] = (x_data, y_data)
        self.curve_time_units[item] = curve_unit
        item.visibleChanged.connect(self._curve_state_changed)
        if precomputed_hist:
            self.precomputed_histogram_items.add(item)
        else:
            self.precomputed_histogram_items.discard(item)

        target_action = self.action_hist if precomputed_hist else self._analysis_action
        if not self.update_analysis_mode(target_action):
            self.update_analysis_mode(self.action_time)

        self._setup_legend_interaction(item)
        self._apply_legend_theme(self._current_theme_colors()["text"])
        self.refresh_analysis_actions()
        # 自适应显示
        if len(self.data_items) == 1:
            self.plot_widget.autoRange()

    """图例功能"""
    def _setup_legend_interaction(self, item):
        """图例交互：左键显示/隐藏，双击重命名，右键删除"""
        target_sample, target_label = None, None

        for sample, label in self.legend.items:
            if label.text == item.opts['name']:
                target_sample, target_label = sample, label
                break

        if not target_label: return
        target_label._plot_item = item

        def mouseClickEvent(ev):
            if ev.button() == Qt.LeftButton:
                # 单击左键：隐藏/显示曲线
                hideEvent()

            elif ev.button() == Qt.RightButton:
                # 单击右键：弹出删除菜单 (使用 QCursor.pos() 完美解决报错)
                menu = QMenu()
                delete_action = menu.addAction("🗑️ 删除该曲线")
                hide_action = menu.addAction("👁️ 隐藏/显示该曲线")
                rename_action = menu.addAction("🖊️ 重命名该曲线")
                action = menu.exec_(QCursor.pos())
                if action == delete_action:
                    self._remove_curve(item)
                elif action == hide_action:
                    hideEvent()
                elif action == rename_action:
                    renameEvent()
            ev.accept()

        def mouseDoubleClickEvent(ev):
            # 双击左键：重命名
            if ev.button() == Qt.LeftButton:
                renameEvent()
            ev.accept()

        def renameEvent():
            new_name, ok = QInputDialog.getText(self, "重命名", "输入新名称:", text=item.opts['name'])
            if ok and new_name:
                self._rename_curve(item, new_name)

        def hideEvent():
            is_visible = item.isVisible()
            item.setVisible(not is_visible)
            target_sample.setOpacity(0.5 if is_visible else 1.0)
            target_label.setOpacity(0.5 if is_visible else 1.0)

        # 绑定重写的方法
        target_label.mouseClickEvent = mouseClickEvent
        target_label.mouseDoubleClickEvent = mouseDoubleClickEvent

    def _current_theme_colors(self):
        manager = get_theme_manager()
        if manager is not None:
            return manager.tokens.colors
        app = QApplication.instance()
        return theme_tokens(
            app.property("appearanceTheme") if app is not None else "light"
        ).colors

    def _rename_curve(self, item, new_name):
        self.legend.removeItem(item)
        item.opts['name'] = new_name
        self.legend.addItem(item, new_name)
        self._setup_legend_interaction(item)  # 重新绑定事件
        self._apply_legend_theme(self._current_theme_colors()["text"])
        self._curve_state_changed()

    def _remove_curve(self, item):
        self.plot_widget.removeItem(item)
        self.legend.removeItem(item)
        self.data_items.remove(item)
        del self.data_cache[item]
        self.curve_time_units.pop(item, None)
        self._analysis_validation_cache.pop(item, None)
        self.precomputed_histogram_items.discard(item)
        self.hover_label.hide()
        self._curve_state_changed()

    def clear_all(self):
        """清空所有绘图内容"""
        self.plot_widget.clear()
        self.plot_widget.addItem(self.peak_region)  # 保留组件
        self.plot_widget.addItem(self.peak_marker)
        self.plot_widget.addItem(self.hover_point)
        self.legend.clear()
        self.data_items.clear()
        self.data_cache.clear()
        self.curve_time_units.clear()
        self._analysis_validation_cache.clear()
        self.precomputed_histogram_items.clear()
        self.base_unit = None  # 重置基准时间单位
        self.hover_label.hide()
        self.hover_point.hide()
        self._toggle_peak_analyzer(False)
        self._analysis_action = self.action_time
        self.action_time.setChecked(True)
        self.refresh_analysis_actions()
        self._update_labels_position()

    def init_custom_context_menu(self):
        """Create actions that can be shared by the plot and the main menu."""
        ctrl_menu = self.plot_widget.plotItem.ctrlMenu
        self.analysis_menu = QMenu("Analysis 信号分析", self.plot_widget)
        ctrl_menu.addMenu(self.analysis_menu)
        self.action_time = QAction("恢复原始曲线", self.plot_widget, checkable=True)
        self.action_psd = QAction("功率谱密度（Welch）", self.plot_widget, checkable=True)
        self.action_hist = QAction("曲线幅值频数分布", self.plot_widget, checkable=True)
        self.action_time.setChecked(True)
        self._analysis_action = self.action_time

        self.analysis_group = QActionGroup(self.plot_widget)
        for action in (self.action_time, self.action_psd, self.action_hist):
            self.analysis_group.addAction(action)
            self.analysis_menu.addAction(action)
        self.analysis_group.triggered.connect(self.update_analysis_mode)

        ctrl_menu.addSeparator()
        self.action_peak = QAction("区间主峰分析", self.plot_widget, checkable=True)
        self.action_peak.toggled.connect(self._toggle_peak_analyzer)
        ctrl_menu.addAction(self.action_peak)
        ctrl_menu.aboutToShow.connect(self.refresh_analysis_actions)
        self.analysis_menu.aboutToShow.connect(self.refresh_analysis_actions)
        self.refresh_analysis_actions()

    def _curve_arrays(self, item, minimum_samples=1):
        x_values, y_values = self.data_cache[item]
        if np.iscomplexobj(x_values) or np.iscomplexobj(y_values):
            raise ValueError("曲线分析需要实数坐标和幅值")
        x_values = np.asarray(x_values, dtype=np.float64)
        y_values = np.asarray(y_values, dtype=np.float64)
        if x_values.ndim != 1 or y_values.ndim != 1 or x_values.size != y_values.size:
            raise ValueError("曲线需要等长的一维坐标和幅值")
        if x_values.size < minimum_samples:
            raise ValueError(f"至少需要 {minimum_samples} 个采样点")
        if not np.all(np.isfinite(x_values)) or not np.all(np.isfinite(y_values)):
            raise ValueError("曲线包含非有限坐标或幅值")
        return x_values, y_values

    def _psd_inputs(self, item):
        if item in self.precomputed_histogram_items:
            raise ValueError("预计算频数分布不能作为时间序列计算 PSD")
        x_values, y_values = self._curve_arrays(item, minimum_samples=2)
        unit = self.curve_time_units.get(item, self.base_unit or 's')
        if unit not in self.unit_powers:
            raise ValueError(f"不支持的时间单位 {unit}")
        with np.errstate(over='ignore', invalid='ignore'):
            differences = np.diff(x_values)
        if not np.all(np.isfinite(differences)) or np.any(differences <= 0):
            raise ValueError("PSD 时间坐标必须严格递增")
        spacing = differences[0]
        if not np.allclose(differences, spacing, rtol=1e-5, atol=spacing * 1e-8):
            raise ValueError("PSD 需要均匀采样的时间序列")
        interval = spacing * 10.0 ** self.unit_powers[unit]
        if not np.isfinite(interval) or interval <= 0 or not np.isfinite(1.0 / interval):
            raise ValueError("采样间隔无法转换为有限采样率")
        return y_values, 1.0 / interval

    def _peak_curve_arrays(self, item):
        if item in self.precomputed_histogram_items:
            raise ValueError("预计算频数分布不能作为时序主峰分析对象")
        x_values, y_values = self._curve_arrays(item, minimum_samples=3)
        with np.errstate(over='ignore', invalid='ignore'):
            differences = np.diff(x_values)
        if not np.all(np.isfinite(differences)) or np.any(differences <= 0):
            raise ValueError("主峰分析的坐标必须严格递增")
        return x_values, y_values

    def _peak_inputs(self):
        item = next((curve for curve in self.data_items
                     if curve.isVisible() and curve in self.data_cache), None)
        if item is None:
            raise ValueError("请先显示一条可分析的曲线")
        x_values, y_values = self._peak_curve_arrays(item)
        return item, x_values, y_values

    def _cached_analysis_errors(self, item):
        inputs = self.data_cache[item]
        unit = self.curve_time_units.get(item, self.base_unit or 's')
        histogram = item in self.precomputed_histogram_items
        cached = self._analysis_validation_cache.get(item)
        if cached is not None:
            cached_inputs, cached_unit, cached_histogram, errors = cached
            if cached_inputs is inputs and cached_unit == unit and cached_histogram == histogram:
                return errors
        errors = {}
        for name, validator in (
            ('psd', self._psd_inputs), ('hist', self._curve_arrays),
            ('peak', self._peak_curve_arrays),
        ):
            try:
                validator(item)
                errors[name] = ''
            except (ValueError, TypeError, OverflowError) as exc:
                errors[name] = str(exc)
        self._analysis_validation_cache[item] = (inputs, unit, histogram, errors)
        return errors

    def _curve_state_changed(self):
        self.refresh_analysis_actions()
        if self.action_peak.isChecked():
            self._update_peak_analysis()

    def refresh_analysis_actions(self):
        """Refresh menus from cached validation without recalculating curves."""
        items = [item for item in self.data_items if item in self.data_cache]
        self.action_time.setEnabled(bool(items))
        self.action_time.setToolTip("恢复当前绘图区的原始数据" if items else "请先添加曲线")
        for action, name in ((self.action_psd, 'psd'), (self.action_hist, 'hist')):
            reason = "请先添加曲线" if not items else ""
            for item in items:
                error = self._cached_analysis_errors(item)[name]
                if error:
                    reason = f"{item.opts.get('name', '曲线')} — {error}"
                    break
            action.setEnabled(not reason)
            action.setToolTip(reason or f"计算当前绘图区全部 {len(items)} 条曲线，包含隐藏曲线")
        peak_item = next((item for item in items if item.isVisible()), None)
        peak_reason = (
            self._cached_analysis_errors(peak_item)['peak'] if peak_item is not None
            else "请先显示一条可分析的曲线"
        )
        peak_tip = peak_reason or f"分析第一条可见曲线 {peak_item.opts.get('name', '曲线')} 的选定区间"
        self.action_peak.setEnabled(not peak_reason)
        self.action_peak.setToolTip(peak_tip)
        if self.action_peak.isChecked() and peak_reason:
            self._toggle_peak_analyzer(False)

    def update_analysis_mode(self, action):
        """Prepare every curve before committing a change of plot-wide mode."""
        if not self.data_cache:
            self._analysis_action.setChecked(True)
            self.refresh_analysis_actions()
            return False
        prepared = []
        try:
            for item in self.data_items:
                if item not in self.data_cache:
                    continue
                if action is self.action_time:
                    x_values, y_values = self.data_cache[item]
                elif action is self.action_psd:
                    y_values, sampling_rate = self._psd_inputs(item)
                    with np.errstate(over='ignore', invalid='ignore'):
                        x_values, y_values = signal.welch(
                            y_values, sampling_rate, nperseg=min(4096, y_values.size)
                        )
                    if not np.all(np.isfinite(y_values)):
                        raise ValueError("PSD 结果包含非有限值")
                elif action is self.action_hist:
                    x_values, y_values = self._curve_arrays(item)
                    if item not in self.precomputed_histogram_items:
                        y_values, bins = np.histogram(y_values, bins='auto')
                        x_values = bins[:-1] + np.diff(bins) / 2
                else:
                    raise ValueError("未知曲线分析模式")
                prepared.append((item, x_values, y_values))
        except (ValueError, TypeError, OverflowError, MemoryError) as exc:
            self._analysis_action.setChecked(True)
            self._analysis_validation_cache.clear()
            self.refresh_analysis_actions()
            action.setToolTip(str(exc))
            logging.warning("曲线分析未执行：%s", exc)
            return False

        if action is not self.action_time:
            self._toggle_peak_analyzer(False)
        self._analysis_action = action
        action.setChecked(True)
        if action is self.action_psd:
            self.plot_widget.setLabel('bottom', "Frequency", units='Hz')
            self.plot_widget.setLabel('left', "PSD (幅值²/Hz)")
        elif action is self.action_hist:
            self.plot_widget.setLabel('bottom', "Value (实际值)")
            self.plot_widget.setLabel('left', "Count (频数)")
        else:
            if all(item in self.precomputed_histogram_items for item, _, _ in prepared):
                self.plot_widget.setLabel('bottom', "Value (实际值)")
                self.plot_widget.setLabel('left', "Count (频数)")
            else:
                self.plot_widget.setLabel('bottom', "Time", units=self.base_unit or 's')
                self.plot_widget.setLabel('left', "Amplitude")
        self.plot_widget.setLogMode(x=False, y=action is self.action_psd)
        for item, x_values, y_values in prepared:
            item.setData(x_values, y_values)
            item.setFillLevel(0 if action is self.action_hist else None)
            brush = None
            if action is self.action_hist:
                pen = item.opts.get('pen')
                color = pen.color() if pen is not None else QColor('#4CAF50')
                color.setAlpha(128)
                brush = pg.mkBrush(color)
            item.setBrush(brush)
        self.plot_widget.autoRange()
        self._curve_state_changed()
        return True

    def _toggle_peak_analyzer(self, checked):
        if self.action_peak.isChecked() != bool(checked):
            self.action_peak.setChecked(bool(checked))
            return
        if not checked:
            self.peak_region.hide()
            self.peak_info_label.hide()
            self.peak_marker.hide()
            return
        try:
            self._peak_inputs()
        except (ValueError, TypeError, OverflowError) as exc:
            self._toggle_peak_analyzer(False)
            self.action_peak.setToolTip(str(exc))
            logging.warning("主峰分析未开启：%s", exc)
            return
        self.update_analysis_mode(self.action_time)
        view_range = self.plot_widget.viewRange()[0]
        midpoint = sum(view_range) / 2
        span = (view_range[1] - view_range[0]) / 6
        self.peak_region.setRegion([midpoint - span, midpoint + span])
        self.peak_region.show()
        self.peak_info_label.show()
        self._update_peak_analysis()

    def _show_peak_issue(self, message):
        self.peak_info_label.setText(message)
        self.peak_info_label.adjustSize()
        self.peak_marker.hide()
        self._update_labels_position()

    def _update_peak_analysis(self):
        if not self.action_peak.isChecked():
            return
        try:
            target_item, x_values, y_values = self._peak_inputs()
        except (ValueError, TypeError, OverflowError) as exc:
            self._show_peak_issue(str(exc))
            return
        minimum, maximum = self.peak_region.getRegion()
        if not np.isfinite(minimum) or not np.isfinite(maximum) or maximum <= minimum:
            self._show_peak_issue("请选择有限且非空的分析区间")
            return
        mask = (x_values >= minimum) & (x_values <= maximum)
        x_roi, y_roi = x_values[mask], y_values[mask]
        if x_roi.size < 3:
            self._show_peak_issue("区间数据过少，至少需要 3 个采样点")
            return
        with np.errstate(over='ignore', invalid='ignore'):
            baseline = np.interp(x_roi, [x_roi[0], x_roi[-1]], [y_roi[0], y_roi[-1]])
            y_corrected = y_roi - baseline
        if not np.all(np.isfinite(y_corrected)):
            self._show_peak_issue("基线扣除结果包含非有限值")
            return
        peak_index = int(np.argmax(y_corrected))
        peak_x, peak_y = x_roi[peak_index], y_roi[peak_index]
        peak_height = y_corrected[peak_index]
        threshold = np.finfo(float).eps * max(np.max(np.abs(y_roi)), np.finfo(float).tiny) * 16
        if peak_height <= threshold:
            self._show_peak_issue("区间内没有可量化的正向主峰")
            return
        with np.errstate(over='ignore', invalid='ignore'):
            area = np.trapezoid(y_corrected, x_roi)
        if not np.isfinite(area):
            self._show_peak_issue("峰面积无法表示为有限值")
            return
        half_height = peak_height / 2
        left_indices = np.flatnonzero(y_corrected[:peak_index] <= half_height)
        right_indices = np.flatnonzero(y_corrected[peak_index:] <= half_height)
        if not left_indices.size or not right_indices.size:
            self._show_peak_issue("分析区间未覆盖主峰两侧的半高边界")
            return
        left_x = x_roi[left_indices[-1]]
        right_x = x_roi[peak_index + right_indices[0]]
        fwhm = right_x - left_x
        if not np.isfinite(fwhm) or fwhm <= 0:
            self._show_peak_issue("半高宽无法表示为有限正数")
            return
        self.peak_marker.setData([peak_x], [peak_y])
        self.peak_marker.show()
        self.peak_info_label.setText(
            f"◆ 寻峰 {target_item.opts.get('name', '曲线')}\n"
            f"  Peak X: {peak_x:.4g}\n"
            f"  Peak Y: {peak_y:.4g} (净高 {peak_height:.4g})\n"
            f"  FWHM: {fwhm:.4g}\n"
            f"  Area: {area:.4g}"
        )
        self.peak_info_label.adjustSize()
        self._update_labels_position()
