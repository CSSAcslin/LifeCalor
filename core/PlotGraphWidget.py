import sys
import numpy as np
import pyqtgraph as pg
from PyQt5.QtGui import QCursor
from PyQt5.QtWidgets import (QTabWidget, QWidget, QVBoxLayout, QApplication,
                             QPushButton, QHBoxLayout, QCheckBox, QLabel, QAction, QMenu, QActionGroup, QInputDialog)
from PyQt5.QtCore import Qt, pyqtSignal
from scipy import signal

# 全局配置：白色背景，黑色前景色 (符合科研论文习惯)
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')
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
            'ns': -9
        }
        self.base_unit = None  # 用于存储第一个添加的数据的单位
        self.data_items = []
        self.data_cache = {}

        # 悬浮取值标签 (替代原来的十字光标)
        self.hover_label = QLabel(self)
        self.hover_label.setStyleSheet("""
                    QLabel {
                        background-color: rgba(255, 255, 255, 220);
                        border: 1px solid #C8E6C9;
                        border-radius: 4px;
                        padding: 4px;
                        color: #4CAF50;
                    }
                """)
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
        self.peak_info_label.setStyleSheet("""
                    QLabel {
                        background-color: rgba(240, 248, 255, 220); /* 淡蓝色背景区分取值框 */
                        border: 1px solid #4682B4; border-radius: 4px;
                        padding: 4px; color: #337AC6;
                    }
                """)
        self.peak_info_label.hide()

        self.peak_region.sigRegionChanged.connect(self._update_peak_analysis)
        self.init_custom_context_menu()

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
                closest_val = (data_x, data_y)

        if closest_item:
            name = closest_item.opts['name']
            self.hover_label.setText(f"◆ 取点 {name}\n  X: {closest_val[0]:.4g}\n  Y: {closest_val[1]:.4g}")
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

        x_data = array[:, 0].copy()  # 复制一份X轴数据，避免修改原数组
        y_data = array[:, 1]

        # 处理单位
        if time_unit and time_unit in self.unit_powers:
            if self.base_unit is None:
                self.base_unit = time_unit
                self.plot_widget.setLabel('bottom', text='Time', units=self.base_unit)
            elif time_unit != self.base_unit:
                power_diff = self.unit_powers[time_unit] - self.unit_powers[self.base_unit]
                x_data = x_data * (10 ** power_diff)

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

        # 根据当前选中的模式立即进行转换
        if self.action_psd.isChecked():
            self._apply_psd(item)
        elif self.action_hist.isChecked():
            self._apply_histogram(item)

        self._setup_legend_interaction(item)
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
            self._update_peak_analysis()

        # 绑定重写的方法
        target_label.mouseClickEvent = mouseClickEvent
        target_label.mouseDoubleClickEvent = mouseDoubleClickEvent

    def _rename_curve(self, item, new_name):
        self.legend.removeItem(item)
        item.opts['name'] = new_name
        self.legend.addItem(item, new_name)
        self._setup_legend_interaction(item)  # 重新绑定事件

    def _remove_curve(self, item):
        self.plot_widget.removeItem(item)
        self.legend.removeItem(item)
        self.data_items.remove(item)
        del self.data_cache[item]
        self.hover_label.hide()
        self._update_peak_analysis()

    def clear_all(self):
        """清空所有绘图内容"""
        self.plot_widget.clear()
        self.plot_widget.addItem(self.peak_region)  # 保留组件
        self.plot_widget.addItem(self.peak_marker)
        self.legend.clear()
        self.data_items.clear()
        self.data_cache.clear()
        self.base_unit = None  # 重置基准时间单位
        self.hover_label.hide()
        self.hover_point.hide()
        # self.peak_marker.hide()
        self.peak_info_label.hide()
        self._update_labels_position()

    def init_custom_context_menu(self):
        """向右键菜单添加自定义功能"""
        # 获取pyqtgraph 原生的 ViewBox 的菜单
        plot_item = self.plot_widget.plotItem
        ctrl_menu = plot_item.ctrlMenu

        # 创建一个 "Analysis" 子菜单
        self.analysis_menu = QMenu("Analysis 信号分析", self.plot_widget)
        ctrl_menu.addMenu(self.analysis_menu)

        # 创建三个视图动作
        self.action_time = QAction("Time Domain (恢复时域)", self.plot_widget, checkable=True)
        self.action_psd = QAction("PSD (功率谱密度)", self.plot_widget, checkable=True)
        self.action_hist = QAction("Histogram (频数分布)", self.plot_widget, checkable=True)

        # 默认选中时域
        self.action_time.setChecked(True)

        # 使用 QActionGroup 实现单选互斥
        self.analysis_group = QActionGroup(self.plot_widget)
        for act in (self.action_time, self.action_psd, self.action_hist):
            self.analysis_group.addAction(act)
            self.analysis_menu.addAction(act)
        self.analysis_group.triggered.connect(self.update_analysis_mode)

        # 寻峰工具开关
        ctrl_menu.addSeparator()
        self.action_peak = QAction("Peak Analyzer (寻峰工具)", self.plot_widget, checkable=True)
        self.action_peak.triggered.connect(self._toggle_peak_analyzer)
        ctrl_menu.addAction(self.action_peak)

    """分析模式"""
    def update_analysis_mode(self, action):
        """统一处理不同分析模式的切换"""
        if action == self.action_time:
            # === 恢复时域 ===
            label_unit = self.base_unit if self.base_unit else 's'
            self.plot_widget.setLabel('bottom', "Time", units=label_unit)
            self.plot_widget.setLabel('left', "Amplitude")
            self.plot_widget.setLogMode(x=False, y=False)
            for item in self.data_items:
                if item in self.data_cache:
                    x, y = self.data_cache[item]
                    item.setData(x, y)
                    item.setFillLevel(None)

        elif action == self.action_psd:
            # === 切换 PSD ===
            self._toggle_peak_analyzer(False)
            self.plot_widget.setLabel('bottom', "Frequency (Hz)")
            self.plot_widget.setLabel('left', "PSD (V²/Hz)")
            self.plot_widget.setLogMode(x=False, y=True)
            for item in self.data_items:
                self._apply_psd(item)

        elif action == self.action_hist:
            # === 切换频数分布 (Histogram) ===
            self._toggle_peak_analyzer(False)
            self.plot_widget.setLabel('bottom', "Amplitude Value (幅度)")
            self.plot_widget.setLabel('left', "Count (频数)")
            self.plot_widget.setLogMode(x=False, y=False)  # 直方图不看对数
            for item in self.data_items:
                self._apply_histogram(item)

        # 刷新坐标轴范围
        self.plot_widget.autoRange()

    def _apply_psd(self, item):
        """计算并应用 PSD 到单个 Item"""
        if item not in self.data_cache:
            return

        x_time, y_amp = self.data_cache[item]

        # 1. 计算采样率 (假设均匀采样)
        if len(x_time) > 1:
            dt = np.mean(np.diff(x_time))
            if self.base_unit in ['ms']:
                dt *= 1e-3
            elif self.base_unit in ['us', 'μs']:
                dt *= 1e-6
            elif self.base_unit in ['ns']:
                dt *= 1e-9
            fs = 1.0 / dt
        else:
            fs = 1.0

        # # 2. 计算 FFT
        # n = len(y_amp)
        # fft_res = np.fft.fft(y_amp)
        # fft_freq = np.fft.fftfreq(n, d=1 / fs)
        #
        # # 3. 计算功率谱密度 (Power Spectral Density)
        # # 只取正频率部分
        # pos_mask = fft_freq > 0
        # freqs = fft_freq[pos_mask]
        # # 简单的归一化: |FFT|^2 / (fs * N)
        # psd = (np.abs(fft_res[pos_mask]) ** 2) / (fs * n)
        # # 4. 更新绘图数据
        # item.setData(freqs, psd)
        seg_len = min(4096, len(y_amp))
        f, Pxx_den = signal.welch(y_amp, fs, nperseg=seg_len)  # Welch 方法更平滑
        item.setData(f, Pxx_den)
        item.setFillLevel(None)

    def _apply_histogram(self, item):
        """统计分布计算 (Numpy Histogram)"""
        if item not in self.data_cache: return
        x_time, y_data = self.data_cache[item]

        # 1. 使用 Numpy 计算直方图
        # bins='auto' 会自动使用最科学的算法(Freedman-Diaconis)决定分桶数量
        counts, bins = np.histogram(y_data, bins='auto')

        # 2. 计算每个 Bin 的中心点作为 X 轴坐标
        x_centers = (bins[:-1] + bins[1:]) / 2

        # 3. 更新图表
        # 因为你原本用的是折线图 PlotDataItem，直接传进去会画出“频率多边形(Frequency Polygon)”
        # 这种图线在数学意义上和柱状直方图是完全等效的，且性能更好。
        item.setData(x_centers, counts)

        # 可选视觉优化: 让线下面有阴影填充，看起来更像分布图
        item.setFillLevel(0)
        item.setBrush(pg.mkBrush(color=item.opts['pen'].color().name() + '80')) # 加点透明度


    def _restore_item_data(self, item):
        """从缓存恢复原始数据"""
        if item in self.data_cache:
            x, y = self.data_cache[item]
            item.setData(x, y)

    """寻峰功能"""
    def _toggle_peak_analyzer(self, checked):
        if checked:
            self.update_analysis_mode(self.action_time)
            # 初始化区间到当前视图的中间 1/3
            view_range = self.plot_widget.viewRange()[0]
            mid = sum(view_range) / 2
            span = (view_range[1] - view_range[0]) / 6
            self.peak_region.setRegion([mid - span, mid + span])

            self.peak_region.show()
            self.peak_info_label.show()
            self._update_peak_analysis()
        else:
            self.peak_region.hide()
            self.peak_info_label.hide()
            self.peak_marker.hide()

    def _update_peak_analysis(self):
        """核心寻峰算法计算"""
        if not self.action_peak.isChecked(): return
        min_x, max_x = self.peak_region.getRegion()

        # 寻找当前第一条可见的曲线作为分析对象
        target_item = None
        for item in self.data_items:
            if item.isVisible() and item in self.data_cache:
                target_item = item
                break

        if target_item is None:
            self.peak_info_label.setText("未找到可见曲线")
            self.peak_info_label.adjustSize()
            self._update_labels_position()
            self.peak_marker.hide()
            return

        x, y = self.data_cache[target_item]

        # 截取区间数据
        mask = (x >= min_x) & (x <= max_x)
        x_roi, y_roi = x[mask], y[mask]

        if len(x_roi) < 3:
            self.peak_info_label.setText("区间数据过少")
            self.peak_info_label.adjustSize()
            self._update_labels_position()
            self.peak_marker.hide()
            return

        # 1. 线性基线扣除 (基于区间两端点)
        baseline = np.interp(x_roi, [x_roi[0], x_roi[-1]], [y_roi[0], y_roi[-1]])
        y_corr = y_roi - baseline

        # 2. 寻找最高峰
        peak_idx = np.argmax(y_corr)
        peak_x = x_roi[peak_idx]
        peak_y_rel = y_corr[peak_idx]  # 相对基线高度
        peak_y_abs = y_roi[peak_idx]  # 绝对高度

        # 3. 计算峰面积 (梯形积分)
        area = np.trapz(y_corr, x_roi)

        # 4. 计算半高宽 FWHM
        half_max = peak_y_rel / 2

        # 寻找左半高点
        left_idx = np.where(y_corr[:peak_idx] <= half_max)[0]
        left_x = x_roi[left_idx[-1]] if len(left_idx) > 0 else x_roi[0]

        # 寻找右半高点
        right_idx = np.where(y_corr[peak_idx:] <= half_max)[0]
        right_x = x_roi[peak_idx + right_idx[0]] if len(right_idx) > 0 else x_roi[-1]

        fwhm = right_x - left_x

        # 五角星标峰
        self.peak_marker.setData([peak_x], [peak_y_abs])
        self.peak_marker.show()
        info = (f"◆ 寻峰 {target_item.opts['name']}\n"
                f"  Peak X: {peak_x:.4g}\n"
                f"  Peak Y: {peak_y_abs:.4g} (净高 {peak_y_rel:.4g})\n"
                f"  FWHM: {fwhm:.4g}\n"
                f"  Area: {area:.4g}")
        self.peak_info_label.setText(info)
        self.peak_info_label.adjustSize()
        self._update_labels_position()  # 重新计算位置，防止文字过长溢出