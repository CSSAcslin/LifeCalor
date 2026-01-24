import sys
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import (QTabWidget, QWidget, QVBoxLayout, QApplication,
                             QPushButton, QHBoxLayout, QCheckBox, QLabel, QAction)
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

        #十字光标相关
        self.v_line = pg.InfiniteLine(angle=90, movable=False, pen='g')  # 垂直线
        self.h_line = pg.InfiniteLine(angle=0, movable=False, pen='g')  # 水平线
        self.label = pg.TextItem(anchor=(0, 1), color='k')  # 坐标显示文本
        self._crosshair_enabled = False
        self.init_custom_context_menu()

    def toggle_crosshair(self, enable):
        """开启/关闭鼠标跟随的十字光标"""
        self._crosshair_enabled = enable
        if enable:
            self.plot_widget.addItem(self.v_line, ignoreBounds=True)
            self.plot_widget.addItem(self.h_line, ignoreBounds=True)
            self.plot_widget.addItem(self.label, ignoreBounds=True)
            # 代理信号：鼠标移动时触发
            self.plot_widget.scene().sigMouseMoved.connect(self._update_crosshair)
        else:
            self.plot_widget.removeItem(self.v_line)
            self.plot_widget.removeItem(self.h_line)
            self.plot_widget.removeItem(self.label)
            try:
                self.plot_widget.scene().sigMouseMoved.disconnect(self._update_crosshair)
            except:
                pass

    def _update_crosshair(self, pos):
        """内部方法：更新光标位置"""
        if self.plot_widget.sceneBoundingRect().contains(pos):
            mouse_point = self.plot_widget.plotItem.vb.mapSceneToView(pos)
            x, y = mouse_point.x(), mouse_point.y()

            self.v_line.setPos(x)
            self.h_line.setPos(y)
            self.label.setText(f"X={x:.2f}, Y={y:.2f}")
            self.label.setPos(x, y)

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
        # 自动添加图例（如果还没加）
        if self.plot_widget.plotItem.legend is None:
            self.plot_widget.addLegend()

        # 提取样式参数
        mode = kwargs.get('mode', 'line')
        if 'color' in kwargs:
            color = kwargs['color']
        else:
            # 取余数实现循环使用颜色
            color_index = len(self.data_items) % len(self.color_cycle)
            color = self.color_cycle[color_index]
        width = kwargs.get('width', 2)
        name = kwargs.get('name', 'data')
        symbol = kwargs.get('symbol', 'o')
        symbol_size = kwargs.get('symbol_size', 8)

        time_unit = kwargs.get('time_unit', None)
        x_data = array[:, 0].copy()  # 复制一份X轴数据，避免修改原数组
        y_data = array[:, 1]

        if time_unit and time_unit in self.unit_powers:
            if self.base_unit is None:
                # 情况A: 这是第一条数据，确立基准单位
                self.base_unit = time_unit
                # 可选: 更新X轴标签显示单位
                self.plot_widget.setLabel('bottom', text='Time', units=self.base_unit)
            else:
                # 情况B: 已有基准单位，需要进行换算
                if time_unit != self.base_unit:
                    # 计算幂次差。例如：基准 ms(-3), 当前 ns(-9)。 diff = -9 - (-3) = -6
                    # 比例因子 = 10 ^ (-6)。即 1ns = 1e-6 ms
                    power_diff = self.unit_powers[time_unit] - self.unit_powers[self.base_unit]
                    scale_factor = 10 ** power_diff
                    x_data = x_data * scale_factor
                    # 提示：你也可以在这里修改 name，比如 name += f" (scaled to {self.base_unit})"
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

        # 如果当前已经是 PSD 模式，新加的数据也要立即转换
        if self.psd_action.isChecked():
            self._apply_psd_to_item(item)

    def clear_all(self):
        """清空所有绘图内容"""
        self.plot_widget.clear()
        self.data_items.clear()

    def init_custom_context_menu(self):
        """向右键菜单添加自定义功能"""
        # 获取 ViewBox 的菜单
        self.vb = self.plot_widget.plotItem.vb

        # 创建一个动作
        self.psd_action = QAction("Show PSD (功率谱密度)", self.plot_widget)
        self.psd_action.setCheckable(True)  # 设为复选框模式
        self.psd_action.triggered.connect(self.toggle_psd_mode)

        # 将动作添加到菜单顶部 (或者你可以加到子菜单里)
        # 注意：pyqtgraph 的菜单构建是懒加载的，直接 addAction 可能需要在 menu 创建后
        # 这里使用 ViewBox 自带的扩展接口
        self.vb.menu.addAction(self.psd_action)

    def toggle_psd_mode(self):
        """切换 PSD 模式和普通模式"""
        is_psd = self.psd_action.isChecked()

        if is_psd:
            # === 切换到 PSD 模式 ===
            self.plot_widget.setLabel('bottom', "Frequency (Hz)")
            self.plot_widget.setLabel('left', "PSD (V²/Hz)")
            self.plot_widget.setLogMode(x=False, y=True)  # PSD 通常看对数坐标

            for item in self.data_items:
                self._apply_psd_to_item(item)
        else:
            # === 恢复到 时域 模式 ===
            label_unit = self.base_unit if self.base_unit else 's'
            self.plot_widget.setLabel('bottom', "Time", units=label_unit)
            self.plot_widget.setLabel('left', "Amplitude")
            self.plot_widget.setLogMode(x=False, y=False)

            for item in self.data_items:
                self._restore_item_data(item)

        # 重新适应坐标范围
        self.plot_widget.autoRange()

    def _apply_psd_to_item(self, item):
        """计算并应用 PSD 到单个 Item"""
        if item not in self.data_cache:
            return

        x_time, y_amp = self.data_cache[item]

        # === PSD 计算逻辑 (自定义) ===
        # 这里演示一个基于 numpy 的简单周期图法
        # 实际建议使用: f, Pxx = scipy.signal.welch(y_amp, fs=fs)

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




    def _restore_item_data(self, item):
        """从缓存恢复原始数据"""
        if item in self.data_cache:
            x, y = self.data_cache[item]
            item.setData(x, y)