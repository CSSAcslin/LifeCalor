import logging
from typing import LiteralString

import numpy as np
import matplotlib as plt
import matplotlib.font_manager as fm
import pandas as pd
from PyQt5.QtCore import pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QTabWidget, QApplication)
from DataManager import *


class ResultDisplayWidget(QTabWidget):
    """结果显示部件"""
    tab_type_changed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTabsClosable(True)
        self.tabCloseRequested.connect(self.close_tab)
        self.setMovable(True)
        self.currentChanged.connect(self._current_index)

        self.font_list = fm.findSystemFonts(fontpaths=r"C:\Windows\Fonts", fontext='ttf')
        self.chinese_fonts = [f for f in self.font_list if
                              any(c in f.lower() for c in ['simhei', 'simsun', 'microsoft yahei', 'fang'])]
        self.plot_settings = {
            'current_mode': 'heatmap',
            'line_style': '--',
            'line_width': 2,
            'marker_style': 's',
            'marker_size': 6,
            'color': '#1f77b4',
            'show_grid': False,
            'heatmap_cmap': 'jet',
            'contour_levels': 10,
            'set_axis': True,
            '_from_start_cal': False
        }
        self._init_counters()
        # 存储每个选项卡的数据
        self.tab_data = {}
        self.current_data = None
        self.current_dataframe = None

        # 设置Matplotlib默认字体
        if self.chinese_fonts:
            plt.rcParams['font.sans-serif'] = ['microsoft yahei']  # Windows常用
            # 或者使用其他字体如: 'Microsoft YaHei', 'SimSun', 'FangSong', 'KaiTi'
            plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    def _init_counters(self):
        """计数器初始化"""
        self.tab_counters = {
            'heatmap': 1,  # 热图
            'curve': 1,  # 寿命曲线
            'roi': 1,  # ROI曲线
            'diff': 1,  # 扩散系数
            'var': 1,  # 方差演化
            'quality': 1,  # 信号评估
            'series': 1,  # EM信号
            'scs': 1,  # 单通道信号
            'pre-scs': 1,  # 单通道信号不定阈值
            'hb': 1  # 心肌细胞跳动
        }

    def _current_index(self, index):
        """存储当前选中的选项卡索引"""
        self.current_index = index
        if index >= 0:
            tab = self.widget(index)
            tab_id = id(tab)

            if tab_id in self.tab_data:
                # # 更新当前原始数据（无用，待删）
                # self.current_data = self.tab_data.get(tab_id).get('raw_data')
                # 更新当前DataFrame
                self.current_dataframe = self.tab_data.get(tab_id).get('dataframe')

            # 发出标签页类型信号
            if tab_id in self.tab_data:
                tab_type = self.tab_data[tab_id]['type']
                self.tab_type_changed.emit(tab_type)
                return
            elif self.current_mode:
                self.tab_type_changed.emit(self.current_mode)
                return

    def store_tab_data(self, tab, tab_type, **kwargs):
        """存储选项卡数据"""
        self.tab_data[id(tab)] = {
            'type': tab_type,
            'raw_data': kwargs,
            'dataframe': self.current_dataframe.copy() if self.current_dataframe is not None else None
        }

    def close_tab(self, index):
        """关闭指定选项卡"""
        tab = self.widget(index)
        tab_id = id(tab)

        # 删除关联数据
        if tab_id in self.tab_data:
            del self.tab_data[tab_id]

        tab.deleteLater()
        self.removeTab(index)

        # 更新当前索引
        if self.count() > 0:
            self.current_index = self.currentIndex()

    def create_tab(self, tab_type, title_prefix, reuse_current=False):
        """创建新的结果选项卡"""
        if reuse_current and self.count() > 0:
            # 重用当前标签
            index = self.current_index
            title = self.tabText(index)

            # 获取现有部件
            tab = self.widget(index)
            canvas = tab.findChild(FigureCanvas)
            figure = canvas.figure
            figure.clear()

            # 返回现有资源
            return figure, canvas, index, title, tab
        else:
            # 创建新标签页
            # 标签名运算
            count = self.tab_counters[tab_type]
            self.tab_counters[tab_type] += 1
            title = f"{title_prefix}{count}"

            # 创建新的绘图画布
            figure = Figure()
            canvas = FigureCanvas(figure)

            # 创建新的标签页
            tab = QWidget()
            layout = QVBoxLayout(tab)
            layout.addWidget(canvas)

            # 添加到选项卡
            index = self.addTab(tab, title)
            self.setCurrentIndex(index)

            return figure, canvas, index, title, tab

    def update_plot_settings(self, new_settings, update=True):
        """更新绘图设置"""
        self.plot_settings.update(new_settings)
        if update:
            self.update_plot()

    def update_plot(self):
        """根据当前设置重新绘图"""
        if self.count() == 0:
            # 没有选项卡时就不需要更新
            return

        tab = self.widget(self.current_index)
        tab_id = id(tab)
        tab_info = self.tab_data.get(tab_id)

        if not tab_info:
            return

        tab_type = tab_info['type']
        raw_data = tab_info.get('raw_data', {})

        # 根据标签页类型调用对应的绘图方法
        if tab_type == 'heatmap':
            self.display_distribution_map(
                raw_data['data'],
                reuse_current=True
            )
        elif tab_type == 'curve':
            self.display_lifetime_curve(
                raw_data['data'],
                raw_data['time_unit'],
                reuse_current=True
            )
        elif tab_type == 'roi':
            self.display_roi_series(
                raw_data['positions'],
                raw_data['intensities'],
                raw_data.get('fig_title', ""),
                reuse_current=True
            )
        elif tab_type == 'diff':
            self.display_diffusion_coefficient(
                raw_data['data'],
                reuse_current=True
            )
        elif tab_type == 'var':
            self.plot_variance_evolution(
                reuse_current=True)

        elif tab_type == 'quality':
            self.display_quality(
                raw_data['data'],
                reuse_current=True
            )

        elif tab_type == 'series':
            self.plot_time_series(
                raw_data['time'],
                raw_data['series'],
                reuse_current=True)

        elif tab_type == 'scs' or tab_type == 'pre-scs':
            self.single_channel(
                raw_data['data'],
                thr_known=raw_data['thr_known'],
                reuse_current=True
            )

    def display_distribution_map(self, data, ax_title=None, reuse_current=False):
        """显示寿命热图"""
        self.current_mode = "heatmap"
        lifetime_map = data.data_processed
        figure, canvas, index, title, tab = self.create_tab(self.current_mode, '热', reuse_current)

        cmap = self.plot_settings['heatmap_cmap']
        levels = self.plot_settings['contour_levels']

        ax = figure.add_subplot(111)
        # 显示热图
        im = ax.imshow(lifetime_map, cmap=cmap)
        ax_title = '指数衰减寿命分布热图' if ax_title is None else ax_title
        figure.colorbar(im, ax=ax, label='lifetime')
        ax.set_title(ax_title)
        ax.axis('off')
        figure.tight_layout()
        canvas.draw()

        # 保存当前数据
        self.current_dataframe = pd.DataFrame(lifetime_map)
        self.store_tab_data(tab, self.current_mode, lifetime_map=lifetime_map)

    def display_lifetime_curve(self, data, time_unit="ps", reuse_current=False):
        """显示区域分析结果"""
        # 使用原来的结果显示区域
        self.current_mode = "curve"
        figure, canvas, index, title, tab = self.create_tab(self.current_mode, '寿', reuse_current)
        phy_signal = data.out_processed['phy_signal']
        lifetime = data.out_processed['lifetime']
        r_squared = data.out_processed['r_squared']
        fit_curve = data.out_processed['fit_curve']
        model_type = data.out_processed['model_type']
        time_point = data.time_point

        line_style = self.plot_settings['line_style']
        line_width = self.plot_settings['line_width']
        marker_style = self.plot_settings['marker_style']
        marker_size = self.plot_settings['marker_size']
        color = self.plot_settings['color']
        show_grid = self.plot_settings['show_grid']
        set_axis = self.plot_settings['set_axis']

        ax = figure.add_subplot(111)
        if set_axis:
            max_bound = np.max(phy_signal)
            min_bound = 0
            ax.set_ylim(min_bound, max_bound)

        # 绘制原始曲线
        # time_points = time_points - time_points[0]  # 从0开始
        ax.plot(time_point, phy_signal,
                markeredgecolor=color,  # 点边缘色
                markeredgewidth=line_width,
                label='原始数据',
                marker=marker_style,  # 方形点
                markersize=marker_size,  # 点大小
                linestyle='')

        if not self.plot_settings['_from_start_cal']:
            # 这是从最大值算的绘制拟合曲线
            max_idx = np.argmax(phy_signal)
            fit_time = time_point[max_idx:]
            ax.plot(fit_time, fit_curve, 'r', linestyle=line_style, label='拟合曲线')
            # 标记最大值
            ax.axvline(time_point[max_idx], color='g', linestyle=':', label='峰值位置')
        elif self.plot_settings['_from_start_cal'] and np.shape(time_point) == np.shape(fit_curve):
            # 这是从头算的拟合曲线绘制
            fit_time = time_point
            ax.plot(fit_time, fit_curve, 'r', linestyle=line_style, label='拟合曲线')

        # 标记r^2和τ
        if model_type == 'single':
            ax.text(0.05, 0.95, f' τ={lifetime:.2f}\n'
                    + r'$R^2$='
                    + f'{r_squared:.3f}',
                    transform=ax.transAxes,
                    ha='left', va='top',
                    bbox=dict(facecolor='white', alpha=0.8))
        elif model_type == 'double':
            ax.text(0.05, 0.95, f' τ1={lifetime[0]:.2f}\n'
                    + f' τ2={lifetime[1]:.2f}\n'
                    + r'$R^2$='
                    + f'{r_squared:.3f}',
                    transform=ax.transAxes,
                    ha='left', va='top',
                    bbox=dict(facecolor='white', alpha=0.8))

        ax.set_xlabel(f'时间/{time_unit}')
        ax.set_ylabel('信号强度')
        ax.set_title('寿命拟合曲线')
        ax.legend()
        ax.grid(show_grid)

        canvas.draw()
        self.current_dataframe = pd.DataFrame({
            'time': pd.Series(time_point),
            'signal': pd.Series(phy_signal),
            'fit_time': pd.Series(fit_time),
            'fit_curve': pd.Series(fit_curve)
        })
        self.store_tab_data(tab, self.current_mode, data=data, time_unit=time_unit)

    def display_roi_series(self, positions, intensities, fig_title="", reuse_current=False):
        """绘制向量ROI信号强度曲线"""
        self.current_mode = "roi"
        figure, canvas, index, title, tab = self.create_tab(self.current_mode, 'ROI', reuse_current)
        ax = figure.add_subplot(111)
        # line_style = self.plot_settings['line_style']
        # line_width = self.plot_settings['line_width']
        marker_style = self.plot_settings['marker_style']
        marker_size = self.plot_settings['marker_size']
        color = self.plot_settings['color']

        # 绘制曲线
        # ax.plot(positions, intensities,
        #         markeredgecolor=color,
        #         marker=marker_style,       # 方形点
        #         markersize=marker_size,    # 点大小
        #         linewidth=line_width,
        #         label='信号强度')
        ax.scatter(positions, intensities,
                   c=color,
                   marker=marker_style,  # 方形点
                   label='采样点')

        # 设置图表属性
        ax.set_title(fig_title)
        ax.set_xlabel("位置 (像素)")
        ax.set_ylabel("对比度")
        ax.legend()

        canvas.draw()

        self.current_dataframe = pd.DataFrame({
            'time': pd.Series(positions),
            'signal': pd.Series(intensities),
        })
        self.store_tab_data(tab, self.current_mode, fig_title=fig_title, positions=positions, intensities=intensities)

    def display_diffusion_coefficient(self, data, reuse_current=False):
        """绘制多帧信号及高斯拟合"""
        # if frame_data_dict is None:
        #     logging.warning('缺数据或数据有问题无法绘图')
        #     return
        self.current_mode = "diff"
        figure, canvas, index, title, tab = self.create_tab(self.current_mode, '扩', reuse_current)
        ax = figure.add_subplot(111)
        self.data = data
        self.dif_result = data.out_processed
        marker_style = self.plot_settings['marker_style']
        marker_size = self.plot_settings['marker_size']
        color = self.plot_settings['color']

        for i, data in enumerate(self.dif_result['signal']):
            positions = data[0]
            intensities = data[1]

            # 绘制原始数据
            ax.scatter(positions, intensities, s=1)

        for i, series in enumerate(self.dif_result['fitting']):
            positions = series[0]
            fitting_curve = series[1]
            ax.plot(positions, fitting_curve, '--',
                    label=f'{self.dif_result["time_series"][i]:.0f}ps')

        # 设置图表属性
        ax.set_title("多帧ROI信号强度及高斯拟合")
        ax.set_xlabel("位置 (μm)")
        ax.set_ylabel("对比度")
        ax.grid(True)
        ax.legend()
        canvas.draw()

        # 以下是整合数据
        try:
            layer1, layer2 = [], []
            times = self.dif_result['time_series']
            layer1.extend(['时间点：'])
            layer2.extend(['位置(μm)'])
            for i in range(times.shape[0]):
                # times0 = np.full(len(times),'时间点：')
                # times2 = np.full(len(times),'μs')
                layer1.extend([f'{times[i]:.2f}', 'μs'])
                layer2.extend(['原始数值', '拟合曲线'])
            max_len = max(data.shape[1] for data in self.dif_result['signal'])
            outcome = []
            position = np.pad(self.dif_result['signal'][0, 0], (0, max_len - len(self.dif_result['signal'][0, 0])),
                              mode='constant', constant_values=np.nan)
            outcome.extend([position])
            for i, data in enumerate(self.dif_result['signal']):
                signal = np.pad(data[1], (0, max_len - len(data[1])),
                                mode='constant', constant_values=np.nan)
                fitting = np.pad(self.dif_result['fitting'][i, 1], (0, max_len - len(self.dif_result['fitting'][i, 1])),
                                 mode='constant', constant_values=np.nan)
                outcome.extend([signal, fitting])
            columns = pd.MultiIndex.from_arrays([layer1, layer2])
            self.current_dataframe = pd.DataFrame(np.array(outcome).T, columns=columns)
            self.store_tab_data(tab, self.current_mode, data=data)
        except Exception as e:
            logging.error(f'数据打包出现问题：{e}')

    def plot_variance_evolution(self, reuse_current=False):
        """绘制方差随时间变化图并计算扩散系数"""
        if not hasattr(self, "dif_result"):
            logging.warning("请按照顺序点击按钮")
            return
        self.current_mode = "var"
        figure, canvas, index, title, tab = self.create_tab(self.current_mode, '方', reuse_current)

        ax = figure.add_subplot(111)
        show_grid = self.plot_settings['show_grid']

        times = self.dif_result["sigma"][0]
        variances = self.dif_result["sigma"][1]
        sigma_trim = self.dif_result["sigma"][:, self.dif_result["sigma"][1, :] != 0]
        # 绘制数据点
        ax.scatter(times, variances, c='r', s=50, label='方差数据')

        # 线性拟合
        slope, intercept = np.polyfit(times, variances, 1)
        fit_line = slope * times + intercept
        ax.plot(times, fit_line, 'b--',
                label=f'线性拟合 (D={slope / 2:.2e})')

        # 设置图表属性
        ax.set_title("高斯方差随时间演化")
        ax.set_xlabel("时间 (s)")
        ax.set_ylabel("方差 (μm²)")
        ax.grid(show_grid)
        ax.legend()
        canvas.draw()

        self.current_dataframe = pd.DataFrame({
            'time': self.dif_result['time_series'],
            'sigma': sigma_trim[1],
        })
        self.store_tab_data(tab, self.current_mode, data=self.data)

    def quality_avg(self, data, reuse_current=False):
        """绘制平均信号STFT结果（信号质量评估）"""
        # 提取目标频率附近的区域
        f = data.out_processed['frequencies']
        t = data.out_processed['time_series']
        coefficients = data.data_processed
        target_freq = data.out_processed['target_freq']
        freq_range = data.out_processed['scale_range']

        self.current_mode = "quality"
        figure, canvas, index, title, tab = self.create_tab(self.current_mode, 'EM')

        ax = figure.add_subplot(111)

        spec = ax.pcolormesh(t, f, 10 * np.log10(np.abs(coefficients) ** 2 + 1e-12),
                             shading='gouraud', cmap='viridis')
        ax.set_ylabel('频率 [Hz]')
        ax.set_xlabel('时间 [秒]')
        figure.colorbar(spec, label='功率 [dB/Hz]')
        # 标记目标频率
        ax.axhline(y=target_freq, color='r', linestyle='--', alpha=0.7)
        ymin = target_freq - freq_range / 2
        ymax = target_freq + freq_range / 2
        ax.axhspan(ymin, ymax, color='red', alpha=0.2)
        ax.set_title(f'信号质量评估 (目标频率: {target_freq} Hz)')

        canvas.draw()
        logging.info("质量评估绘制完成")

        # self.current_dataframe = pd.DataFrame({"Zxx":Zxx}) 目前有问题
        self.current_dataframe = pd.DataFrame(10 * np.log10(np.abs(coefficients) ** 2 + 1e-12))
        self.store_tab_data(tab, self.current_mode, data=data)

    def plot_time_series(self, time, series, reuse_current=False):
        """信号处理结果"""
        self.current_mode = "series"
        time = time[1:-1]
        series = series[1:-1]
        figure, canvas, index, title, tab = self.create_tab(self.current_mode, 'signal', reuse_current)
        show_grid = self.plot_settings['show_grid']
        line_style = self.plot_settings['line_style']
        line_width = self.plot_settings['line_width']

        ax = figure.add_subplot(111)
        ax.plot(time, series, 'b-', linewidth=line_width)
        ax.set_title("选区信号均值变化")
        ax.set_xlabel("time (s)")
        ax.set_ylabel(r"$\Delta$S")
        ax.grid(show_grid)
        canvas.draw()

        self.current_dataframe = pd.DataFrame({
            'time': time,
            'series': series,
        })
        self.store_tab_data(tab, self.current_mode, time=time, series=series)

    def single_channel(self, data, thr_known=True, reuse_current=False):
        """单通道信号"""
        if thr_known:
            self.current_mode = "scs"
        else:
            self.current_mode = "pre-scs"
        time_series = data.time_point
        signal = data.data_processed
        figure, canvas, index, title, tab = self.create_tab(self.current_mode, self.current_mode, reuse_current)
        show_grid = self.plot_settings['show_grid']
        line_style = self.plot_settings['line_style']
        line_width = self.plot_settings['line_width']

        ax = figure.add_subplot(111)
        ax.plot(time_series, signal, 'r-', linewidth=line_width)
        ax.set_title(f"单通道信号（thr = {data.out_processed['thr']})")
        ax.set_xlabel("time (s)")
        ax.set_ylabel(r"$\Delta$A")
        ax.grid(show_grid)
        canvas.draw()
        self.current_dataframe = pd.DataFrame({
            'time_series': time_series,
            'signal': signal,
        })
        self.store_tab_data(tab, self.current_mode, data=data, thr_known=thr_known)

    # def display_heartbeat(self,data,reuse_current=False):
    #     self.current_mode = "hb"
    #     step = data.out_processed['step']
    #     base_img = data.out_processed['base_data']
    #     h, w = base_img.shape
    #     y, x = np.mgrid[0:h:step, 0:w:step].reshape(2, -1).astype(int)
    #
    #     for i,n in enumerate(data.out_processed['after_series']):
    #         figure, canvas, index, title, tab = self.create_tab(self.current_mode, self.current_mode, reuse_current)
    #         figure.subplots_adjust(left=0.05, right=1, bottom=0, top=0.95)
    #         ax = figure.add_subplot(111)
    #         ax.imshow(base_img, cmap= 'gray', alpha = 0.5)
    #
    #         fx = data.data_processed[i, y, x, 0]  # U分量
    #         fy = data.data_processed[i, y, x, 1]
    #
    #         color_speed = data.out_processed['magnitude_list'][i, y, x]
    #
    #         Q = ax.quiver(x, y, fx, fy, color_speed,
    #                        cmap='jet',  # 颜色表，红色代表高速，蓝色代表低速
    #                        angles='xy', scale_units='xy', scale=1,  # scale=1 表示箭头长度直接对应像素位移
    #                        width=0.003, headwidth=3.5, headlength=4)
    #
    #         # 添加颜色条
    #         cbar = figure.colorbar(Q,ax=ax)
    #         cbar.set_label(f'Motion Magnitude ({data.out_processed['space_unit']}/s)', fontsize=10)
    #         cbar.ax.tick_params(labelsize=8)
    #         # 调整颜色条轴的位置
    #         cbar_ax = cbar.ax
    #
    #         # 获取图像轴的位置
    #         img_pos = ax.get_position()
    #
    #         # 设置颜色条轴的位置，使其与图像高度相同
    #         cbar_ax.set_position([img_pos.x1 + 0.02, img_pos.y0, 0.02, img_pos.height])
    #
    #         ax.set_title(f"Cardiomyocyte Motion Field\n(Step size: {step}px; No.{data.out_processed['base_num']} + {data.time_point[i]}帧)", fontsize=10)
    #         ax.axis('off')  # 关闭坐标轴
    #         # figure.tight_layout()
    #         canvas.draw()

    def display_heartbeat(self, data, reuse_current=False):
        """
        UI 显示入口函数
        """
        self.current_mode = "hb"

        # 警告：如果 after_series 很大，这里会崩溃。限制最大显示数量。
        MAX_TABS = 20
        if data.out_processed['process_name'] == '比较模式':
            series_to_show = data.out_processed['after_series']
        else:
            series_to_show = data.out_processed['after_series'][: -1]
        if len(series_to_show) > MAX_TABS:
            # 如果没有保存路径且帧数太多，只显示前几帧，避免卡死
            logging.warning(
                f"警告：帧数过多 ({len(series_to_show)})，仅显示前 {MAX_TABS} 帧。建议使用保存功能查看完整视频。")
            series_to_show = series_to_show[:MAX_TABS]

        step = data.out_processed['sampling_step']
        base_img = data.out_processed['base_data']  # 注意处理相邻帧模式下的 base_img 变化
        step_size = step * data.out_processed.get('space_step', 1)

        for i, n in enumerate(series_to_show):
            # 创建 UI Tab
            figure, canvas, index, title, tab = self.create_tab(self.current_mode, self.current_mode, reuse_current)
            figure.subplots_adjust(left=0.05, right=0.9, bottom=0.05, top=0.95)  # 调整布局给colorbar留位置
            ax = figure.add_subplot(111)

            # 兼容处理：如果是相邻帧模式，base_img 应该是一个列表或数组
            # 这里需要你确保 data.out_processed['base_data'] 的格式
            # 简单处理：如果是单张图就用单张，如果是列表就取第 i 张
            if isinstance(base_img, np.ndarray) and base_img.ndim == 3:  # (N, H, W)
                current_base = base_img[i]
                base_num = data.out_processed['base_num'][i]
                next_num = data.time_point[i] + data.out_processed['base_num'][0]
            else:
                current_base = base_img
                base_num = data.out_processed['base_num']
                next_num = data.out_processed['after_series'][i]

            # 调用核心绘图
            title_text = f"Motion: No.{base_num} ->{next_num},Step size:{step_size:.1f} {data.out_processed.get('space_unit', 'px')},"
            Q = HeartbeatDraw.plot_quiver_on_ax(
                ax, current_base,
                data.data_processed[i, ..., 0],
                data.data_processed[i, ..., 1],
                data.out_processed['magnitude_list'][i],
                step,
                title_text
            )

            # UI 独有的 Colorbar 布局调整 (这部分比较繁琐，保留你原来的或者简化)
            cbar = figure.colorbar(Q, ax=ax)
            cbar.set_label(f'Velocity ({data.out_processed['space_unit']}/s)', fontsize=10)
            cbar.ax.tick_params(labelsize=8)
            # 调整颜色条轴的位置
            cbar_ax = cbar.ax

            # 获取图像轴的位置
            img_pos = ax.get_position()

            # 设置颜色条轴的位置，使其与图像高度相同
            cbar_ax.set_position([img_pos.x1 + 0.02, img_pos.y0, 0.02, img_pos.height])

            canvas.draw()

            # 如果只显示一帧就够了，可以在这里 break
            # break


class HeartbeatDraw:

    @staticmethod
    def plot_quiver_on_ax(ax, base_img, flow_u, flow_v, speed_map, step, title_info, vmin=None, vmax=None):
        """
        纯粹的绘图逻辑，不依赖任何 PyQt 或 UI 组件
        """
        h, w = base_img.shape
        y, x = np.mgrid[0:h:step, 0:w:step].reshape(2, -1).astype(int)

        # 简单的边界保护
        y = np.clip(y, 0, h - 1)
        x = np.clip(x, 0, w - 1)

        ax.imshow(base_img, cmap='gray', alpha=0.5)

        Q = ax.quiver(x, y, flow_u[y, x], flow_v[y, x], speed_map[y, x],
                      cmap='jet', angles='xy', scale_units='xy', scale=1,
                      width=0.003, headwidth=3.5, headlength=4,
                      clim=(vmin, vmax))  # 设定统一的颜色范围

        ax.set_title(title_info, fontsize=10)
        ax.axis('off')
        return Q

    @classmethod
    def save_video_task(cls, data, save_path, step, export_mode='video', compact_layout=True):
        """
        心肌细胞数据便捷导出
        :param data: ProcessedData 对象
        :param save_path: 保存根目录
        :param step: 网格步长
        :param fps: 视频帧率 (默认 15)
        :param export_mode: 导出模式 - 'video' (仅视频), 'images' (仅图片序列), 'both' (都有)
        :param compact_layout: 是否裁剪边距 (True则尽量填满画面)
        :return: 啥也没有
        """
        # 导出目录
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        base_name = data.name.split('@')[0]
        video_out_path = None
        images_out_dir = None
        if data.out_processed['process_name'] == '比较模式':
            series = data.out_processed['after_series']
        else:
            series = data.out_processed['after_series'][: -1]
        fps = data.out_processed.get('fps', 30) // np.mean(np.diff(series)) if len(series) > 1 else 24

        # 视频路径处理
        if export_mode in ['video', 'both']:
            video_out_path = cls.get_unique_path(save_path, f"{base_name}_motion", is_folder=False, ext=".mp4")
            logging.info(f"准备导出视频: {video_out_path}, FPS={fps}")

        # 图片序列路径处理 (创建独立子文件夹)
        if export_mode in ['images', 'both']:
            images_out_dir = cls.get_unique_path(save_path, f"{base_name}_frames", is_folder=True)
            os.makedirs(images_out_dir)
            logging.info(f"准备导出图片序列至: {images_out_dir}")

        # 获取全局最大速度
        global_vmax = np.max(data.out_processed['magnitude_list'])

        # 初始化视频写入
        base_data_collection = data.out_processed['base_data']
        # 兼容处理：如果是列表取第一帧，如果是数组取shape
        # if isinstance(base_data_collection, list):
        #     h_img, w_img = base_data_collection[0].shape
        # elif base_data_collection.ndim == 3:
        #     h_img, w_img = base_data_collection[0].shape
        # else:
        #     h_img, w_img = base_data_collection.shape
        fig_w, fig_h = 10, 8
        dpi = 100

        # 初始化 VideoWriter
        video_writer = None
        if video_out_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # 视频分辨率必须是整数
            video_writer = cv2.VideoWriter(video_out_path, fourcc, fps, (fig_w * dpi, fig_h * dpi))

        # 使用 Agg 后端（无界面模式）
        fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
        canvas = FigureCanvasAgg(fig)

        for i, n in enumerate(series):
            fig.clf()  # 清理画布

            # 控制边距
            if compact_layout:
                # left/right/bottom/top 是归一化坐标 (0-1)
                # 留出 top=0.92 给标题，right=0.92 给 colorbar
                fig.subplots_adjust(left=0.02, right=0.92, bottom=0.02, top=0.92)
            ax = fig.add_subplot(111)

            # 处理基准图 (兼容单张或列表)
            if isinstance(base_data_collection, list) or (
                    isinstance(base_data_collection, np.ndarray) and base_data_collection.ndim == 3):
                current_base = base_data_collection[i] if i < len(base_data_collection) else base_data_collection[0]
                base_num = data.out_processed['base_num'][i]
                next_num = data.out_processed['after_series'][i + 1]
            else:
                current_base = base_data_collection
                base_num = data.out_processed['base_num']
                next_num = data.out_processed['after_series'][i]

            # 调用上面的纯绘图函数
            title = f"Frame: {n}"
            Q = cls.plot_quiver_on_ax(
                ax, current_base,
                data.data_processed[i, ..., 0],
                data.data_processed[i, ..., 1],
                data.out_processed['magnitude_list'][i],
                step, title, vmin=0, vmax=global_vmax
            )

            # 添加 Colorbar
            cbar = fig.colorbar(Q, ax=ax)
            # 添加颜色条
            cbar.set_label(f'Velocity ({data.out_processed['space_unit']}/s)', fontsize=10)
            cbar.ax.tick_params(labelsize=8)
            # 调整颜色条轴的位置
            cbar_ax = cbar.ax

            # 获取图像轴的位置
            img_pos = ax.get_position()

            # 设置颜色条轴的位置，使其与图像高度相同
            cbar_ax.set_position([img_pos.x1 + 0.02, img_pos.y0, 0.02, img_pos.height])

            ax.set_title(
                f"Cardiomyocyte Motion Field\n(Space step: {step}px; No.{base_num} ->{next_num}帧; fps: {data.out_processed.get('fps')})",
                fontsize=10)
            ax.axis('off')  # 关闭坐标轴

            # 转为图像并写入
            canvas.draw()
            image_rgba = np.asarray(canvas.buffer_rgba())
            image_bgr = cv2.cvtColor(image_rgba, cv2.COLOR_RGBA2BGR)

            # === 保存 ===
            # 1. 写入视频帧
            if video_writer:
                video_writer.write(image_bgr)

            # 2. 保存单帧图片
            if images_out_dir:
                # 文件名格式: frame_001.png
                img_name = os.path.join(images_out_dir, f"frame_{i:04d}.png")
                cv2.imwrite(img_name, image_bgr)

        if video_writer:
            video_writer.release()
        plt.close(fig)

        return video_out_path, images_out_dir

    @staticmethod
    def get_unique_path(directory, name, is_folder=False, ext=".mp4") -> str | LiteralString:
        """
        智能生成唯一路径。
        如果是文件：返回 /path/to/name_1.mp4
        如果是文件夹：返回 /path/to/name_1/
        """
        full_path = os.path.join(directory, name + (ext if not is_folder else ""))

        counter = 1
        while os.path.exists(full_path):
            new_name = f"{name}_{counter}"
            full_path = os.path.join(directory, new_name + (ext if not is_folder else ""))
            counter += 1

        return full_path