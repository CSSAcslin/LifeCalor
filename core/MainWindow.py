import logging
import multiprocessing
from unittest import case

import resources_rc
from datetime import datetime
from logging.handlers import RotatingFileHandler
import numpy as np
from PyQt5 import sip
from PyQt5.QtGui import QPixmap, QIcon, QFontDatabase
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QPushButton, QComboBox, QScrollArea,
                             QFileDialog, QSlider, QSpinBox, QDoubleSpinBox, QGroupBox,
                             QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QStackedWidget, QDockWidget,
                             QStatusBar, QScrollBar, QFrame
                             )
from PyQt5.QtCore import Qt, pyqtSignal, QThread, QMetaObject, QElapsedTimer, QSettings, QCoreApplication

from ImportManager import *
from DataProcessor import DataProcessor, MassDataProcessor
from ImageDisplayWindow import *
from LifetimeCalculator import LifetimeCalculator, CalculationThread
from ResultDisplayWidget import *
from ConsoleUtils import *
from ExtraDialog import *
from DataManager import *
from UpdateModule import *
from PlotGraphWidget import *
from widget import TriStateSwitch


class MainWindow(QMainWindow):
    """主窗口"""
    # import
    data_import_signal = pyqtSignal(str, str, dict)
    # process
    amend_data_signal = pyqtSignal(object)
    detect_bad_frames_auto_signal = pyqtSignal(np.ndarray, float)
    fix_bad_frames_signal = pyqtSignal(object, list, int)
    # cal
    start_reg_cal_signal = pyqtSignal(object, float, np.ndarray, str)
    start_dis_cal_signal = pyqtSignal(object, float, str, str, int, str, int, bool, int)
    start_heat_cal_signal = pyqtSignal(object, float, str, str, int, str, int, bool, int)
    start_dif_cal_signal = pyqtSignal(object, float, float, float, str)
    # mass
    pre_process_signal = pyqtSignal(object,int,bool)
    stft_quality_signal = pyqtSignal(object,float, int, int, int, int, int, str)
    stft_python_signal = pyqtSignal(object,object, int, int, int, int, int, str, bool, int, int)
    cwt_quality_signal = pyqtSignal(object,float, int, int, int, str)
    cwt_python_signal = pyqtSignal(object,float, int, int, str, float)
    mass_export_signal = pyqtSignal(np.ndarray, str, str, str, bool, dict)
    atam_signal = pyqtSignal(object)
    tDgf_signal = pyqtSignal(object,int,float,bool)
    sscs_signal = pyqtSignal(object, int, float, bool)
    tDFT_signal = pyqtSignal(object)
    heartbeat_signal = pyqtSignal(object, int, int ,list, str, str, float)
    easy_process = pyqtSignal(object, str, object)
    roi_processed_signal = pyqtSignal(object,np.ndarray,float,bool,bool,float)

    def __init__(self):
        super().__init__()
        # 基本信息初始化
        self.current_version = "0.12.3"  # 当前程序版本
        self.repo_owner = "CSSAcslin"  # 程序作者
        self.repo_name = "Carrier-Lifetime-Calculator"  # 程序仓库名
        self.PAT = "Bearer <your PAT>"

        # 参数初始化
        self.settings = QSettings()
        self.mode = 1
        self.data = None
        self.processed_data = None
        self.time_points = None
        self.time_step = 1.0
        self.space_step = 1.0
        self.idx = None
        self.vector_array = None
        self.focus_canvas = None
        self.init_params()

        # 界面加载
        self.init_ui()
        self.log_file = self.get_log_path()
        self.setup_menus()
        self.setup_logging()
        self.help_dialog = None

        # 进度条与计时器
        self.elapsed_timer = QElapsedTimer()
        self.last_time = 0 # 记录运算的时间
        self.last_progress = 0 # 记录进度
        self.last_percent = -1 # 记录百分比进度
        self.cached_remaining = "计算中..." # 记录剩余时长

        # 状态控制
        self._is_calculating = False
        # 信号连接
        self.signal_connect()
        # 更新检查
        self.auto_update_check()
        # 线程开启（默认不关闭的线程
        self.import_thread_open()
        self.import_thread.start()
        self.data_thread_open()
        self.data_thread.start()
        self.process_thread.start()
        self.cal_thread_open()
        self.EM_thread_open()
        self.log_startup_message()

    """参数配置相关功能"""
    def init_params(self):
        """初始化参数库"""
        # 基础参数
        self.basic_params = self._load_param_group('basic', {
            'time_step': 1.000,
            'space_step': 1.000,
            'time_unit': 'ps',
            'space_unit': 'μm',
            'region_size': 5,
            'bg_nums': 300,
        })

        # 绘图参数
        self.plot_params = self._load_param_group('plot', {
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
        })

        # 计算设置参数
        self.cal_set_params = self._load_param_group('cal_set', {
            'from_start_cal': False,
            'r_squared_min': 0.4,
            'peak_min': 0.0,
            'peak_max': 100.0,
            'tau_min': 1e-3,
            'tau_max': 1e3
        })

        # 电学测量参数
        self.EM_params = self._load_param_group('EM', {
            'EM_fps': 360,
            'target_freq': 30.0,
            'type': '',
            'stft_window_size': 128,
            'stft_noverlap': 120,
            'stft_window_type': 'hann',
            'stft_scale_range': 1,
            'custom_nfft': 360,
            'cwt_type': 'cmor3-3',
            'cwt_total_scales': 256,
            'cwt_scale_range': 10.0,
            'scs_thr': 2.5,
            'scs_zoom': 2,
            'thr_known': False,
        })

        # 工具参数
        self.tool_params = self._load_param_group('tool', {
            'pen_size': 2,
            'pen_color': '#008000',  # Qt.green
            'fill_color': '#006400',  # Qt.darkGreen
            'vector_color': '#FFFF00',  # Qt.yellow
            'anchor_select': False,
            'anchor_shape': 'square',
            'anchor_size' : 5,
            'anchor_method': 'mean',
            'angle_step': 0.7853981633974483,  # pi/4
            'auto_fill': False,
            'vector_width': 2,
            'colormap': 'Jet',
            'use_colormap': False,
            'auto_boundary_set': True,
            'min_value': '',
            'max_value': '',
        })

        self.save_params()
        self.save_timer = QTimer()
        self.save_timer.timeout.connect(self.save_params)
        self.save_timer.start(30000)  # 每10秒自动保存一次

    def _load_param_group(self, group_name, defaults):
        """加载参数组，如果没有则使用默认值"""
        params = {}
        self.settings.beginGroup(group_name)

        for key, default_value in defaults.items():
            # 尝试从QSettings读取
            saved_value = self.settings.value(key, default_value)

            # 处理类型转换
            if isinstance(default_value, bool):
                # 处理布尔值
                params[key] = self.settings.value(key, default_value, type=bool)
            elif isinstance(default_value, int):
                # 处理整数
                try:
                    params[key] = int(self.settings.value(key, default_value))
                except (ValueError, TypeError):
                    params[key] = default_value
            elif isinstance(default_value, float):
                # 处理浮点数
                try:
                    params[key] = float(self.settings.value(key, default_value))
                except (ValueError, TypeError):
                    params[key] = default_value
            elif isinstance(default_value, str) and not saved_value:
                # 处理空字符串
                params[key] = default_value
            else:
                # 其他情况（主要是字符串）
                params[key] = str(self.settings.value(key, default_value))

        self.settings.endGroup()
        return params

    def save_params(self):
        """保存所有参数到QSettings"""
        # 保存基本参数
        self._save_param_group('basic', self.basic_params)

        # 保存绘图参数
        self._save_param_group('plot', self.plot_params)

        # 保存计算设置参数
        self._save_param_group('cal_set', self.cal_set_params)

        # 保存电学测量参数
        self._save_param_group('EM', self.EM_params)

        # 保存工具参数
        self._save_param_group('tool', self.tool_params)

        # 同步到磁盘
        self.settings.sync()

    def _save_param_group(self, group_name, params):
        """保存参数组到QSettings"""
        self.settings.beginGroup(group_name)

        for key, value in params.items():
            self.settings.setValue(key, value)

        self.settings.endGroup()

    def update_param(self, group_name, key, value):
        """更新单个参数"""
        if group_name == 'basic':
            self.basic_params[key] = value
        elif group_name == 'plot':
            self.plot_params[key] = value
        elif group_name == 'cal_set':
            self.cal_set_params[key] = value
        elif group_name == 'EM':
            self.EM_params[key] = value
        elif group_name == 'tool':
            self.tool_params[key] = value
        else:
            raise ValueError(f"未知的参数组: {group_name}")

        # 立即保存到QSettings
        self.settings.beginGroup(group_name)
        self.settings.setValue(key, value)
        self.settings.endGroup()

    def get_param(self, group_name, key, default=None):
        """获取参数值"""
        param_groups = {
            'basic': self.basic_params,
            'plot': self.plot_params,
            'cal_set': self.cal_set_params,
            'EM': self.EM_params,
            'tool': self.tool_params
        }

        if group_name in param_groups and key in param_groups[group_name]:
            return param_groups[group_name][key]
        return default

    """GUI生成"""
    def init_ui(self):
        self.setWindowTitle(f"成像数据分析工具箱 v{self.current_version}")
        self.setGeometry(100, 50, 1700, 900)

        # 主部件和布局
        # main_widget = QWidget()
        # self.setCentralWidget(main_widget)

        # 左侧设置区域
        self.setup_left_panel()
        self.param_dock = QDockWidget("基础设置", self)
        self.param_dock.setWidget(self.left_panel)
        self.param_dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.param_dock.setMinimumSize(300, 700)
        self.param_dock.setMaximumWidth(350)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.param_dock) # 加到左侧

        # 右侧图像区域
        self.image_display = ImageDisplayWindow(self.tool_params,self)
        image_widget = QWidget()
        image_layout = QVBoxLayout(image_widget)


        image_layout.addWidget(self.image_display)

        # 时间滑块
        # self.time_slider = QSlider(Qt.Horizontal)
        # self.time_slider.setMinimum(0)
        # self.time_slider.setMaximum(0)
        # self.time_label = QLabel("时间点: 0/0")
        # slider_layout = QHBoxLayout()
        # slider_layout.addWidget(QLabel("时间序列:"))
        # slider_layout.addWidget(self.time_slider)
        # slider_layout.addWidget(self.time_label)
        # image_layout.addLayout(slider_layout)
        self.image_dock = QDockWidget("图像显示", self)
        self.image_dock.setWidget(image_widget)
        self.image_dock.setMinimumSize(700, 600)
        self.image_dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.addDockWidget(Qt.RightDockWidgetArea, self.image_dock)

        # 结果显示区域
        self.result_dock = QDockWidget("绘图结果", self)
        result_widget = QWidget()
        result_layout = QVBoxLayout(result_widget)
        # 垂直滑块添加
        right_layout_horizontal = QHBoxLayout()
        self.time_slider_vertical = QSlider(Qt.Vertical)
        self.time_slider_vertical.setRange(0, 0)
        self.time_slider_vertical.setVisible(False)
        self.result_display = ResultDisplayWidget()
        right_layout_horizontal.addWidget(self.time_slider_vertical)
        right_layout_horizontal.addWidget(self.result_display)
        result_layout.addLayout(right_layout_horizontal)
        self.result_dock.setWidget(result_widget)
        self.result_dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.result_dock.setMinimumSize(350, 350)
        self.addDockWidget(Qt.RightDockWidgetArea, self.result_dock)

        self.plot_dock = QDockWidget("数据结果", self)
        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)
        inner_layout = QHBoxLayout()
        self.add_data_btn = QPushButton("添加数据")
        self.reset_data_btn = QPushButton("清除数据")
        inner_layout.addWidget(self.add_data_btn)
        inner_layout.addWidget(self.reset_data_btn)
        inner_layout.addStretch()
        plot_layout.addLayout(inner_layout)
        self.graph_plot = PlotGraphWidget()
        plot_layout.addWidget(self.graph_plot)
        self.plot_dock.setWidget(plot_widget)
        self.plot_dock.setMinimumSize(350, 250)
        # self.plot_dock.setLayout(plot_layout)
        self.addDockWidget(Qt.RightDockWidgetArea, self.plot_dock)

        self.splitDockWidget(self.param_dock, self.image_dock, Qt.Horizontal)
        self.splitDockWidget(self.image_dock, self.result_dock, Qt.Horizontal)
        self.splitDockWidget(self.result_dock, self.plot_dock, Qt.Vertical)
        self.resizeDocks([self.image_dock, self.result_dock], [800, 600], Qt.Horizontal)
        self.resizeDocks([self.result_dock, self.plot_dock], [500, 400], Qt.Vertical)

        self.setup_status_bar()

        # 设置控制台
        self.setup_console()

    def setup_left_panel(self):
        """设置左侧面板"""
        button_style_sheet = """
        QPushButton {
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 6px 10px;
            font-weight: 500;
            min-width: 100px;
        }
        
        QPushButton:hover {
            background-color: #388E3C;
        }
        
        QPushButton:pressed {
            background-color: #2E7D32;
        }
        
        QPushButton:disabled {
            background-color: #A5D6A7;
            color: #E8F5E9;
        }
        
        QPushButton:focus {
            outline: 1px solid #81C784;
            border-radius: 4px;
            padding: 5px 10px;
            outline-offset: 1px;
        }"""
        self.left_panel = QWidget()
        self.left_panel_layout = QVBoxLayout()
        self.left_panel_layout.setContentsMargins(15,15,15,15)
        self.setup_data_panel()
        self.setup_parameter_panel()
        self.setup_modes_panel()
        self.left_panel_layout.addWidget(self.data_import)
        self.left_panel_layout.addWidget(self.parameter_panel)
        self.left_panel_layout.addWidget(self.modes_panel, stretch=1)
        self.left_panel_layout.addSpacing(15)
        # 添加分析按钮和导出按钮
        data_save_layout = QHBoxLayout()
        self.export_image_btn = QPushButton("导出结果为图片")
        self.export_data_btn = QPushButton("导出结果为数据")
        self.export_data_btn.setStyleSheet(button_style_sheet)
        self.export_image_btn.setStyleSheet(button_style_sheet)
        data_save_layout.addWidget(self.export_image_btn)
        data_save_layout.addWidget(self.export_data_btn)
        self.left_panel_layout.addLayout(data_save_layout)
        self.left_panel.setLayout(self.left_panel_layout)

    def setup_data_panel(self):
    # 数据导入面板
        self.data_import = self.QGroupBoxCreator('导入设置')
        left_layout0 = QVBoxLayout()
        left_layout0.setSpacing(2)

        # 基础参数设置
        param_panel = self.QGroupBoxCreator("","inner")
        param_layout = QVBoxLayout(param_panel)
        param_layout.setSpacing(2)
        # param_layout.setContentsMargins(1,7,1,7)
        time_step_layout = QHBoxLayout()
        time_step_layout.addWidget(QLabel("时间单位:"))
        self.time_step_input = QDoubleSpinBox()
        self.time_step_input.setMinimum(0.001)
        self.time_step_input.setMaximum(10000)
        self.time_step_input.setValue(self.basic_params['time_step'])
        self.time_step_input.setDecimals(3)
        self.time_step_input.valueChanged.connect(
            lambda: self.update_param('basic', 'time_step', self.time_step_input.value()))
        time_step_layout.addWidget(self.time_step_input)
        self.time_unit_combo = QComboBox()
        self.time_unit_combo.addItems(["ms", "μs", "ns", "ps", "fs"])
        self.time_unit_combo.setCurrentText(self.basic_params['time_unit'])
        self.time_unit_combo.currentTextChanged.connect(
            lambda: self.update_param('basic', 'time_unit', self.time_unit_combo.currentText()))
        time_step_layout.addWidget(self.time_unit_combo)
        time_step_layout.addWidget(QLabel("/帧"))
        param_layout.addLayout(time_step_layout)
        param_layout.addSpacing(5)
        space_step_layout = QHBoxLayout()
        space_step_layout.addWidget(QLabel("空间单位:"))
        self.space_step_input = QDoubleSpinBox()
        self.space_step_input.setMinimum(0.001)
        self.space_step_input.setDecimals(3)
        self.space_step_input.setValue(self.basic_params['space_step'])
        self.space_step_input.valueChanged.connect(
            lambda: self.update_param('basic', 'space_step', self.space_step_input.value()))
        space_step_layout.addWidget(self.space_step_input)
        self.space_unit_combo = QComboBox()
        self.space_unit_combo.addItems(["mm", "μm", "nm"])
        self.space_unit_combo.setCurrentText(self.basic_params['space_unit'])
        self.space_unit_combo.currentTextChanged.connect(
            lambda: self.update_param('basic', 'space_unit', self.space_unit_combo.currentText()))
        space_step_layout.addWidget(self.space_unit_combo)
        space_step_layout.addWidget(QLabel("/像素"))
        param_layout.addLayout(space_step_layout)

        param_layout.addSpacing(5)
        fps_layout = QHBoxLayout()
        fps_layout.addWidget(QLabel("视频帧率:"))
        self.fps_input = QSpinBox()
        self.fps_input.setRange(1, 100000)
        self.fps_input.setValue(self.EM_params['EM_fps'])
        self.fps_input.valueChanged.connect(lambda: self.update_param('EM', 'EM_fps', self.fps_input.value()))
        fps_layout.addWidget(self.fps_input)
        fps_layout.addWidget(QLabel(" Hz"))
        param_layout.addLayout(fps_layout)

        # 模式选择
        self.fuction_select = QComboBox()
        self.fuction_select.addItems(['请选择分析模式','超快成像动态分析','EM-iSCAT','其他方法'])
        left_layout0.addWidget(self.fuction_select)
        left_layout0.addWidget(param_panel)

        self.funtion_stack = QStackedWidget()
        nothing_group = self.QGroupBoxCreator(style="inner")
        nothing_layout = QVBoxLayout()
        nothing_layout.addWidget(QLabel("首先：请选择分析模式!"))
        nothing_group.setLayout(nothing_layout)
        self.funtion_stack.addWidget(nothing_group)

        # FS&PA模式下的文件夹选择
        fs_iSCAT_group = self.QGroupBoxCreator(style="inner")
        type_choose1 = QHBoxLayout()
        self.file_type_selector1 = QComboBox()
        self.file_type_selector1.addItems(['tif格式', 'sif格式'])
        self.file_type_stack1 = QStackedWidget()
        fs_group = self.QGroupBoxCreator(style="noborder") # tif
        tiff_layout = QVBoxLayout()
        self.group_selector = QComboBox()
        self.group_selector.addItems(['n', 'p', '不区分'])
        self.tiff_folder_btn = QPushButton("选择TIFF文件夹")
        tiff_layout.addWidget(self.group_selector)
        tiff_layout.addWidget(self.tiff_folder_btn)
        fs_group.setLayout(tiff_layout)
        self.file_type_stack1.addWidget(fs_group)
        # 光热信号处理模式下的文件夹选择
        PA_group = self.QGroupBoxCreator(style="noborder") # sif
        sif_layout = QVBoxLayout()
        sif_layout_inner = QHBoxLayout()
        method_label = QLabel("归一化方法:")         # 归一化方法选择
        self.method_combo = QComboBox()
        self.method_combo.addItems(["linear", "percentile", "sigmoid", "log", "clahe"])
        self.sif_folder_btn = QPushButton('选择SIF文件夹')
        sif_layout_inner.addWidget(method_label)
        sif_layout_inner.addWidget(self.method_combo)
        sif_layout.addLayout(sif_layout_inner)
        sif_layout.addWidget(self.sif_folder_btn)
        PA_group.setLayout(sif_layout)
        self.file_type_stack1.addWidget(PA_group)
        type_choose1.addWidget(self.file_type_selector1)
        type_choose1.addWidget(self.file_type_stack1)
        fs_iSCAT_group.setLayout(type_choose1)
        self.funtion_stack.addWidget(fs_iSCAT_group)

        # 文件类型为tiff
        EM_iSCAT_group = self.QGroupBoxCreator(style="inner")
        v_layout = QVBoxLayout()
        type_choose = QHBoxLayout()
        self.file_type_selector = QComboBox()
        self.file_type_selector.addItems(['avi格式', 'tiff格式'])
        type_choose.addWidget(self.file_type_selector)
        self.file_type_stack = QStackedWidget()
        avi_group = self.QGroupBoxCreator(style = "noborder") # avi 选择
        avi_layout = QVBoxLayout()
        self.avi_select_btn = QPushButton("选择avi文件")
        avi_layout.addWidget(self.avi_select_btn)
        avi_group.setLayout(avi_layout)
        self.file_type_stack.addWidget(avi_group)
        tiff_group = self.QGroupBoxCreator(style = "noborder") # tiff 选择
        tiff_layout = QVBoxLayout()
        self.EMtiff_folder_btn = QPushButton("选择TIFF文件夹")
        tiff_layout.addWidget(self.EMtiff_folder_btn)
        tiff_group.setLayout(tiff_layout)
        self.file_type_stack.addWidget(tiff_group)
        type_choose.addWidget(self.file_type_stack)

        v_layout.addLayout(type_choose)
        EM_iSCAT_group.setLayout(v_layout)
        self.funtion_stack.addWidget(EM_iSCAT_group)

        # 科学分析模块
        Sim_group = self.QGroupBoxCreator(style="inner")
        sim_layout = QVBoxLayout()
        # self.text_box = QTextEdit()
        # self.text_box.setPlaceholderText("输入Python代码或拖入.py文件")
        # self.text_box.setMaximumHeight(40)
        # self.text_box.setMinimumHeight(20)
        # sim_layout.addWidget(self.text_box)
        self.code_button = QPushButton('执行代码')
        sim_layout.addWidget(self.code_button)
        Sim_group.setLayout(sim_layout)
        self.funtion_stack.addWidget(Sim_group)

        # 总提示
        # self.folder_path_label = QLabel("未选择文件夹")
        # self.folder_path_label.setMaximumWidth(300)
        # self.folder_path_label.setWordWrap(True)
        # # self.folder_path_label.setStyleSheet("font-size: 14px;")  # 后续还要改

        left_layout0.addWidget(self.funtion_stack)
        # left_layout0.addSpacing(3)
        # left_layout0.addWidget(self.folder_path_label)
        self.data_import.setLayout(left_layout0)

    def setup_parameter_panel(self):
        """处理的模式设置"""
        self.parameter_panel = self.QGroupBoxCreator("处理设置")
        process_layout = QVBoxLayout()
        switch_layout = QHBoxLayout()
        self.tri_switch = TriStateSwitch.TriStateSwitch()
        self.tri_switch.setValue(1)
        self.mode_label = QLabel("默认模式")
        switch_layout.addWidget(QLabel('处理模式：'))
        self.tri_switch.setFixedWidth(100)
        switch_layout.addWidget(self.tri_switch)
        self.tri_switch.valueChanged.connect(self.mode_switch_change)
        self.mode_label.setStyleSheet(f"font-size: 18px; font-weight: bold; color: #999999")
        switch_layout.addWidget(self.mode_label)
        process_layout.addLayout(switch_layout)
        process_layout.addSpacing(5)
        self.mode_explain = QLabel("按标准处理流程执行的快速操作，\n数据源会自动选择")
        process_layout.addWidget(self.mode_explain)
        self.parameter_panel.setLayout(process_layout)

    def setup_modes_panel(self):
    # 分析总体设置
        self.modes_panel = self.QGroupBoxCreator("分析设置")
        left_layout1 = QVBoxLayout()
        left_layout1.setContentsMargins(1, 0, 1, 0)
        self.between_stack = QStackedWidget()
        # 默认显示
        nothing_GROUP = self.QGroupBoxCreator(style="noborder")
        nothing_layout1 = QVBoxLayout()
        nothing_layout1.addWidget(QLabel("首先：请选择分析模式!"))
        nothing_GROUP.setLayout(nothing_layout1)
        self.between_stack.addWidget(nothing_GROUP)
        self.setup_fs_GROUP()
        self.setup_EM_GROUP()
        self.setup_Other_GROUP()
        left_layout1.addWidget(self.between_stack)
        # left_layout1.addStretch(1)
        self.modes_panel.setLayout(left_layout1)

    def setup_fs_GROUP(self):
    # fs_iSCAT下的功能选择
        fs_iSCAT_GROUP = self.QGroupBoxCreator(style="noborder")
        operation_layout = QVBoxLayout()
        # 寿命模型选择
        lifetime_layout = QHBoxLayout()
        lifetime_layout.addWidget(QLabel("寿命模型:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["单指数衰减", "双指数-仅区域"])
        lifetime_layout.addWidget(self.model_combo)
        # 区域分析设置
        # operation_layout.addSpacing(10)
        operation_mode_layout = QHBoxLayout()
        operation_mode_layout.addWidget(QLabel("模式:"))
        self.FS_mode_combo = QComboBox()
        self.FS_mode_combo.addItems(["选区寿命热图", "指数衰减寿命曲线","载流子扩散系数计算"])
        operation_mode_layout.addWidget(self.FS_mode_combo)
        operation_layout.addLayout(operation_mode_layout)
        self.FS_mode_stack = QStackedWidget()
        # 载流子寿命分布图参数板
        heatmap_group = self.QGroupBoxCreator(style = "inner")
        heatmap_layout = QVBoxLayout()
        heatmap_layout.addStretch(1)

        multipro_layout = QHBoxLayout()
        self.multiprocess_check = QCheckBox()
        multipro_layout.addWidget(QLabel("启用加速："))
        multipro_layout.addWidget(self.multiprocess_check)
        heatmap_layout.addLayout(multipro_layout)
        heatmap_layout.addWidget(QLabel("建议关闭无用程序,启用后偶尔卡顿属正常现象"))
        cpunum_layout = QHBoxLayout()
        self.cpu_use_input = QSpinBox()
        self.cpu_use_input.setRange(0, 100)
        self.cpu_use_input.setValue(0)
        self.cpu_use_input.setSuffix(f"/{ToolBucket.available_cpu_count()[0]}")
        self.multiprocess_check.toggled.connect(lambda: self.cpu_use_input.setValue(ToolBucket.available_cpu_count()[1]))
        cpunum_layout.addWidget(QLabel("核数"))
        cpunum_layout.addWidget(self.cpu_use_input)
        heatmap_layout.addLayout(cpunum_layout)
        heatmap_layout.addLayout(lifetime_layout)

        cov_layout = QFormLayout()
        self.pre_cov_combo = QComboBox()
        self.pre_cov_combo.addItems(['不使用','smooth', 'gaussian', 'sharpen', 'edge', 'laplacian', 'average'])
        self.pre_cov_size = QSpinBox()
        self.pre_cov_size.setRange(2,1000)
        cov_layout.addRow(QLabel("预处理卷积选择：\n（计算前卷积）"),self.pre_cov_combo)
        cov_layout.addRow(QLabel("卷积核尺寸：\n（实际大小为尺寸*2-1）"),self.pre_cov_size)
        self.post_cov_combo = QComboBox()
        self.post_cov_combo.addItems(['不使用','smooth', 'gaussian', 'sharpen', 'edge', 'laplacian', 'average'])
        self.post_cov_size = QSpinBox()
        self.post_cov_size.setRange(2,1000)
        cov_layout.addRow(QLabel("后处理卷积选择：\n（结果卷积）"),self.post_cov_combo)
        cov_layout.addRow(QLabel("卷积核尺寸：\n（实际大小为尺寸*2-1）"),self.post_cov_size)
        heatmap_layout.addLayout(cov_layout)
        btn_layout = QHBoxLayout()
        self.analyze_btn = QPushButton("寿命热图")
        btn_layout.addWidget(self.analyze_btn)
        self.heat_transfer_btn = QPushButton("传热系数热图")
        btn_layout.addWidget(self.heat_transfer_btn)
        heatmap_layout.addLayout(btn_layout)
        heatmap_layout.addStretch(1)
        heatmap_group.setLayout(heatmap_layout)
        self.FS_mode_stack.addWidget(heatmap_group)
        # 特定区域寿命分析功能参数板
            # 区域分析参数
        self.region_shape_combo = QComboBox()
        self.region_shape_combo.addItems(["正方形", "圆形"])
        self.region_size_input = QSpinBox()
        self.region_size_input.setMinimum(1)
        self.region_size_input.setMaximum(50)
        self.region_size_input.setValue(self.basic_params['region_size'])
        self.region_size_input.valueChanged.connect(lambda: self.update_param('basic','region_size',self.region_size_input.value()))
        self.analyze_region_btn = QPushButton("分析选定区域")
            # 区域坐标输入
        self.region_x_input = QSpinBox()
        self.region_y_input = QSpinBox()
        self.region_x_input.setMaximum(131)
        self.region_y_input.setMaximum(131)
            # 区域分析面板生成
        region_group = self.QGroupBoxCreator(style = "inner")
        region_layout = QVBoxLayout()
        lifetime_layout = QHBoxLayout()
        lifetime_layout.addWidget(QLabel("寿命模型:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["单指数衰减", "双指数-仅区域"])
        lifetime_layout.addWidget(self.model_combo)
        coord_layout = QHBoxLayout()
        coord_layout.addWidget(QLabel("中心X:"))
        coord_layout.addWidget(self.region_x_input)
        coord_layout.addWidget(QLabel("中心Y:"))
        coord_layout.addWidget(self.region_y_input)
        shape_layout = QHBoxLayout()
        shape_layout.addWidget(QLabel("区域形状:"))
        shape_layout.addWidget(self.region_shape_combo)
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("区域大小:"))
        size_layout.addWidget(self.region_size_input)
        region_layout.addLayout(lifetime_layout)
        region_layout.addLayout(coord_layout)
        region_layout.addLayout(shape_layout)
        region_layout.addLayout(size_layout)
        region_layout.addWidget(self.analyze_region_btn)
        region_group.setLayout(region_layout)
        self.FS_mode_stack.addWidget(region_group)
        # 载流子扩散系数计算参数板
        diffusion_group = self.QGroupBoxCreator(style = "inner")
        diffusion_layout = QVBoxLayout()
        self.vector_signal_btn = QPushButton("1.计算ROI上全时信号强度")
        self.frame_input = QTextEdit()
        self.frame_input.setPlaceholderText("2.输入帧位（起始帧位为0），以逗号或分号分隔，范围用-\n输入all选取全部帧")
        self.frame_input.setFixedHeight(70)
        self.select_frames_btn = QPushButton("3.计算选定时刻信号强度")
        self.diffusion_coefficient_btn = QPushButton("4.展示方差演化图及扩散系数")
        diffusion_layout.addWidget(self.vector_signal_btn)
        diffusion_layout.addWidget(self.frame_input)
        diffusion_layout.addWidget(self.select_frames_btn)
        diffusion_layout.addWidget(self.diffusion_coefficient_btn)
        diffusion_group.setLayout(diffusion_layout)
        self.FS_mode_stack.addWidget(diffusion_group)
        operation_layout.addWidget(self.FS_mode_stack)
        fs_iSCAT_GROUP.setLayout(operation_layout)
        self.between_stack.addWidget(fs_iSCAT_GROUP)

    def setup_EM_GROUP(self):
    # EM_iSCAT下的功能选择
        EM_iSCAT_GROUP = self.QGroupBoxCreator(style="noborder")
        EM_iSCAT_layout = QVBoxLayout()
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)  # 关键设置
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        scroll_content = QWidget()
        scroll_content.setStyleSheet(""" QWidget {background-color: white; }""")
        EM_iSCAT_layout1 = QVBoxLayout(scroll_content)
        preprocess_set_layout = QHBoxLayout()
        preprocess_set_layout.addWidget(QLabel("背景帧数："))
        self.bg_nums_input = QSpinBox()
        self.bg_nums_input.setMinimum(1)
        self.bg_nums_input.setMaximum(9999)
        self.bg_nums_input.setValue(self.basic_params['bg_nums'])
        self.bg_nums_input.valueChanged.connect(lambda: self.update_param('basic','bg_nums',self.bg_nums_input.value()))
        preprocess_set_layout.addWidget(self.bg_nums_input)
        self.preprocess_data_btn = QPushButton("数据预处理")
        # preprocess_set_layout2 = QHBoxLayout()
        # preprocess_set_layout2.addWidget(QLabel("是否显示结果："))
        # self.show_stft_check = QCheckBox()
        # self.show_stft_check.setChecked(False)
        # preprocess_set_layout2.addWidget(self.show_stft_check,alignment=Qt.AlignRight)
        EM_iSCAT_layout1.addLayout(preprocess_set_layout)
        EM_iSCAT_layout1.addWidget(self.preprocess_data_btn)
        EM_iSCAT_layout1.addSpacing(4)
        # EM_iSCAT_layout1.addLayout(preprocess_set_layout2)
        # EM_iSCAT_layout1.addSpacing(4)
        process_set_layout = QHBoxLayout()
        self.EM_mode_combo = QComboBox()
        self.EM_mode_combo.addItems(["stft短时傅里叶","cwt连续小波变换"])
        process_set_layout.addWidget(QLabel("变换方法："))
        process_set_layout.addWidget(self.EM_mode_combo)
        EM_iSCAT_layout1.addLayout(process_set_layout)
        self.EM_mode_stack = QStackedWidget()
        # stft 短时傅里叶变换
        stft_GROUP = self.QGroupBoxCreator(style="inner")
        stft_layout = QVBoxLayout()
        stft_GROUP.setLayout(stft_layout)
        process_set_layout1 = QHBoxLayout()
        # process_set_layout1.addWidget(QLabel("处理方法："))
        # self.stft_program_select = QComboBox()
        # self.stft_program_select.addItems(["python", "julia（未实现）"])
        # process_set_layout1.addWidget(self.stft_program_select)
        self.stft_window_select = QComboBox()
        self.stft_window_select.addItems(["汉宁窗(hann)", "汉明窗(hanming)","gabor变换(gaussian)","矩形窗","blackman",'blackman-harris'])
        process_set_layout2 = QHBoxLayout()
        process_set_layout2.addWidget(QLabel("窗选择："))
        process_set_layout2.addWidget(self.stft_window_select)
        self.stft_quality_btn = QPushButton("stft质量评价（功率密度谱）")
        self.retransform_input = QTextEdit()
        self.retransform_input.setPlaceholderText("频率范围设定，若留空，则取质量评价中的设置值")
        self.retransform_input.setFixedHeight(30)
        self.stft_process_btn = QPushButton("执行短时傅里叶变换")
        stft_layout.addLayout(process_set_layout1)
        stft_layout.addLayout(process_set_layout2)
        stft_layout.addWidget(self.stft_quality_btn)
        stft_layout.addWidget(self.retransform_input)
        stft_layout.addWidget(self.stft_process_btn)
        self.EM_mode_stack.addWidget(stft_GROUP)
        # cwt 小波变换
        cwt_GROUP = self.QGroupBoxCreator(style='inner')
        cwt_layout = QVBoxLayout()
        cwt_GROUP.setLayout(cwt_layout)
        cwt_set_layout1 = QHBoxLayout()
        cwt_set_layout1.addWidget(QLabel("处理方法："))
        self.cwt_program_select = QComboBox()
        self.cwt_program_select.addItems(["python","julia"])
        cwt_set_layout1.addWidget(self.cwt_program_select)
        self.cwt_quality_btn = QPushButton("cwt质量检验（功率谱）")
        self.cwt_process_btn = QPushButton("执行连续小波变换")
        cwt_layout.addLayout(cwt_set_layout1)
        cwt_layout.addWidget(self.cwt_quality_btn)
        cwt_layout.addWidget(self.cwt_process_btn)
        self.EM_mode_stack.addWidget(cwt_GROUP)
        EM_iSCAT_layout1.addWidget(self.EM_mode_stack)
        self.EM_output_btn = QPushButton("时频变换结果快捷导出")
        EM_iSCAT_layout1.addWidget(self.EM_output_btn)

        EM_iSCAT_layout2 = QHBoxLayout()
        self.after_process_select = QComboBox()
        self.after_process_select.addItems(["全细胞电生理分析","单通道电生理分析"])
        EM_iSCAT_layout2.addWidget(QLabel("后处理方法："))
        EM_iSCAT_layout2.addWidget(self.after_process_select)
        self.after_process_stack = QStackedWidget()
        whole_cell_GROUP = self.QGroupBoxCreator(style='inner')
        whole_cell_layout = QVBoxLayout()
        self.tDFT_btn = QPushButton("二维傅里叶变换")
        # self.retransform_input = QTextEdit()
        # self.retransform_input.setPlaceholderText("频率范围设定")
        # self.retransform_input.setFixedHeight(30)
        # self.retransform_btn = QPushButton("重设频率范围的变换")
        self.roi_signal_btn = QPushButton("选区信号均值变化(快速选择ROI)")
        # whole_cell_layout.addWidget(self.retransform_input)
        # whole_cell_layout.addWidget(self.retransform_btn)
        whole_cell_layout.addWidget(self.roi_signal_btn)
        whole_cell_layout.addWidget(self.tDFT_btn)
        whole_cell_GROUP.setLayout(whole_cell_layout)
        self.after_process_stack.addWidget(whole_cell_GROUP)

        single_channel_GROUP = self.QGroupBoxCreator(style='inner')
        single_channel_layout = QVBoxLayout()
        self.atam_btn = QPushButton("累计时间振幅图")
        self.tDgf_btn = QPushButton("选区二维高斯拟合")
        self.sscs_btn = QPushButton("简单单通道提取")
        single_channel_layout.addWidget(self.atam_btn)
        single_channel_layout.addWidget(self.tDgf_btn)
        single_channel_layout.addWidget(self.sscs_btn)
        single_channel_GROUP.setLayout(single_channel_layout)
        self.after_process_stack.addWidget(single_channel_GROUP)

        EM_iSCAT_layout1.addLayout(EM_iSCAT_layout2)
        EM_iSCAT_layout1.addWidget(self.after_process_stack)

        EM_iSCAT_layout1.addStretch(1)
        scroll_area.setWidget(scroll_content)
        EM_iSCAT_layout.addWidget(scroll_area)
        EM_iSCAT_GROUP.setLayout(EM_iSCAT_layout)
        self.between_stack.addWidget(EM_iSCAT_GROUP)

    def setup_Other_GROUP(self):
        # 其他方法模块
        Other_GROUP = self.QGroupBoxCreator(style='noborder')
        Other_layout = QVBoxLayout()
        other_inner_layout1 = QHBoxLayout()
        self.roi_fast_btn = QPushButton("ROI快速选择")
        other_inner_layout1.addWidget(QLabel("画布选择："))
        self.roi_pick = QComboBox()
        other_inner_layout1.addWidget(self.roi_pick)
        self.roi_pick.addItem("无画布数据")
        Other_layout.addLayout(other_inner_layout1)
        Other_layout.addWidget(self.roi_fast_btn)
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        Other_layout.addWidget(separator)
        self.data_name_label = QLabel()
        self.signal_extract = QPushButton("时序信号快速提取")
        Other_layout.addWidget(self.signal_extract)
        self.tDFT_btn2 = QPushButton("二维傅里叶变换")
        Other_layout.addWidget(self.tDFT_btn2)
        self.atam_btn2 = QPushButton("累计时间振幅图")
        Other_layout.addWidget(self.atam_btn2)
        self.heartbeat_btn = QPushButton("心肌细胞跳动分析")
        Other_layout.addWidget(self.heartbeat_btn)
        Other_GROUP.setLayout(Other_layout)
        self.between_stack.addWidget(Other_GROUP)

    def between_stack_change(self):
        if self.fuction_select.currentIndex() == 0: # nothing
            self.between_stack.setCurrentIndex(0)
            self.FS_mode_combo.setCurrentIndex(0)
        if self.fuction_select.currentIndex() == 1:  # FS-iSCAT & PA
            self.between_stack.setCurrentIndex(1)
            self.FS_mode_combo.setCurrentIndex(1)
            self.update_status('准备就绪')
            self.fps_input.setEnabled(False)
            self.time_step_input.setEnabled(True)
            self.time_unit_combo.setEnabled(True)
        if self.fuction_select.currentIndex() == 2:  # ES-iSCAT
            self.between_stack.setCurrentIndex(2)
            self.update_status('准备就绪')
            self.fps_input.setEnabled(True)
            self.time_step_input.setEnabled(False)
            self.time_unit_combo.setEnabled(False)
        if self.fuction_select.currentIndex() == 3:
            self.between_stack.setCurrentIndex(3)
            self.update_status('准备就绪')

    def mode_switch_change(self, mode:int):
        """模式改变后"""
        self.mode = mode
        if mode == 0:
            self.mode_label.setText('ROI模式')
            self.mode_explain.setText("每次处理前都需要选择ROI，\n数据也需要选择，ROI需要与数据匹配")
        elif mode == 1:
            self.mode_label.setText('默认模式')
            self.mode_explain.setText("按标准处理流程执行的快速操作，\n数据源会自动选择")
        elif mode == 2:
            self.mode_label.setText('自由模式')
            self.mode_explain.setText("每次处理前都需要选择数据，\n数据自由选择，但可能会报错（无法处理）")
        colors = ["#34C759", "#999999", "#007AFF"]

        self.mode_label.setStyleSheet(f"font-size: 18px; font-weight: bold; color: {colors[mode]}")

    def setup_menus(self):
        """加入菜单栏"""
        self.menu = self.menuBar()
        self.menu.addMenu('主窗口')

        # 控制台
        view_menu = self.menu.addMenu("控制台")
        toggle_console = view_menu.addAction("显示/隐藏控制台")
        toggle_console.triggered.connect(lambda: self.console_dock.setVisible(not self.console_dock.isVisible()))

        # 编辑菜单
        edit_menu = self.menu.addMenu("编辑")

        # 编辑菜单-坏点处理功能
        bad_frame_edit = edit_menu.addAction("坏点处理")
        bad_frame_edit.triggered.connect(self.bad_frame_edit_dialog)

        # 编辑菜单-计算设置功能
        data_select_edit = edit_menu.addAction("计算设置")
        data_select_edit.triggered.connect(self.calculation_set_edit_dialog)

        # 编辑菜单-绘图设置调整
        plt_settings_edit = edit_menu.addAction("绘图设置")
        plt_settings_edit.triggered.connect(self.plt_settings_edit_dialog)

        # # ROI绘图
        # ROI_function = self.menu.addAction("ROI选取")
        # ROI_function.triggered.connect(self.roi_select_dialog)

        # 历史数据管理
        data_menu = self.menu.addMenu('历史数据')
        # 清除历史
        data_history_clear = data_menu.addAction('历史清除')
        data_history_clear.triggered.connect(self.data_history_clear)
        # 数据导入历史查看
        data_history_view = data_menu.addAction('历史导入查看')
        data_history_view.triggered.connect(self.data_history_view) # 临时
        # 数据处理历史查看
        process_history_view = data_menu.addAction('历史处理查看')
        process_history_view.triggered.connect(self.process_history_view)
        # 详细历史查看
        data_all_view = data_menu.addAction("所有历史详情查看")
        data_all_view.triggered.connect(self.data_plot_add)

        # 指南帮助
        help_menu = self.menu.addMenu('使用指南')
        all_help = help_menu.addAction('全部指南')
        all_help.triggered.connect(lambda: self.help_show('程序使用指南大全'))
        fs_help = help_menu.addAction('超快成像')
        fs_help.triggered.connect(lambda: self.help_show('超快成像分析帮助',["general","lifetime"]))
        EM_help = help_menu.addAction('电化学调制iSCAT')
        EM_help.triggered.connect(lambda: self.help_show('电化学调制分析帮助',["general","stft","cwt"]))
        about_action = help_menu.addAction("关于")


        update_action = self.menu.addAction('检查更新')
        update_action.triggered.connect(self.update_dialog)

    @staticmethod
    def QGroupBoxCreator(title="",style="default"):
        # 全局Box样式定义
        group_box = QGroupBox(title)
        styles = {
            "default": """
            QGroupBox{
                border:1px solid #aaaaaa;
                border-radius:5px;
                margin-top:5px;
                padding:15px;
                padding-left: 5px;
                padding-right: 5px;
            }
            QGroupBox::title{
                ubcontrol-origin: margin;
                left: 10px;
                padding: 0 3px;
                color: #2E7D32;
                font-weight: 1000;
            }
            """,
            "inner":"""
            QGroupBox{
                border: 1px solid #aaaaaa;
                border-radius: 5px;
                margin-top: 5px;
                padding: 5px;
                padding-left: 0px;
                padding-right: 0px;
            }""",
            "noborder":"""
            QGroupBox{
                border: 0px;
                border-radius: 0px;
                margin: 0px;
                padding:0px;
            }"""
        }
        group_box.setStyleSheet(styles.get(style, styles["default"]))
        return group_box

    def setup_status_bar(self):
        """设置状态条"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # 状态文本
        self.status_label = QLabel("准备就绪")
        self.status_label.setFixedWidth(250)
        self.status_bar.addWidget(self.status_label)
        # 鼠标悬停显示
        self.mouse_pos_label = QLabel("光标位置: x= -, y= -, t= -; 值: -")
        self.mouse_pos_label.setFixedWidth(500)
        self.status_bar.addWidget(self.mouse_pos_label)
        self._handle_hover = self.make_hover_handler()
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFixedWidth(650)
        self.status_bar.addWidget(self.progress_bar)

        # 状态指示灯 (红绿灯)
        self.status_light = QLabel()
        self.status_light.setPixmap(QPixmap(":/icons/green_light.png").scaled(16, 16))
        self.status_bar.addPermanentWidget(self.status_light)

    """控制台相关"""
    def setup_console(self):
        """设置控制台停靠窗口"""
        self.console_dock = QDockWidget("控制台", self)
        self.console_dock.setObjectName("ConsoleDock")

        # 创建控制台部件
        self.console_widget = ConsoleWidget(self)
        self.command_processor = CommandProcessor(self)

        self.console_dock.setWidget(self.console_widget)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.console_dock)
        self.splitDockWidget(self.plot_dock, self.console_dock, Qt.Vertical)
        self.resizeDocks([self.plot_dock, self.console_dock], [300, 100], Qt.Vertical)
        # 设置控制台特性
        self.console_dock.setMinimumWidth(200)
        self.console_dock.setMinimumHeight(50)
        self.console_dock.setFeatures(QDockWidget.DockWidgetMovable |
                                      QDockWidget.DockWidgetFloatable |
                                      QDockWidget.DockWidgetClosable)

    def get_log_path(self):
        """生成配置文件地址"""
        if hasattr(sys, '_MEIPASS'):  # 检测是否在PyInstaller打包环境中运行
            # 使用os.environ获取标准路径
            appdata_local = os.environ.get('LOCALAPPDATA')
            appdata_local = os.path.join(appdata_local, 'LifeCalor')
        else:  # 开发环境
            appdata_local = os.path.dirname(os.path.abspath(__file__))

        os.makedirs(appdata_local, exist_ok=True)

        # 设置日志文件路径
        return os.path.join(appdata_local, "carrier_lifetime.log")

    def setup_logging(self):
        """配置日志系统"""
        # 确保日志目录存在
        log_dir = os.path.dirname(self.log_file)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # 设置轮转文件处理器 (每个文件最大5MB，保留3个备份)
        file_handler = RotatingFileHandler(
            self.log_file, maxBytes=5 * 1024 * 1024, backupCount=3, encoding='utf-8'
        )
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))

        # 设置控制台处理器
        console_handler = ConsoleHandler(self)

        # 配置根日志记录器
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        logger.handlers.clear()  # 清除现有处理器
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        sys.stdout = StreamLogger(logging.INFO)
        sys.stderr = StreamLogger(logging.ERROR)

    def log_to_console(self, message):
        """将消息输出到控制台"""
        self.console_widget.console_output.append(message)
        self.console_widget.console_output.verticalScrollBar().setValue(
            self.console_widget.console_output.verticalScrollBar().maximum()
        )

    def log_startup_message(self):
        """记录程序启动消息"""
        startup_msg = f"""\n
============================================
成像数据分析工具箱启动
启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
日志位置: {self.log_file}
程序版本: {self.current_version}
============================================
        """
        logging.info(startup_msg.strip())
        logging.info("程序已进入准备状态，等待用户操作...（第一次计算可能较慢）")

    def help_show(self,title,topics :list = None):
        self.help_dialog = CustomHelpDialog(title, topics=topics, parent=self)
        self.help_dialog.show()

    """程序更新"""
    def update_dialog(self):
        """显示更新对话框
        注：更新功能中使用的线程都是直接继承QThread的方法，因为功能比较简单"""
        dialog = UpdateDialog(self)
        dialog.download_progress.connect(self.update_progress)
        dialog.update_status.connect(self.update_status)
        dialog.exec_()
        self.update_status("准备就绪")

    def auto_update_check(self):
        """设置自动检查更新"""
        # 检查设置，避免过于频繁检查
        self.settings.beginGroup("sys")
        last_check = self.settings.value("last_update_check")
        should_check = self.settings.value("should_check", True, type=bool)
        self.settings.endGroup()

        # 如果从未检查过或超过24小时，则检查更新
        now = datetime.now()

        if last_check:
            last_check_date = datetime.fromisoformat(last_check)
            if (now - last_check_date).days < 1:  # 1天内检查过
                should_check = False

        if should_check:
            # 延迟2秒启动检查，避免影响程序启动
            QTimer.singleShot(2000, self.check_updates_on_startup)

            # 更新最后检查时间
            self.settings.beginGroup("sys")
            self.settings.setValue("last_update_check", now.isoformat())
            self.settings.endGroup()

    def check_updates_on_startup(self):
        """启动时检查更新"""
        logging.info("正在检查更新...")
        self.update_status("检查更新中",'working')

        self.startup_checker = UpdateChecker(
            self.repo_owner,
            self.repo_name,
            self.current_version,
            self.PAT
        )
        self.startup_checker.version_info.connect(self.handle_startup_update)
        self.startup_checker.check_completed.connect(self.handle_check_completed)
        self.startup_checker.start()

    def handle_startup_update(self, update_info):
        """处理启动时发现的更新"""
        if not update_info['update_available']:
            return

        logging.info(f"发现新版本 v{update_info['latest_version']}")

        # 更新状态栏提示
        self.update_status("有新版本可用",'idle')

        # 创建并显示更新对话框
        self.dialog = UpdateDialog(self, startup_check=True)
        self.dialog.update_info = update_info

        # 直接显示更新信息，不需要再次检查
        self.dialog.check_button.setEnabled(True)
        self.dialog.update_button.setEnabled(True)
        self.dialog.status_label.setText(f"发现新版本: v{update_info['latest_version']}")
        self.dialog.log_message(f"发现新版本 v{update_info['latest_version']}")
        self.dialog.show_release_notes(update_info.get('release_notes', '暂无更新说明'))
        self.dialog.tab_widget.setCurrentIndex(1)

        # 显示对话框
        self.dialog.show()
        self.dialog.raise_()
        self.dialog.activateWindow()

    def handle_check_completed(self, has_update):
        """处理检查完成"""
        if not has_update:
            QMessageBox.information(self,"提示","当前已是最新版本")
            logging.info("已是最新版本")
            self.update_status("准备就绪",'idle')

    """线程与信号连接"""
    def import_thread_open(self):
        """数据导入线程开启（.11.1版本加入）"""
        self.import_thread = QThread()
        self.imp_thread = ImportManager()
        self.imp_thread.moveToThread(self.import_thread)
        # 信号连接
        self.data_import_signal.connect(self.imp_thread.import_dispatch)
        self.imp_thread.update_status.connect(self.update_status)
        self.imp_thread.update_QMessageBox.connect(self.update_QMessageBox)
        self.imp_thread.processing_progress_signal.connect(self.update_progress)
        self.imp_thread.import_finished.connect(self.import_result)

    def data_thread_open(self):
        """图像数据操作线程（.10.10版本加入 ）和例外数据处理线程（.11.2版本加入）"""
        self.data_thread = QThread()
        self.dat_thread = DataManager()
        self.dat_thread.moveToThread(self.data_thread)

        self.image_display.image_style_change_signal.connect(self.dat_thread.to_colormap)
        self.image_display.image_export_signal.connect(self.dat_thread.export_data)
        self.dat_thread.data_progress_signal.connect(self.update_progress)
        self.dat_thread.process_finish_signal.connect(self.image_display.update_canvas_by_stamp)
        self.mass_export_signal.connect(self.dat_thread.export_data)
        self.roi_processed_signal.connect(self.dat_thread.ROI_processed)
        self.dat_thread.processed_result.connect(self.processed_result)

        #
        self.process_thread = QThread()
        self.proc_thread = DataProcessor()
        self.proc_thread.moveToThread(self.process_thread)
        self.amend_data_signal.connect(self.proc_thread.amend_data)
        self.detect_bad_frames_auto_signal.connect(self.proc_thread.detect_bad_frames_auto)
        self.fix_bad_frames_signal.connect(self.proc_thread.fix_bad_frames)
        self.proc_thread.plot_singal.connect(self.graph_plot.handle_plot_signal)
        self.proc_thread.plot_series_signal.connect(self.graph_plot.handle_from_image)

    def cal_thread_open(self):
        """计算线程相关 以及信号槽连接都放在这里了"""
        self.calc_thread = QThread()
        self.cal_thread = CalculationThread()
        self.cal_thread.moveToThread(self.calc_thread)
        # 计算状态更新
        self.start_reg_cal_signal.connect(self.cal_thread.region_analyze)
        self.start_dis_cal_signal.connect(self.cal_thread.distribution_analyze)
        self.start_heat_cal_signal.connect(self.cal_thread.heat_transfer_calculation)
        self.start_dif_cal_signal.connect(self.cal_thread.diffusion_calculation)
        self.cal_thread.calculating_progress_signal.connect(self.update_progress)
        self.cal_thread.processed_result.connect(self.processed_result)
        # self.cal_thread.stop_thread_signal.connect(self.stop_thread)
        self.cal_thread.cal_running_status.connect(self.btn_safety)
        self.cal_thread.update_status.connect(self.update_status)
        self.easy_process.connect(self.cal_thread.easy_process)

    def EM_thread_open(self):
        """加载EM文件的线程开启"""
        # 初始化数据处理线程
        self.avi_thread = QThread()
        self.mass_data_processor = MassDataProcessor()
        self.mass_data_processor.moveToThread(self.avi_thread)
        self.mass_data_processor.processing_progress_signal.connect(self.update_progress)
        self.mass_data_processor.processed_result.connect(self.processed_result)
        self.pre_process_signal.connect(self.mass_data_processor.pre_process)
        self.stft_python_signal.connect(self.mass_data_processor.python_stft)
        self.stft_quality_signal.connect(self.mass_data_processor.quality_stft)
        self.cwt_quality_signal.connect(self.mass_data_processor.quality_cwt)
        self.cwt_python_signal.connect(self.mass_data_processor.python_cwt)
        self.atam_signal.connect(self.mass_data_processor.accumulate_amplitude)
        self.tDgf_signal.connect(self.mass_data_processor.twoD_gaussian_fit)
        self.sscs_signal.connect(self.mass_data_processor.simple_single_channel)
        self.tDFT_signal.connect(self.mass_data_processor.twoD_fourier_transform)
        self.heartbeat_signal.connect(self.mass_data_processor.heartbeat_movement)

        # self.avi_thread.start()

    def signal_connect(self):
        # 连接参数区域按钮
        self.fuction_select.currentIndexChanged.connect(self.funtion_stack.setCurrentIndex)
        self.fuction_select.currentIndexChanged.connect(self.between_stack_change)
        self.file_type_selector1.currentIndexChanged.connect(self.file_type_stack1.setCurrentIndex)
        self.file_type_selector.currentIndexChanged.connect(self.file_type_stack.setCurrentIndex)
        self.tiff_folder_btn.clicked.connect(self.load_tiff_folder)
        self.sif_folder_btn.clicked.connect(self.load_sif_folder)
        self.avi_select_btn.clicked.connect(self.load_avi)
        self.EMtiff_folder_btn.clicked.connect(self.load_tiff_folder_EM)
        self.analyze_region_btn.clicked.connect(self.region_analyze_start)
        self.analyze_btn.clicked.connect(self.distribution_analyze_start)
        self.heat_transfer_btn.clicked.connect(self.heat_transfer_start)
        self.FS_mode_combo.currentIndexChanged.connect(self.FS_mode_stack.setCurrentIndex)
        # self.PA_mode_combo.currentIndexChanged.connect(self.PA_mode_stack.setCurrentIndex)
        self.EM_mode_combo.currentIndexChanged.connect(self.EM_mode_stack.setCurrentIndex)
        self.after_process_select.currentIndexChanged.connect(self.after_process_stack.setCurrentIndex)
        self.preprocess_data_btn.clicked.connect(self.pre_process_EM)
        self.stft_process_btn.clicked.connect(self.process_EM_stft)
        self.stft_quality_btn.clicked.connect(self.quality_EM_stft)
        self.cwt_quality_btn.clicked.connect(self.quality_EM_cwt)
        self.cwt_process_btn.clicked.connect(self.process_EM_cwt)
        self.EM_output_btn.clicked.connect(self.export_EM_data)
        self.atam_btn.clicked.connect(self.process_atam)
        self.atam_btn2.clicked.connect(self.process_atam)
        self.tDgf_btn.clicked.connect(self.process_tDgf)
        self.sscs_btn.clicked.connect(self.process_simple_scs)
        self.roi_signal_btn.clicked.connect(self.roi_signal_avg)
        self.tDFT_btn.clicked.connect(self.process_tDFT)
        self.tDFT_btn2.clicked.connect(self.process_tDFT)
        self.vector_signal_btn.clicked.connect(self.vectorROI_signal_show)
        self.select_frames_btn.clicked.connect(self.vectorROI_selection)
        self.diffusion_coefficient_btn.clicked.connect(self.result_display.plot_variance_evolution)
        self.roi_fast_btn.clicked.connect(self.fast_roi_result)
        self.heartbeat_btn.clicked.connect(self.process_heartbeat)
        self.signal_extract.clicked.connect(self.process_signal_avg)
        self.export_image_btn.clicked.connect(self.export_image)
        self.export_data_btn.clicked.connect(self.export_data)
        # 成像绘制信号
        self.image_display.add_canvas_signal.connect(self.add_new_canvas)
        self.image_display.draw_result_signal.connect(self.draw_result)
        self.image_display.params_update_signal.connect(lambda params : self.tool_params.update(params))
        # 时间滑块
        # self.time_slider.valueChanged.connect(self.image_display.update_time_slice)
        self.time_slider_vertical.valueChanged.connect(self.update_result_display)
        # 连接控制台信号
        self.command_processor.terminate_requested.connect(self.stop_calculation)
        self.command_processor.save_config_requested.connect(self.save_config)
        self.command_processor.load_config_requested.connect(self.load_config)
        self.command_processor.clear_result_requested.connect(self.clear_result)
        # 结果区域信号
        self.result_display.tab_type_changed.connect(self._handle_result_tab)
        self.add_data_btn.clicked.connect(self.data_plot_add)
        self.reset_data_btn.clicked.connect(self.data_plot_clear)

    def canvas_signal_connect(self):
        self.roi_pick.clear()
        for canvas in self.image_display.display_canvas:
            canvas.disconnect()
            canvas.mouse_position_signal.connect(self._handle_hover)
            canvas.mouse_clicked_signal.connect(self._handle_click)
            canvas.current_canvas_signal.connect(self.image_display.set_cursor_id)
            canvas.draw_result_signal.connect(self.draw_result)
            canvas.get_fast_selection.connect(self.proc_thread.get_fast_selection)
            self.roi_pick.addItem(canvas.windowTitle())

    '''上面是初始化预设，下面是功能响应'''
    """数据导入相关"""
    def get_data_all(self) ->  List[Dict[str, Any]]:
        Data_list = []
        if self.data is None:
            return []
        # 直接读取历史数据
        for data in self.data.history:
            Data_list.append({
                "type": 'Data',
                "name": data.name,
                "序号": data.serial_number,
                "导入格式": data.format_import,
                "数据大小": data.datashape,
                "timestamp": data.timestamp,
            })
            Data_list.reverse()
        return Data_list

    def get_processed_data_all(self) ->  List[Dict[str, Any]]:
        ProcessedData_list = []
        if self.processed_data is None:
            return []
        # 直接读取历史数据
        for processed in self.processed_data.history:
                ProcessedData_list.append({
                    "type": "ProcessedData",
                    "name": processed.name,
                    "处理类型": processed.type_processed,
                    "数据大小": processed.datashape,
                    "数据源": self._find_parent_name(processed.timestamp_inherited),
                    "timestamp": processed.timestamp,
                })
        ProcessedData_list.reverse()
        return ProcessedData_list

    def _find_parent_name(self, timestamp: float) -> Optional[str]:
        """通过时间戳查找父数据名称"""
        # 首先在原始数据中查找
        for data in list(self.data.history):
            if data.timestamp == timestamp:
                return data.name

        # 然后在处理数据中查找
        for processed in list(self.processed_data.history):
            if processed.timestamp == timestamp:
                return processed.name

        return None

    def load_tiff_folder(self):
        """加载TIFF文件夹(FS-iSCAT)"""
        self.time_step = float(self.time_step_input.value())
        folder_path = QFileDialog.getExistingDirectory(self, "选择TIFF图像文件夹",self.settings.value("last_folder", ""))
        if folder_path:
            logging.info(folder_path)
            self.update_status("已加载TIFF文件夹",'idle')
            current_group = self.group_selector.currentText()
            self.data_import_signal.emit('tif_series', folder_path, {'current_group':current_group,**self.basic_params})

        elif not folder_path:
            self.update_status("文件夹选择已取消", 'idle')
            return

    def load_sif_folder(self):
        '''加载SIF文件夹'''
        folder_path = QFileDialog.getExistingDirectory(self, "选择SIF图像文件夹",self.settings.value("last_folder", ""))
        if folder_path:
            logging.info(folder_path)
            self.update_status("已加载SIF文件夹",'idle')

            # 读取文件夹中的所有sif文件
            self.data_import_signal.emit('sif_folder',folder_path,{'normalize_type' : self.method_combo.currentText(),**self.basic_params})

        elif not folder_path:
            self.update_status("文件夹选择已取消", 'idle')
            return

    def load_avi(self):
        """加载avi读取线程传递函数"""
        self.status_label.setText("正在处理数据...")
        self.avi_thread.start()
        file_types = "AVI视频文件 (*.avi);;所有文件 (*)"

        # 获取文件路径
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择AVI视频文件",
            self.settings.value("last_folder", ""),  # 起始目录
            file_types
        )

        if not file_path:
            self.update_status("文件夹选择已取消", 'idle')
            logging.info("用户取消选择")
            return

        logging.info(f"已选择AVI文件: {file_path}")
        self.update_status("正在加载AVI文件...",'working')

        self.data_import_signal.emit('avi_EM',file_path, {'fps':self.fps_input.value(),**self.basic_params})
        self.update_param('EM','EM_fps', self.fps_input.value())

    def load_tiff_folder_EM(self):
        """加载TIFF文件夹(FS-iSCAT)"""
        self.status_label.setText("正在处理数据...")
        self.avi_thread.start()
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "选择TIFF图像序列文件夹",
            self.settings.value("last_folder", "")
        )

        if not folder_path:
            # self.folder_path_label.setText("未选择文件夹")
            logging.info("用户取消选择")
            return

        for f in os.listdir(folder_path):
            if f.lower().endswith(('.tif', '.tiff')):
                self.data_import_signal.emit('tif_EM',folder_path, {'fps':self.fps_input.value(),**self.basic_params})
                self.update_status("正在加载tiff文件...",'working')
                self.update_param('EM','EM_fps', self.fps_input.value())
                return
            else:
                self.update_status("文件夹中没有TIFF文件",'warning')
                return

    def import_result(self, data):
        """导入的数据放到这里来处理"""
        self.data = data
        self.settings.setValue("last_folder", os.path.dirname(data.parameters['file_path']))
        logging.info(f'成功加载{data.format_import}数据({data.name})')
        self.update_status("已加载文件", 'idle')
        # 成像显示
        self.load_image(origin_data=self.data)

    """画布设置相关"""
    def add_new_canvas(self, assign_data=None):
        """新建图像显示画布"""
        if self.data is None and self.processed_data is None:
            logging.warning('请先导入或处理数据')
            return
        self.update_progress(-1)
        data_display = None
        if assign_data is not None:
            data_display = ImagingData.create_image(assign_data)
            data_display.colormode = self.tool_params['colormap'] if self.tool_params['use_colormap'] else None
            logging.info("数据选择成功（原初）") if isinstance(assign_data, Data) else logging.info("数据选择成功（处理）""")
        else:
            dialog = DataViewAndSelectPop(datadict=self.get_data_all(),
                                          processed_datadict=self.get_processed_data_all(), add_canvas=True)
            if dialog.exec_():
                selected_timestamp, selected_table = dialog.get_selected_timestamp()
                if selected_table == 'data':
                    for data in self.data.history:
                        if data.timestamp == selected_timestamp:
                            data_display = ImagingData.create_image(data)
                            data_display.colormode = self.tool_params['colormap'] if self.tool_params[
                                'use_colormap'] else None
                            logging.info("数据选择成功（原初）")
                            break
                else:
                    for data in self.processed_data.history:
                        if data.timestamp == selected_timestamp:
                            data_display = ImagingData.create_image(data)
                            data_display.colormode = self.tool_params['colormap'] if self.tool_params[
                                'use_colormap'] else None
                            logging.info("数据选择成功（处理）")
                            break
        if data_display is not None:
            self.image_display.add_canvas(data_display)
            self.focus_canvas = self.image_display.cursor_id
        # else:
        #     QMessageBox.warning(self,"数据错误","数据已经遗失（不可能错误）")
        #     return
        self.canvas_signal_connect()
        # self.image_display.update_time_slice(0, True)

        self.update_status("准备就绪", 'idle')

    def load_image(self, data_type='original', other_params: str = None, origin_data=None):
        """图像加载，后面会进一步修改"""
        if len(self.image_display.display_canvas) == 0:  # 初次创建
            # self.add_new_canvas()
            if data_type == 'original':
                self.imaging_main = ImagingData.create_image(self.data)
            self.image_display.add_canvas(self.imaging_main)
            totalframes = self.imaging_main.totalframes
            # self.time_slider.setMaximum(totalframes - 1)
            # self.time_label.setText(f"时间点: 0/{totalframes - 1}")
            self.canvas_signal_connect()
        else:
            if data_type == 'original':
                # imports_done = self.other_imports
                msg_box = QMessageBox()
                msg_box.setWindowTitle("画布操作")
                msg_box.setText("请选择是否要覆盖当前画布或新建画布")

                # 添加标准按钮
                overwrite_btn = msg_box.addButton("覆盖", QMessageBox.ActionRole)
                new_btn = msg_box.addButton("新建", QMessageBox.ActionRole)
                hide_btn = msg_box.addButton("隐藏", QMessageBox.ActionRole)
                msg_box.exec_()

                # 返回结果
                if msg_box.clickedButton() == overwrite_btn:
                    self.image_display.del_canvas(-1)
                    self.imaging_main = ImagingData.create_image(self.data)
                    totalframes = self.imaging_main.totalframes
                    # self.time_slider.setMaximum(totalframes - 1)
                    # self.time_label.setText(f"时间点: 0/{totalframes - 1}")
                    self.add_new_canvas(origin_data)
                elif msg_box.clickedButton() == new_btn:
                    self.add_new_canvas(origin_data)
                elif msg_box.clickedButton() == hide_btn:
                    return False

        # 显示第一张图像
        # self.image_display.update_time_slice(0, True)
        # self.time_slider.setValue(0)

        # 根据图像大小调节region范围
        self.region_x_input.setMaximum(self.data.datashape[1])
        self.region_y_input.setMaximum(self.data.datashape[2])

    def other_imports(self): # 没用
        msg_box = QMessageBox()
        msg_box.setWindowTitle("画布操作")
        msg_box.setText("请选择是否要覆盖当前画布或新建画布")

        # 添加标准按钮
        overwrite_btn = msg_box.addButton("覆盖", QMessageBox.ActionRole)
        new_btn = msg_box.addButton("新建", QMessageBox.ActionRole)
        hide_btn = msg_box.addButton("隐藏", QMessageBox.ActionRole)
        msg_box.exec_()

        # 返回结果
        if msg_box.clickedButton() == overwrite_btn:
            return "overwrite"
        elif msg_box.clickedButton() == new_btn:
            self.add_new_canvas('latest')
            return "new"
        elif msg_box.clickedButton() == hide_btn:
            return "hide"
        return None

    def make_hover_handler(self):
        args = {'x': None, 'y': None, 't': None, 'value': None, 'origin': None}
        def _handle_hover(x=None, y=None, t=None, value=None, origin=None):
            """鼠标位置显示"""
            # 更新传入的参数（未传入的保持原值）
            if x is not None: args['x'] = x
            if y is not None: args['y'] = y
            if t is not None: args['t'] = t
            if value is not None:
                args['value'] = value
            else:
                args['value'] = self.data.image_import[args['t'], args['y'], args['x']]
            if args['x'] is None or args['y'] is None:
                return
            if origin is not None: args['origin'] = origin

            # 更新显示
            self.mouse_pos_label.setText(
                f"光标位置: x={args['x']}, y={args['y']}, t={args['t']}; 图像值: {args['value']}, 实际值：{args['origin']:.3f}")

        return _handle_hover

    def _handle_click(self, x, y, id):
        """处理图像点击事件"""
        if self.FS_mode_combo.currentIndex() == 1 :  # 区域分析模式 or self.PA_mode_combo.currentIndex() == 0
            self.region_x_input.setValue(x)
            self.region_y_input.setValue(y)
            self.focus_canvas = id
    """编辑设置对话框"""
    def bad_frame_edit_dialog(self):
        """显示坏点处理对话框"""
        if self.data is None and self.processed_data is None:
            logging.warning("无数据，请先加载数据文件")
            return

        dialog = BadFrameDialog(self)
        self.update_status("坏点修复ing", 'working')
        if dialog.exec_():
            # 更新图像显示
            # self.time_label.setText(self.image_display.update_time_slice(0))
            # self.time_slider.setValue(0)
            logging.info(f"坏点处理完成，修复了 {len(dialog.bad_frames)} 个坏帧")
        self.update_status("准备就绪", 'idle')

    def calculation_set_edit_dialog(self):
        """计算设置调整"""
        if self.data is None or self.processed_data is None:
            logging.warning("无数据，请先加载数据文件")
            return
        self.update_status("计算设置ing", 'working')
        dialog = CalculationSetDialog(self.cal_set_params, parent=self)
        if dialog.exec_():
            # self.time_label.setText(self.image_display.update_time_slice(0))
            # self.time_slider.setValue(0)
            self.cal_set_params = dialog.params
            LifetimeCalculator.set_cal_parameters(self.cal_set_params)
            # 同步修改绘图设置并传参
            self.plot_params['_from_start_cal'] = self.cal_set_params['from_start_cal']
            self.result_display.update_plot_settings(self.plot_params, update=False)
            logging.info("设置已更新，请重新绘图")
        self.update_status("准备就绪", 'idle')

    def plt_settings_edit_dialog(self):
        """绘图设置"""
        dialog = PltSettingsDialog(params=self.plot_params, parent=self)
        self.update_status("绘图设置ing", 'working')
        if dialog.exec_():
            # 将参数传递给ResultDisplayWidget
            self.result_display.update_plot_settings(dialog.params)
            self.plot_params = dialog.params
            logging.info("绘图已更新")
        self.update_status("准备就绪", 'idle')

    def start_calculation(self):
        """开始计算时调用此方法"""
        self.elapsed_timer.start()
        self.last_time = 0
        self.last_progress = 0
        self.last_percent = -1
        self.cached_remaining = "计算中..."

    """状态响应与更新"""
    def update_progress(self, current, total=None):
        """更新进度条"""
        if total is not None:
            self.progress_bar.setMaximum(total)

        self.progress_bar.setValue(current)
        # 计算当前进度
        current_percent = current / self.progress_bar.maximum() * 100 if self.progress_bar.maximum() > 0 else 0
        elapsed_ms = self.elapsed_timer.elapsed()
        elapsed_sec = elapsed_ms / 1000.0

        # 只有当进度变化超过1%时才更新剩余时间
        if int(current_percent) > self.last_percent:

            # 计算剩余时间（仅当进度变化超过1%时）
            if current > self.last_progress and elapsed_ms > self.last_time:
                # 计算速度
                progress_diff = current - self.last_progress
                time_diff = (elapsed_ms - self.last_time) / 1000.0
                speed = progress_diff / time_diff if time_diff > 0 else 0

                # 更新记录点
                self.last_progress = current
                self.last_time = elapsed_ms

                # 计算并缓存剩余时间
                if speed > 0 and total is not None:
                    remaining_sec = (total - current) / speed
                    self.cached_remaining = self.format_time(remaining_sec)

            # 更新百分比记录
            self.last_percent = int(current_percent)

        # 格式化时间显示
        elapsed_str = self.format_time(elapsed_sec)

        # 更新进度条格式
        self.progress_bar.setFormat(
            f"进度: {current}/{self.progress_bar.maximum()} "
            f"({current_percent:.1f}%) | "
            f"已用: {elapsed_str} | 预计剩余: {self.cached_remaining}"
        )

        # self.console_widget.update_progress(current, total)

        if current == 0: # 我老是忘记到底是1启动还是0启动，干脆兼容吧， 还得是0
            self.start_calculation() # 启动计时器
        elif current >= self.progress_bar.maximum():
            logging.info(f"计算完成，总耗时{elapsed_str}")
            self.update_status("进程任务完成,准备就绪",'idle')
            self.progress_bar.reset()
        elif current == -1:
            self.progress_bar.reset()

    def _handle_result_tab(self, tab_type):
        """特殊标签页类型处理"""
        if tab_type == 'roi':
            # 如果是roi结果
            self.time_slider_vertical.setVisible(True)
            self.time_slider_vertical.setMaximum(self.data.timelength - 1)
            self.time_slider_vertical.setValue(0)
        elif tab_type == 'pre-scs':
            self.time_slider_vertical.setVisible(True)
            self.time_slider_vertical.setMaximum(int(self.processed_data.out_processed['mean_signal'].max() * 10))
        else:
            if self.time_slider_vertical.isVisible():
                self.time_slider_vertical.setVisible(False)

    def update_result_display(self,idx,reuse_current=True):
        """目前有两个地方用到垂直滚动条"""
        data = self.processed_data
        if self.vector_array is not None and 0 <= idx < self.vector_array.shape[0]:
            frame_data = self.vector_array[idx]
            self.result_display.display_roi_series(
                positions=frame_data[:, 0],
                intensities=frame_data[:, 1],
                fig_title=f"ROI信号强度 (帧:{idx})",
                reuse_current = reuse_current

            )
        elif self.processed_data.type_processed == 'Single_channel_signal' and not data.out_processed['thr_known']:
            thr = idx/10
            for m in range(self.processed_data.framesize):
                if data.out_processed['mean_signal'][m] > thr:
                    data.data_processed[m] = data.out_processed['mean_signal'][m]
                else:
                    data.data_processed[m] = data.out_processed['whole_mean'][m]
            data.out_processed['thr'] = thr
            self.result_display.single_channel(data,False,
                reuse_current = reuse_current)
        else:
            logging.debug("结果垂直滚动条失去更新源，不可能错误")

    def update_status(self, status, working_status='idle'):
        """更新状态条的显示"""
        self.status_label.setText(status)
        if working_status == 'idle' : # idle
            light = "green_light.png"
        elif working_status == 'working' :
            light = "yellow_light.png"
        elif working_status == 'warning':
            light = "red_light.png"
            logging.warning(status)
        elif working_status == 'error':
            QMessageBox.warning(self,'错误！',status)
            logging.error(status)
            light = "green_light.png"
        else:
            light = "red_light.png"
        self.status_light.setPixmap(QPixmap(f":/icons/{light}").scaled(16, 16))

    def update_QMessageBox(self, message_type, title, message):
        if message_type == 'warning':
            QMessageBox.warning(self, title, message)
        elif message_type == 'error':
            QMessageBox.error(self, title, message)
        elif message_type == 'information':
            QMessageBox.information(self, title, message)

    """各种计算方法"""
    def vectorROI_signal_show(self):
        """向量选取信号全部展示"""
        if not hasattr(self, 'data') or self.data is None:
            logging.warning("无数据，请先加载数据文件")
            return
        if self.vector_array is None :
            logging.warning("未选取向量直线ROI")
            return
        elif self.data.timelength == self.vector_array.shape[0]:
            # self.time_slider_vertical.setVisible(True)
            # self.time_slider_vertical.setMaximum(self.data['data_origin'].shape[0] - 1)
            # self.time_slider_vertical.setValue(0)
            self.update_result_display(0,reuse_current = False)
            return
        else:
            logging.error("数据长度不匹配")
            return

    def vectorROI_selection(self):
        """向量选取信号选择展示"""
        self.max_frame = self.vector_array.shape[0] - 1
        frames = ToolBucket.parse_frame_input(self.frame_input.toPlainText(),self.max_frame)
        if not frames :
            logging.warning("请输入选取的帧数")
        elif not hasattr(self, 'vector_array'):
            logging.warning("请选择矢量直线绘制ROI选区")
            return
        elif frames is None:
            logging.warning("帧数无效，请重新输入")
            return

        # 检查帧数是否有效
        invalid_frames = [f for f in frames if f < 0 or f > self.max_frame]
        if invalid_frames:
            QMessageBox.warning(
                self, "帧数超出范围",
                f"有效帧数范围: 0-{self.max_frame}\n无效帧: {invalid_frames}"
            )
            logging.warning("请输入有效帧数")
            frames = [f for f in frames if 0 <= f <= self.max_frame]
            if not frames:
                return
        else:
            logging.info(f"输入帧数{frames}，帧数有效可以处理")

        # 收集选定帧的数据
        self.vectorROI_data = {f: self.vector_array[f] for f in frames}

        # 信号拟合和绘制
        self.diffusion_calculation_start()

        # # 自动显示方差演化图
        # self.display_diffusion_coefficient()

    def region_analyze_start(self):
        """分析选定区域载流子寿命"""
        aim_data = self.data_selection('data')
        if aim_data is None:
            return False
        # 如果线程没了，要开启
        if not self.is_thread_active("calc_thread"):
            self.cal_thread_open()
        if not self.calc_thread.isRunning():
            self.calc_thread.start()
        self.update_status('计算进行中...', 'working')
        self.time_step = float(self.time_step_input.value())
        center = (self.region_y_input.value(), self.region_x_input.value())
        shape = 'square' if self.region_shape_combo.currentText() == "正方形" else 'circle'
        size = self.region_size_input.value()
        model_type = 'single' if self.model_combo.currentText() == "单指数衰减" else 'double'
        mask = PublicEasyMethod.quick_mask(aim_data.framesize, center = center, shape = shape,size = size)
        self.image_display.display_canvas[self.focus_canvas].add_fast_selection(*center,mask)
        self.start_reg_cal_signal.emit(aim_data,self.time_step,mask,model_type)
        return None

    def distribution_analyze_start(self):
        """分析载流子寿命"""
        aim_data = self.data_selection('data')
        if aim_data is None:
            return False
        # 如果线程没了，要创建
        if not self.is_thread_active("calc_thread"):
            self.cal_thread_open()
        if not self.calc_thread.isRunning():
            self.calc_thread.start()
        self.update_status('长时计算进行中...', 'working')
        self.time_step = float(self.time_step_input.value())
        model_type = 'single' if self.model_combo.currentText() == "单指数衰减" else 'double'
        pre_cov = self.pre_cov_combo.currentText() if self.pre_cov_combo.currentIndex() != 0 else None
        post_cov = self.post_cov_combo.currentText() if self.post_cov_combo.currentIndex() != 0 else None
        pre_size = self.pre_cov_size.value() if self.pre_cov_combo.currentIndex() != 0 else None
        post_size = self.post_cov_size.value() if self.post_cov_combo.currentIndex() != 0 else None
        self.start_dis_cal_signal.emit(aim_data,self.time_step,model_type,pre_cov,pre_size,post_cov,post_size,
                                       self.multiprocess_check.isChecked(),self.cpu_use_input.value()-1)
        return None

    def heat_transfer_start(self):
        """分析载流子寿命"""
        # 如果线程没了，要创建
        aim_data = self.data_selection('data')
        if aim_data is None:
            return False
        if not self.is_thread_active("calc_thread"):
            self.cal_thread_open()
        if not self.calc_thread.isRunning():
            self.calc_thread.start()
        self.update_status('传热系数计算进行中...', 'working')
        self.time_step = float(self.time_step_input.value())
        model_type = 'single' if self.model_combo.currentText() == "单指数衰减" else 'double'
        pre_cov = self.pre_cov_combo.currentText() if self.pre_cov_combo.currentIndex() != 0 else None
        post_cov = self.post_cov_combo.currentText() if self.post_cov_combo.currentIndex() != 0 else None
        pre_size = self.pre_cov_size.value() if self.pre_cov_combo.currentIndex() != 0 else None
        post_size = self.post_cov_size.value() if self.post_cov_combo.currentIndex() != 0 else None
        self.start_heat_cal_signal.emit(aim_data,self.time_step,model_type,pre_cov,pre_size,post_cov,post_size,
                                       self.multiprocess_check.isChecked(),self.cpu_use_input.value())
        return True

    def diffusion_calculation_start(self):
        """扩散系数计算"""
        data = self.data_selection('data')
        if data is None:
            return logging.warning('无数据载入')
        elif self.vectorROI_data is None:
            return  logging.warning("无有效ROI数据")
        # self.time_slider_vertical.setVisible(False)
        # 如果线程没了，要创建
        if not self.is_thread_active("calc_thread"):
            self.cal_thread_open()
        if not self.is_thread_active("calc_thread"):
            self.cal_thread_open()

        self.calc_thread.start()
        self.update_status('计算进行中...', 'working')
        self.time_step = float(self.time_step_input.value())
        self.space_step = float(self.space_step_input.value())
        self.start_dif_cal_signal.emit(self.vectorROI_data,self.time_step, self.space_step, data.timestamp,data.name)
        return None

    def pre_process_EM(self):
        """EM的数据预处理"""
        data = self.data_selection('data')
        if data is None:
            return False
        if not self.is_thread_active("avi_thread"):
            # self.EM_thread_open()
            pass
        # 如果有线程在运算，要提示（不过目前不需要，保留语句）
        if not self.avi_thread.isRunning():
            self.avi_thread.start()
        self.pre_process_signal.emit(data, self.bg_nums_input.value(), True)
        return True

    def quality_EM_stft(self):
        """stft质量评价"""
        data = self.data_selection(['EM_pre_processed'])
        if data is not None:
            # 窗函数选择转义
            window_dict = ['hann', 'hamming', 'gaussian', 'boxcar','blackman','blackmanharris']
            self.update_param('EM','stft_window_type',window_dict[self.stft_window_select.currentIndex()])
            dialog = STFTComputePop(self.EM_params,'quality')
            self.update_status("STFT计算ing", 'working')
            if dialog.exec_():
                self.update_param('EM','target_freq',dialog.target_freq_input.value())
                self.update_param('EM', 'EM_fps', dialog.fps_input.value())
                self.update_param('EM', 'stft_scale_range',dialog.scale_range_input.value())
                self.update_param('EM', 'stft_window_size',dialog.window_size_input.value())
                self.update_param('EM', 'stft_noverlap', dialog.noverlap_input.value())
                self.update_param('EM', 'custom_nfft',dialog.custom_nfft_input.value())
                if not self.avi_thread.isRunning():
                    self.avi_thread.start()
                self.stft_quality_signal.emit(data,
                                              self.EM_params['target_freq'],self.EM_params['stft_scale_range'],self.EM_params['EM_fps'],
                                             self.EM_params['stft_window_size'],
                                             self.EM_params['stft_noverlap'],
                                             self.EM_params['custom_nfft'],
                                             self.EM_params['stft_window_type'])
                logging.info("请稍等，出图会有点慢")
                # self.stft_quality_btn.setEnabled(False)
        else:
            logging.warning("查找不到预处理数据，请先对数据进行预处理")
            self.update_status("准备就绪")
            return

    def process_EM_stft(self):
        """EM的数据处理"""
        data = self.data_selection(['EM_pre_processed'])
        if data is None:
            return False
        self.update_status("STFT计算ing", 'working')
        # target_freq = self.parse_frame_input('freq', data.out_processed['frequencies'])
        # 窗函数选择转义
        window_dict = ['hann', 'hamming', 'gaussian', 'boxcar','blackman','blackmanharris']
        self.update_param('EM','stft_window_type',window_dict[self.stft_window_select.currentIndex()])
        dialog = STFTComputePop(self.EM_params, 'process',time_length=data.timelength)
        if dialog.exec_():
            self.update_param('EM','target_freq',dialog.target_freq_input.value())
            self.update_param('EM', 'EM_fps', dialog.fps_input.value())
            self.update_param('EM', 'stft_scale_range',dialog.scale_range_input.value())
            self.update_param('EM', 'stft_window_size',dialog.window_size_input.value())
            self.update_param('EM', 'stft_noverlap', dialog.noverlap_input.value())
            self.update_param('EM', 'custom_nfft',dialog.custom_nfft_input.value())
            if not self.avi_thread.isRunning():
                self.avi_thread.start()
            target_freq = self.EM_params['target_freq'] #if target_freq is None else target_freq
            logging.info(f"目标频率：{target_freq}")
            self.stft_python_signal.emit(data,
                                         target_freq, self.EM_params['stft_scale_range'], self.EM_params['EM_fps'],
                                         self.EM_params['stft_window_size'],
                                         self.EM_params['stft_noverlap'],
                                         self.EM_params['custom_nfft'],
                                         self.EM_params['stft_window_type'],
                                         dialog.multiprocess_check.isChecked(), # 目前加速计算的参数不保存上传，需要每次都确认
                                         dialog.batch_size_input.value(),
                                         dialog.cpu_use_input.value(),)
            self.stft_process_btn.setEnabled(False)
            return True
        return False

    def quality_EM_cwt(self):
        """小波变换的质量评价"""
        data = self.data_selection(['EM_pre_processed'])
        if data is None:
            return False
        dialog = CWTComputePop(self.EM_params,'quality')

        if dialog.exec_():
            self.update_param('EM', 'target_freq', dialog.target_freq_input.value())
            self.update_param('EM', 'EM_fps', dialog.fps_input.value())
            self.update_param('EM', 'cwt_total_scales',dialog.cwt_size_input.value())
            self.update_param('EM', 'cwt_scale_range',dialog.cwt_scale_range.value())
            self.update_param('EM', 'cwt_type', dialog.wavelet.currentText())
            if not self.avi_thread.isRunning():
                self.avi_thread.start()
            self.cwt_quality_signal.emit(data,
                                         self.EM_params['target_freq'],
                                         int(self.EM_params['cwt_scale_range']),
                                         self.EM_params['EM_fps'],
                                         self.EM_params['cwt_total_scales'],
                                         self.EM_params['cwt_type'])
            # self.cwt_quality_btn.setEnabled(False)
            return True
        else:
            return False

    def process_EM_cwt(self):
        """小波变换"""
        data = self.data_selection(['EM_pre_processed'])
        if data is None:
            return False
        dialog = CWTComputePop(self.EM_params, 'signal')
        if dialog.exec_():
            self.update_status("CWT计算ing", 'working')
            self.update_param('EM', 'target_freq', dialog.target_freq_input.value())
            self.update_param('EM', 'EM_fps', dialog.fps_input.value())
            self.update_param('EM', 'cwt_total_scales',dialog.cwt_size_input.value())
            self.update_param('EM', 'cwt_scale_range',dialog.cwt_scale_range.value())
            self.update_param('EM', 'cwt_type', dialog.wavelet.currentText())
            if not self.avi_thread.isRunning():
                self.avi_thread.start()
            self.cwt_python_signal.emit(data,
                                        self.EM_params['target_freq'],
                                        self.EM_params['EM_fps'],
                                        self.EM_params['cwt_total_scales'],
                                        self.EM_params['cwt_type'],
                                        self.EM_params['cwt_scale_range'])
            self.cwt_process_btn.setEnabled(False)
            return True
        else:
            return False

    def roi_signal_avg(self):
        """计算选区信号平均值并显示"""
        data = self.data_selection(['ROI_stft', 'ROI_cwt'])
        if data is not None:
            pass
        else:
            logging.warning("无变换后数据，请先处理数据")
            return
        mask = self.roi_selection(True)
        if mask is None:
            return
        if mask.shape != data.framesize:
            QMessageBox.warning(self, "蒙版错误", "蒙版尺寸与数据不匹配")
            return
        if not self.calc_thread.isRunning():
            self.calc_thread.start()
        self.update_status('计算进行中...', 'working')
        self.easy_process.emit(data,'avg',mask)


    def process_signal_avg(self):
        """广义信号平均"""
        aim_data = self.data_selection()
        if aim_data is None:
            return False
        if not self.avi_thread.isRunning():
            self.avi_thread.start()
        if not self.calc_thread.isRunning():
            self.calc_thread.start()
        self.update_status('计算进行中...', 'working')
        self.easy_process.emit(aim_data,'avg',None)
        return None

    def process_atam(self):
        """累计时间振幅图"""
        aim_data = self.data_selection()
        if aim_data is None:
            return False
        if aim_data.ndim not in [2, 3]:
            logging.info("数据无法被处理，请重选数据")
            return False
        self.atam_signal.emit(aim_data)
        self.atam_btn.setEnabled(False)
        return True
        return False
        # if self.processed_data is None:
        #     if self.data.ndim not in [2,3]:
        #         logging.info("数据无法被处理，请重选数据焦点")
        #         return False
        #     self.atam_signal.emit(self.data)
        #     self.atam_btn.setEnabled(False)
        #     logging.info(f"累计时间振幅图，对意料之外的数据{self.data.name}进行处理")
        #     return False
        # elif self.processed_data.type_processed == 'ROI_stft' or self.processed_data.type_processed =='ROI_cwt':
        #     data = self.processed_data
        # else:
        #     data = next(
        #         (data for data in reversed(self.processed_data.history) if
        #          data.type_processed == "ROI_cwt" or data.type_processed == "ROI_stft"),
        #         None)
        # if data is not None:
        #     if self.processed_data.ndim not in [2,3]:
        #         logging.info("数据无法被处理，请重选数据焦点")
        #         return False
        #     self.atam_signal.emit(data)
        #     self.atam_btn.setEnabled(False)
        #     return True
        # else:
        #     if self.processed_data.ndim not in [2,3]:
        #         logging.info("数据无法被处理，请重选数据焦点")
        #         return False
        #     self.atam_signal.emit(self.processed_data)
        #     self.atam_btn.setEnabled(False)
        #     logging.info(f"累计时间振幅图，对意料之外的数据{self.processed_data.name}进行处理")
        #     return False

    def process_tDgf(self):
        """单通道二维高斯拟合以及信号显示"""
        data = self.data_selection(['Roi_applied'])
        if data is not None:
            dialog = SCSComputePop(self.EM_params)
            if dialog.exec_():
                self.update_status("单通道计算ing", 'working')
                self.update_param('EM', 'scs_thr',dialog.thr_input.value())
                self.update_param('EM', 'thr_known', dialog.thr_known_check.isChecked())
                self.update_param('EM', 'scs_zoom',dialog.zoom_input.value())
                if not self.avi_thread.isRunning():
                    self.avi_thread.start()
                self.tDgf_signal.emit(data,
                                      self.EM_params['scs_zoom'],
                                      self.EM_params['scs_thr'],
                                      self.EM_params['thr_known'])
                self.tDgf_btn.setEnabled(False)
        else:
            QMessageBox.warning(self,"数据错误","不支持的数据类型，请确认前序处理是否正确（是否确认ROI）")
            self.update_status("准备就绪", 'idle')
            return

    def process_simple_scs(self):
        """单通道处理开始"""
        data = self.data_selection(['Roi_applied'])
        if data is not None:
            dialog = SCSComputePop(self.EM_params)
            if dialog.exec_():
                self.update_status("单通道计算ing", 'working')
                self.update_param('EM', 'scs_thr', dialog.thr_input.value())
                self.update_param('EM', 'thr_known', dialog.thr_known_check.isChecked())
                self.update_param('EM', 'scs_zoom', dialog.zoom_input.value())
                if not self.avi_thread.isRunning():
                    self.avi_thread.start()
                self.sscs_signal.emit(data,
                                      self.EM_params['scs_zoom'],
                                      self.EM_params['scs_thr'],
                                      self.EM_params['thr_known'])
                self.sscs_btn.setEnabled(False)
        else:
            QMessageBox.warning(self,"数据错误","不支持的数据类型，请确认前序处理是否正确（是否确认ROI）")
            self.update_status("准备就绪", 'idle')
            return

    def process_tDFT(self):
        """二维傅里叶变换"""
        aim_data = self.data_selection()
        if aim_data is None:
            return False
        if not self.avi_thread.isRunning():
            self.avi_thread.start()
        self.update_status('二维傅里叶变换计算进行中...', 'working')
        self.tDFT_signal.emit(aim_data)
        return True

    def process_heartbeat(self):
        """心肌细胞跳动分析"""
        aim_data = self.data_selection()
        if aim_data is None:
            return False
        self.dialog = HeartBeatFrameSelectDialog(aim_data, self)
        self.dialog.show()
        self.dialog.raise_()
        self.update_status('心肌细胞跳动分析中...', 'working')
        return True

    def data_selection(self, aim_type:str | list = 'all'):
        """数据选择代码（模式流程）"""
        aim_data = None
        if self.data is None and self.processed_data is None:
            logging.warning("无数据可处理，请先加载数据")
            return None
        match self.mode:
            case 0:
                aim_data = self.data_pick()
            case 1:
                if aim_type == 'data':
                    aim_data = self.data
                elif aim_type == 'all':
                    aim_data = self.data_pick()
                elif self.processed_data is None:
                    return None
                elif aim_type == 'processed':
                    aim_data = self.processed_data
                elif self.processed_data.type_processed in aim_type:
                    aim_data = self.processed_data
                else:
                    aim_data = next(
                        (data for data in reversed(self.processed_data.history) if
                         data.type_processed in aim_type),
                        None)
            case 2:
                aim_data = self.data_pick()
        return aim_data

    def roi_selection(self, select = False):
        """ROI选择"""
        mask = None
        match self.mode:
            case 2:
                mask = self.image_display.get_draw_roi()[1]
                if mask is None:
                    logging.warning("选中画布没有绘制有效的ROI")
                    return None
            case 0:
                dialog = ROIInfoDialog(self.image_display.get_all_canvas_info(), self)
                if dialog.exec_():
                    aim_id = dialog.canvas_id
                    if dialog.roi_type == 'pixel_roi':
                        mask = self.image_display.get_draw_roi(aim_id)[1]
                    elif dialog.roi_type == 'v_line':
                        QMessageBox.warning(self,"警告","不支持该类型")
                        return None
                    elif dialog.roi_type == 'v_rect':
                        rect_mask = self.image_display.display_canvas[aim_id].v_rect_roi
                        x, y, w, h = rect_mask[0][0], rect_mask[0][1], rect_mask[1], rect_mask[2]
                        if w == 0 or h == 0:
                            return None
                        else:
                            mask = np.zeros(self.image_display.display_canvas[aim_id].data.framesize, dtype=bool)
                            mask[y:y + h, x:x + w] = True
                    elif dialog.roi_type == 'anchor':
                        mask = self.image_display.display_canvas[aim_id].anchor_mask
                    else:
                        return None
            case 1:
                if select:
                    mask = self.image_display.get_draw_roi()[1]
                    if mask is None:
                        logging.warning("选中画布没有绘制有效的ROI")
                        return None
                else:
                    return None
        return mask

    def data_pick(self, need_all=True):
        """数据选择"""
        dialog = DataViewAndSelectPop(datadict=self.get_data_all(),
                                      processed_datadict=self.get_processed_data_all(),
                                      add_canvas=False, parent=self)
        aim_data = None
        if dialog.exec_():
            selected_timestamp, selected_table = dialog.get_selected_timestamp()
            if selected_table == 'data':
                for data in self.data.history:
                    if data.timestamp == selected_timestamp:
                        aim_data = data
                        logging.info(f"数据选择成功（初始导入）：{data.name}")
                        break
            else:
                for data in self.processed_data.history:
                    if data.timestamp == selected_timestamp:
                        aim_data = data
                        logging.info(f"数据选择成功（处理过）：{data.name}")
                        break
            if aim_data is None:
                QMessageBox.warning(self, "数据错误", "没有选取数据")
                return None
            return aim_data
        return None

    """结果处理"""
    def processed_result(self, data):
        """处理过后的数据都来这里重整再分配"""
        if isinstance(data, ProcessedData):
            pass
        else:
            self.cwt_quality_btn.setEnabled(True)
            self.stft_quality_btn.setEnabled(True)
            self.stft_process_btn.setEnabled(True)
            self.cwt_process_btn.setEnabled(True)
            self.tDgf_btn.setEnabled(True)
            self.sscs_btn.setEnabled(True)
            QMessageBox.warning(self,"运算错误",f"在{data['type']}处理中报错：\n{data['error']}")
            logging.error("运算因错误而终止")
            self.update_progress(-1) # 进度条重置
            return False
        self.processed_data = data
        # 各处理后响应
        process_type = self.processed_data.type_processed
        match process_type:
            case "ROI_lifetime":
                self.result_display.display_lifetime_curve(self.processed_data,self.time_unit_combo.currentText())
            case 'lifetime_distribution':
                self.result_display.display_distribution_map(self.processed_data,'指数衰减寿命分布图')
            # 中间还有一个取向量ROI的，先不管他
            case 'diffusion':
                self.result_display.display_diffusion_coefficient(self.processed_data)
                pass
            case 'heat_transfer':
                self.result_display.display_distribution_map(self.processed_data,'传热系数分布图') # 临时之举
                pass
            case 'EM_pre_processed':
                pass
            case 'stft_quality':
                self.stft_quality_btn.setEnabled(True)
                self.result_display.quality_avg(self.processed_data)
            case 'cwt_quality':
                logging.info("请稍等，出图会有点慢")
                self.cwt_quality_btn.setEnabled(True)
                self.result_display.quality_avg(self.processed_data)
            case 'ROI_stft':
                result = self.processed_data.data_processed
                self.stft_process_btn.setEnabled(True)
                # if self.show_stft_check.isChecked():
                #     self.data.image_import = (result - np.min(result)) / (np.max(result) - np.min(result)) # 要改
                #     self.load_image()
                pass
            case 'ROI_cwt':
                result = self.processed_data.data_processed
                self.cwt_process_btn.setEnabled(True)
                # if self.show_stft_check.isChecked():
                #     self.data.image_import = (result - np.min(result)) / (np.max(result) - np.min(result))  # 要改
                #     self.load_image()
                pass
            case 'Accumulated_time_amplitude_map':
                self.atam_btn.setEnabled(True)
                pass
            case 'Single_channel_signal':
                self.tDgf_btn.setEnabled(True)
                self.sscs_btn.setEnabled(True)
                if self.processed_data.out_processed['thr_known']:
                    self.result_display.single_channel(self.processed_data,True)
                else:
                    thr = int(self.processed_data.out_processed['thr'])
                    self.time_slider_vertical.setVisible(True)
                    self.time_slider_vertical.setMaximum(int(self.processed_data.out_processed['mean_signal'].max()*10+21))
                    # self.time_slider_vertical.setValue(thr*10)
                    self.update_result_display(thr*10, reuse_current=False)
            case '2D_Fourier_transform':
                pass
            case 'signal_average':
                self.result_display.plot_time_series(data.time_point, data.data_processed[:,1])
                self.graph_plot.plot_data(data.data_processed, name = data.name)
                pass
            case 'Roi_applied':
                logging.info("ROI应用完成")
            case 'Heartbeat':
                logging.info("心肌细胞处理完成，开始作图")
                self.result_display.display_heartbeat(self.processed_data)
                logging.info("所有图绘制完成")

    def draw_result(self,draw_type:str,canvas_id:int,result,roi_info = None):
        """canvas绘图结果处理"""
        timestamp = self.image_display.display_canvas[canvas_id].data.timestamp_inherited
        draw_data = None
        data_type = None
        bool_mask = None
        if self.data is not None:
            for data in self.data.history:
                if data.timestamp == timestamp:
                    draw_data = data
                    break
        if self.processed_data is not None and draw_data is None:
            draw_data = next(data for data in self.processed_data.history if data.timestamp == timestamp)
            data_type = draw_data.type_processed
        crop_roi = False
        dialog = ROIProcessedDialog(draw_type, canvas_id, result,roi_info,data_type, self)
        if dialog.exec_():
            if dialog.crop_check.isChecked():
                crop_roi = True
            if draw_type == "v_rect":
                x,y,w,h = result[0][0],result[0][1],result[1],result[2]
                if w == 0 or h == 0:
                    logging.warning("未选中像素")
                    return None
                else:
                    bool_mask = np.zeros(draw_data.framesize, dtype=bool)
                    bool_mask[y:y + h, x:x + w] = True
                if crop_roi:
                    self.roi_processed_signal.emit(draw_data, bool_mask, dialog.reset_value.value(), crop_roi, dialog.zoom_check.isChecked(), dialog.zoom_factor.value())
                if dialog.fast_check.isChecked():
                    if hasattr(draw_data,'type_processed') and draw_data.type_processed == 'Accumulated_time_amplitude_map':
                        try:
                            source_data = next(data for data in self.processed_data.history if data.timestamp == draw_data.timestamp_inherited)
                        except: # 如果不行就从data里找
                            source_data = next(data for data in self.data.history if data.timestamp == draw_data.timestamp_inherited)
                        if isinstance(source_data, Data):
                            roi_data = source_data.data_origin[:,y:y+h,x:x+w]
                        elif isinstance(source_data, ProcessedData):
                            roi_data = source_data.data_processed[:,y:y+h,x:x+w]
                        else:
                            logging.error("roi应用错误（不可能错误）")
                            roi_data = None
                        self.processed_data = ProcessedData(draw_data.timestamp,
                                                        f"{draw_data.name}@ROIed",
                                                        "Roi_applied",
                                                        time_point=draw_data.time_point,
                                                        data_processed=roi_data,
                                                        out_processed=draw_data.out_processed,
                                                        ROI_applied=True)
            elif draw_type == "v_line":
                self.vector_array = result.getPixelValues(draw_data, self.space_step, self.time_step)
            elif draw_type == "pixel_roi":
                bool_mask = result[1]
                if dialog.inverse_check.isChecked():
                    bool_mask = ~bool_mask
                self.roi_processed_signal.emit(draw_data, bool_mask, dialog.reset_value.value(),crop_roi, dialog.zoom_check.isChecked(), dialog.zoom_factor.value())
            logging.info("ROI已确认选取")
        return None

    def fast_roi_result(self):
        """roi快速选取，仅支持pixel_roi"""
        canvas_id = self.roi_pick.currentIndex()
        _, bool_mask = self.image_display.get_draw_roi(canvas_id)
        if bool_mask is None: # 不再赋给 self.bool_mask
            logging.warning("没有有效蒙版")
        timestamp = self.image_display.display_canvas[canvas_id].data.timestamp_inherited
        draw_data = None
        if self.data is not None:
            for data in self.data.history:
                if data.timestamp == timestamp:
                    draw_data = data
                    break
        if self.processed_data is not None and draw_data is None:
            draw_data = next(data for data in self.processed_data.history if data.timestamp == timestamp)
            # data_type = draw_data.type_processed
        self.roi_processed_signal.emit(draw_data, bool_mask, 1, True,
                                       False, 0)
        logging.info(f"像素roi已快速选取，数据名{draw_data.name}")

    def data_plot_add(self):
        """选取数据送入结果显示（graphplot驱动）"""
        self.data_plot_selector = DataTreeViewDialog(self)
        # 连接信号：当数据管理器中的“导出绘图”被点击时
        self.data_plot_selector.sig_plot_request.connect(self.proc_thread.plot_data_prepare)
        self.data_plot_selector.sig_canvas_signal.connect(self.upgrade_and_imaging)
        self.data_plot_selector.refresh_data()
        self.data_plot_selector.show()

    def data_plot_clear(self):
        """plot清空"""
        self.graph_plot.clear_all()

    def upgrade_and_imaging(self, data:ProcessedData, key:str):
        """从树结构选择器中来，出新成像"""
        processed_data = data.upgrade_processed(key)
        if processed_data is not None:
            self.processed_data = processed_data
            self.add_new_canvas(self.processed_data)
            return None
        else:
            return logging.error("导入成像失败")

    '''其他功能'''
    def is_thread_active(self, thread_name: str) -> bool:
        """检查指定名称的线程是否存在且正在运行"""
        # :param thread_name: 线程对象的变量名（str）
        # :return: True（线程存在且运行中）/ False（线程不存在或已结束）

        if hasattr(self, thread_name):
            thread = getattr(self, thread_name)  # 动态获取线程对象
            if isinstance(thread, QThread) and not sip.isdeleted(thread):
                return thread.isRunning()
        return False

    def btn_safety(self, cal_run=False):
        """关闭按钮的功能"""
        if cal_run:
            self.analyze_btn.setEnabled(False)
            self.analyze_region_btn.setEnabled(False)
            self.heat_transfer_btn.setEnabled(False)
        elif not cal_run:
            self.analyze_btn.setEnabled(True)
            self.analyze_region_btn.setEnabled(True)
            self.heat_transfer_btn.setEnabled(True)
        return

    def stop_thread(self,type = 0):
        """彻底删除线程（反正关闭也不能重启）后续线程多了加入选择关闭的能力"""
        if type == 0 and self.is_thread_active("calc_thread"):
            try:
                self.calc_thread.quit()  # 请求退出
                self.calc_thread.wait()  # 等待结束
                self.calc_thread.deleteLater()  # 标记删除
                logging.info("计算线程关闭")
            except Exception as e:
                logging.error(f"线程退出错误{e}")
        if type == 1 and hasattr(self,"avi_thread") and self.is_thread_active("avi_thread"):
            try:
                self.avi_thread.quit()
                self.avi_thread.wait()
                self.avi_thread.deleteLater()
                logging.info("大数据处理线程关闭")
            except Exception as e:
                logging.error(f"线程退出错误{e}")

    def export_image(self):
        """导出热图为图片"""
        current_index = self.result_display.currentIndex()
        if current_index < 0:
            QMessageBox.warning(self, "导出失败", "没有可导出的图像")
            return

        tab = self.result_display.widget(current_index)
        canvas = tab.findChild(FigureCanvas)

        if not canvas:
            QMessageBox.warning(self, "导出失败", "未找到图像画布")
            return

        try:
            path, _ = QFileDialog.getSaveFileName(
                self, "保存图像", "", "PNG(*.png);;JPEG(*.jpg);;TIFF图像 (*.tif *.tiff);;所有文件(*.*)"
            )

            if path:
                canvas.figure.savefig(path, dpi=300)
                QMessageBox.information(self, "导出成功", f"图像已保存至:\n{path}")
                logging.info(f"导出成功,图像已保存至:{path}")
        except:
            logging.info("数据未保存")

        # if hasattr(self.result_display, 'current_data'):
        #     file_path, _ = QFileDialog.getSaveFileName(
        #         self, "保存图像", "", "PNG图像 (*.png);;JPEG图像 (*.jpg *.jpeg);;TIFF图像 (*.tif *.tiff)")
        #
        #     if file_path:
        #         # 从matplotlib保存图像
        #         self.result_display.figure.savefig(file_path, dpi=300, bbox_inches='tight')
        #     logging.info("图片已保存")

    def export_data(self):
        """导出寿命数据"""
        if self.result_display.current_dataframe is not None:
            dialog = DataSavingPop(self)
            file_path = None
            self.update_status("数据导出ing", 'working')
            if dialog.exec_():
                isfiting = dialog.fitting_check.isChecked()
                hasheader = dialog.index_check.isChecked()
                extra_check = dialog.extra_check.isChecked()
                file_path, _ = QFileDialog.getSaveFileName(
                    self, "保存数据", "", "CSV文件 (*.csv);;文本文件 (*.txt)")
        else:
            logging.warning("没有数据可以导出")
            self.update_status("准备就绪")
            return

        if file_path:
            df = self.result_display.current_dataframe
            if not isfiting:
                if self.result_display.current_mode == 'curve':
                    df = df.loc[:, df.columns != 'fit_curve']
                elif self.result_display.current_mode == 'diff':
                    df = df.loc[:, df.columns.get_level_values(1) != '拟合曲线']
                elif self.result_display.current_mode == 'heatmap':
                    pass
                elif self.result_display.current_mode == 'roi':
                    pass
                elif self.result_display.current_mode == 'var':
                    pass
                elif self.result_display.current_mode == 'series':
                    pass
            # 保存为CSV或TXT
            if file_path.lower().endswith('.csv'):
                try:
                    df.to_csv(file_path, index=False, header=hasheader)
                    logging.info("数据已保存")
                except:
                    logging.info("数据未保存")
            else:
                try:
                    df.to_csv(file_path, sep='\t', index=False, header=hasheader)
                    logging.info("数据已保存")
                except:
                    logging.info("数据未保存")
            self.update_status("准备就绪", 'idle')
        else:
            logging.info("数据未保存")
            self.update_status("准备就绪", 'idle')
            return

    def export_EM_data(self,result):
        """时频变换后目标频率下的结果导出"""
        if self.processed_data is not None:
            if self.processed_data.type_processed == 'ROI_stft' or 'ROI_cwt':
                dialog = DataExportDialog(datatypes=['tif','avi','gif','png'])
                if dialog.exec_():
                    directory = dialog.directory
                    prefix = dialog.text_edit.text().strip()
                    filetype = dialog.type_combo.currentText()
                    duration = dialog.duration_input.value()
                    self.mass_export_signal.emit(self.processed_data.data_processed,directory,prefix,filetype,True,
                                                 {'duration':duration})
                return
            else:
                QMessageBox.warning(self,'提示','请先变换处理数据')
        else:
            logging.warning('请先加载并处理数据')
            return

    def data_history_view(self):
        """查看历史数据"""
        if self.data is None :
            logging.warning('请先导入数据')
            return

        dialog = DataViewAndSelectPop(datadict=self.get_data_all())
        if dialog.exec_():
            selected_timestamp,_ = dialog.get_selected_timestamp()
            self.data = self.data.find_history(selected_timestamp)
            logging.info(f"当前数据焦点已更新至{self.data.name}")

    def process_history_view(self):
        """查看历史数据-处理"""
        if self.processed_data is None:
            logging.warning('请先处理数据')
            return
        # if self.image_display is []:  走不到这里
        #     logging.warning("请先导入数据")

        dialog = DataViewAndSelectPop(processed_datadict=self.get_processed_data_all())
        if dialog.exec_():
            selected_timestamp,_ = dialog.get_selected_timestamp()
            self.processed_data = self.processed_data.find_history(selected_timestamp)
            logging.info(f"当前数据焦点已更新至{self.processed_data.name}")

    def data_history_clear(self):
        """历史数据清除（所有）"""
        if Data is not None:
            Data.clear_history()
            logging.info('导入数据已清除')
        else:
            logging.warning('没有数据可清除')
        if ProcessedData is not None:
            ProcessedData.clear_history()
            logging.info('处理数据已清除')

    '''以下控制台命令更新'''
    def stop_calculation(self):
        """终止当前计算"""
        # 这里需要实现终止计算的逻辑
        # 可以通过设置标志位或直接终止计算线程
        logging.warning("计算终止请求已接收，正在停止...")
        if hasattr(self, 'cal_thread'):
            # self.cal_thread.stop()
            self.stop_thread(0)
        if hasattr(self, 'avi_thread'):
            # self.avi_thread.stop()
            self.stop_thread(1)
        return
        # 实际终止逻辑需要根据你的计算实现来添加

    def save_config(self):
        """保存当前配置(留空暂不实现)"""
        logging.info("正在保存当前配置...")

    def load_config(self, preset_name):
        """加载预设参数(留空暂不实现)"""
        logging.info(f"正在加载预设参数: {preset_name}")

    def clear_result(self):
        self.result_display.clear()

    @staticmethod
    def format_time(seconds):
        """将秒数格式化为 HH:MM:SS"""
        if seconds < 0:
            return "--:--:--"
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"


class StreamLogger(object):
    """重定向标准输出到日志系统"""

    def __init__(self, log_level):
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            logging.log(self.log_level, line.rstrip())

    def flush(self):
        pass


if __name__ == "__main__":
    multiprocessing.freeze_support()
    app = QApplication([])
    QFontDatabase.addApplicationFont("C:/Windows/Fonts/NotoSansSC-VF.ttf")  # 如：思源黑体、阿里巴巴普惠体
    QFontDatabase.addApplicationFont("C:/Windows/Fonts/calibril.ttf")  # 如：Roboto、Fira Code

    def read_qss_file(qss_file_name):
        if hasattr(sys, '_MEIPASS'):
            # 如果是，基础路径是临时解压目录
            base_path = sys._MEIPASS
        else:
            # 如果不是（开发环境），基础路径是当前脚本所在目录
            base_path = os.path.dirname(os.path.abspath(__file__))

            # 拼接出QSS文件的完整绝对路径
        qss_path = os.path.join(base_path, qss_file_name)
        with open(qss_path, 'r', encoding='UTF-8') as file:
            return file.read()
    # 应用全局样式
    app.setStyle('Fusion')
    app.setStyleSheet(read_qss_file("style.qss"))
    QCoreApplication.setOrganizationName("CSSA")
    QCoreApplication.setApplicationName("LifeCalor")
    os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    # app.setFont(QFont("Noto Sans"))
    app.setWindowIcon(QIcon(':/LifeCalor.ico'))
    window = MainWindow()
    window.setWindowIcon(QIcon(':/LifeCalor.ico'))
    window.show()
    app.exec_()