import multiprocessing
from pathlib import Path

import resources_rc # 重要不能删
from logging.handlers import RotatingFileHandler
from PyQt5 import sip
from PyQt5.QtGui import QFontDatabase, QDesktopServices
from PyQt5.QtWidgets import (QStackedWidget, QStatusBar, QFrame, QSplitter, QDesktopWidget, QSizePolicy
                             )
from PyQt5.QtCore import QElapsedTimer, QSettings, QCoreApplication, QUrl, QStandardPaths

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
from SpatialExtractor import SpatialExtractor
from calculator.dialog import DataCalculatorDialog
from widget import TriStateSwitch
from AppConfig import get_github_auth_header
from settings import load_param_group
from TaskState import TaskState
from ThreadController import is_thread_active as thread_is_active, stop_thread as stop_qthread
from exporting import ExportController
from selection import SelectionController
from history import HistoryController
from history.manifest import HistoryManifestStore, array_refs_for_manifest
from TaskController import ensure_thread_running
from progress import normalize_progress
from display.status import render_status_update
from display.canvas_signals import CanvasSignalBinder, disconnect_canvas_signal as disconnect_display_canvas_signal
from display.canvas_controller import DisplayCanvasController
from display.hover import format_hover_value
from diagnostics import AppError, report_warning, show_app_error
from tasks import TaskCoordinator, TaskStatus


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
    managed_export_signal = pyqtSignal(object, str, str, str, bool, dict)
    atam_signal = pyqtSignal(object)
    tDgf_signal = pyqtSignal(object,int,float,bool)
    sscs_signal = pyqtSignal(object, int, float, bool)
    tDFT_signal = pyqtSignal(object)
    tDiFT_signal = pyqtSignal(object)
    heartbeat_signal = pyqtSignal(object, int, int ,list, str, str, float)
    calculator_signal = pyqtSignal(object)
    easy_process = pyqtSignal(object, str, object)
    roi_value_distribution_signal = pyqtSignal(object, np.ndarray, int, object, object, str)
    roi_processed_signal = pyqtSignal(object,np.ndarray,float,bool,bool,float)
    cache_progress_signal = pyqtSignal(object, object, str)

    def __init__(self):
        super().__init__()
        # 基本信息初始化
        self.current_version = "1.0.9"  # 当前程序版本
        self.repo_owner = "CSSAcslin"  # 程序作者
        self.repo_name = "Carrier-Lifetime-Calculator"  # 程序仓库名
        self.PAT = get_github_auth_header()

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
        self.cache_progress_signal.connect(self.cache_progress_update)
        self.init_params()

        # 界面加载
        self.init_ui()
        self.task_coordinator = TaskCoordinator(self)
        self.task_coordinator.task_updated.connect(self._on_task_updated)
        self.task_coordinator.task_finished.connect(self._on_task_finished)
        self._legacy_task_ids = {}
        self._displayed_task_id = None
        self.selection_controller = SelectionController(self)
        self.export_controller = ExportController(self)
        self.history_controller = HistoryController(self)
        if self._cleanup_cache_on_startup:
            QTimer.singleShot(0, self.history_controller.cleanup_orphans)
        self.canvas_signal_binder = CanvasSignalBinder(self)
        self.display_canvas_controller = DisplayCanvasController(self)
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
        self.task_states = {
            "import": TaskState("import"),
            "calculation": TaskState("calculation"),
            "em_processing": TaskState("em_processing"),
            "export": TaskState("export"),
        }
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
            'peak_min': 0,
            'peak_max': 50,
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
            'cache_directory': self.default_cache_directory(),
            'cache_threshold_mb': 512,
            'cache_cleanup_startup': True,
        })
        self.apply_cache_settings()
        self._cleanup_cache_on_startup = bool(self.tool_params.get('cache_cleanup_startup', True))

        self.save_params()
        self.save_timer = QTimer()
        self.save_timer.timeout.connect(self.save_params)
        self.save_timer.start(30000)  # 每10秒自动保存一次

    def default_cache_directory(self):
        """返回默认缓存目录。"""
        base_path = QStandardPaths.writableLocation(QStandardPaths.AppLocalDataLocation)
        if not base_path:
            base_path = os.path.join(os.getcwd(), ".lifecalor_cache")
        return os.path.join(base_path, "cache")

    def apply_cache_settings(self):
        """应用缓存目录和大数组写入阈值。"""
        cache_directory = self.tool_params.get('cache_directory') or self.default_cache_directory()
        cache_threshold_mb = int(self.tool_params.get('cache_threshold_mb', 512))
        configure_array_cache(ArrayCacheConfig(
            cache_dir=Path(cache_directory),
            threshold_bytes=cache_threshold_mb * 1024 * 1024,
        ))
        set_array_cache_progress_callback(self.cache_progress_signal.emit)

    def cache_progress_update(self, current, total, message):
        """在主线程更新缓存读写进度。"""
        self.update_status(message, 'working')
        self.update_progress(current, total)

    def _load_param_group(self, group_name, defaults):
        """加载参数组，如果没有则使用默认值"""
        self.settings.beginGroup(group_name)
        try:
            return load_param_group(self.settings.value, defaults)
        finally:
            self.settings.endGroup()

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
        screen = QDesktopWidget().screenGeometry()
        screen_width = screen.width()
        screen_height = screen.height()

        # 计算窗口大小（例如，设为屏幕的80%）
        window_width = int(screen_width * 0.86)
        window_height = int(screen_height * 0.86)
        self.setGeometry(int(screen_width * 0.07), int(screen_height * 0.07), window_width, window_height)

        # 左侧设置区域
        self.setup_left_panel()
        self.param_dock = QDockWidget("基础设置", self)
        self.param_dock.setWidget(self.left_panel)
        self.param_dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        # self.param_dock.setMinimumSize(300, 700)
        self.param_dock.setMaximumWidth(350)
        # self.addDockWidget(Qt.LeftDockWidgetArea, self.param_dock) # 加到左侧

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
        # self.image_dock.setMinimumSize(700, 600)
        self.image_dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        # self.addDockWidget(Qt.RightDockWidgetArea, self.image_dock)

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
        # 添加分析按钮和导出按钮
        data_save_layout = QHBoxLayout()
        save_help = InfoButton("此处需要先有绘图结果才能导出。<br>该处绘图结果为正常流程中的出图，暂无法自由绘制。<br>若需检验数据并绘制请使用下面的<b>数据结果</b>窗口，不能绘制的请请重新计算。")
        data_save_layout.addWidget(save_help)
        data_save_layout.addStretch()
        self.export_image_btn = QPushButton("导出结果为图片")
        self.export_data_btn = QPushButton("导出结果为数据")
        self.export_data_btn.setObjectName("StressButton")
        self.export_image_btn.setObjectName("StressButton")
        data_save_layout.addWidget(self.export_image_btn)
        data_save_layout.addWidget(self.export_data_btn)
        result_layout.addLayout(data_save_layout)
        self.result_dock.setWidget(result_widget)
        self.result_dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        # self.result_dock.setMinimumSize(350, 300)

        self.plot_dock = QDockWidget("数据结果", self)
        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)
        inner_layout = QHBoxLayout()
        self.add_data_btn = QPushButton("添加/导出数据")
        self.reset_data_btn = QPushButton("清除数据")
        self.add_data_btn.setObjectName("StressButton")
        self.reset_data_btn.setObjectName("StressButton")
        inner_layout.addWidget(self.add_data_btn)
        inner_layout.addWidget(self.reset_data_btn)
        inner_layout.addStretch()
        inner_layout.addWidget(InfoButton("""<div style='width: 250px;'>本区域也会显示一些一维数据结果，但具有更灵活的功能
                                                <br>可以即时显示，随意放大拖动，还能<b>添加数据</b>进行比较<div>"""))
        plot_layout.addLayout(inner_layout)
        self.graph_plot = PlotGraphWidget()
        plot_layout.addWidget(self.graph_plot)
        self.plot_dock.setWidget(plot_widget)
        self.plot_dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        # self.plot_dock.setMinimumSize(350, 300)
        # self.plot_dock.setLayout(plot_layout)
        self.setup_status_bar()

        # 设置控制台
        self.setup_console()

        # splitter架构的分区器
        main_splitter = QSplitter(Qt.Horizontal, self)
        right_splitter = QSplitter(Qt.Horizontal, self)
        result_splitter = QSplitter(Qt.Vertical, self)
        plot_splitter = QSplitter(Qt.Vertical, self)

        main_splitter.addWidget(self.param_dock)
        main_splitter.addWidget(right_splitter)
        main_splitter.setSizes([1000, 4000])

        right_splitter.addWidget(self.image_dock)
        right_splitter.addWidget(result_splitter)
        right_splitter.setSizes([3000, 2000])

        plot_splitter.addWidget(self.result_dock)
        plot_splitter.addWidget(self.plot_dock)
        plot_splitter.setSizes([1000, 1000])

        result_splitter.addWidget(plot_splitter)
        result_splitter.addWidget(self.console_dock)
        result_splitter.setSizes([2000, 300])

        self.setCentralWidget(main_splitter)

    def setup_left_panel(self):
        """设置左侧面板"""
        self.left_panel = QScrollArea()
        self.left_panel_widget = QWidget()
        self.left_panel_widget.setStyleSheet(""" QWidget {background-color: white; }""")
        self.left_panel.setWidgetResizable(True)
        self.left_panel_layout = QVBoxLayout()
        self.left_panel_layout.setContentsMargins(15,15,15,15)
        self.setup_data_panel()
        self.setup_parameter_panel()
        self.setup_modes_panel()
        self.left_panel_layout.addWidget(self.data_import)
        self.left_panel_layout.addWidget(self.parameter_panel)
        self.left_panel_layout.addWidget(self.modes_panel, stretch=1)
        self.left_panel_layout.addSpacing(15)
        self.left_panel_widget.setLayout(self.left_panel_layout)
        self.left_panel.setWidget(self.left_panel_widget)

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
        fps_layout.addStretch(1)
        fps_info = InfoButton("会根据<b>选择模式</b>自动调整可设置参数。\n帧率和时间单位不会同时设置", topic_key=None)
        fps_layout.addWidget(fps_info)
        param_layout.addLayout(fps_layout)

        # 模式选择
        self.fuction_select = QComboBox()
        self.fuction_select.addItems(['请选择分析模式','超快成像动态分析','EM-iSCAT','通用文件导入','其他方法'])
        left_layout0.addWidget(self.fuction_select)
        left_layout0.addWidget(param_panel)

        self.funtion_stack = QStackedWidget()
        nothing_group = self.QGroupBoxCreator(style="inner")
        nothing_layout = QHBoxLayout()
        nothing_layout.addWidget(QLabel("首先：请选择分析模式!"))
        nothing_layout.addStretch(1)
        nothing_layout.addWidget(InfoButton("选择分析模式后，此处会显示对应的数据导入选项。"
                                            "<br><b>点击按钮</b>显示<span style='font-weight: bold; color: #2E7D32;'>全体帮助指南</span>",
                                            ['general']))
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
        type_choose1.addWidget(InfoButton("选择不区分可以导入无后缀的任意tif文件"))
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

        # 通用文件导入
        general_import_group = self.QGroupBoxCreator(style="inner")
        general_import_layout = QVBoxLayout()
        self.general_format_selector = QComboBox()
        self.general_format_selector.addItems(["自动识别", "NumPy NPY", "TIFF 图像/堆栈"])
        self.general_color_policy = QComboBox()
        self.general_color_policy.addItems(["保留 TIFF 颜色显示", "转换为灰度显示"])
        self.general_time_basis = QComboBox()
        self.general_time_basis.addItems(["使用时间间隔", "使用 FPS"])
        self.general_file_btn = QPushButton("选择数据文件")
        general_import_layout.addWidget(self.general_format_selector)
        general_import_layout.addWidget(self.general_color_policy)
        general_import_layout.addWidget(self.general_time_basis)
        general_import_layout.addWidget(self.general_file_btn)
        general_import_group.setLayout(general_import_layout)
        self.funtion_stack.addWidget(general_import_group)

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
        self.parameter_panel = self.QGroupBoxCreator("处理模式")
        process_layout = QVBoxLayout()
        switch_layout = QHBoxLayout()
        self.tri_switch = TriStateSwitch.TriStateSwitch()
        self.tri_switch.setValue(1)
        self.mode_label = QLabel("默认模式")
        # switch_layout.addWidget(QLabel('处理模式：'))
        self.tri_switch.setFixedWidth(100)
        switch_layout.addWidget(self.tri_switch)
        self.tri_switch.valueChanged.connect(self.mode_switch_change)
        self.mode_label.setStyleSheet(f"font-size: 18px; font-weight: bold; color: #999999")
        switch_layout.addWidget(self.mode_label)
        self.mode_info = InfoButton("""<div style='white-space: pre;'>默认模式:<br>按标准处理流程执行的快速操作，<br>数据源和ROI会按照默认流程<b>自动</b>选择</div>""")
        switch_layout.addWidget(self.mode_info)
        process_layout.addLayout(switch_layout)
        process_layout.addSpacing(5)
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
        """fs_iSCAT下的功能选择"""
        fs_iSCAT_GROUP = self.QGroupBoxCreator(style="noborder")
        operation_layout = QVBoxLayout()
        # 寿命模型选择
        lifetime_layout = QHBoxLayout()
        lifetime_layout.addWidget(QLabel("寿命模型："))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["单指数衰减", "双指数-仅区域"])
        lifetime_layout.addWidget(self.model_combo)
        lifetime_layout.addWidget(InfoButton("<div style='white-space: pre;'><b>寿命模型</b>：双指数有问题不要用<br><b>卷积处理</b>：建议首选smooth或gaussian</div>"))
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
        multipro_layout.addWidget(
            InfoButton("建议关闭闲职程序，保证最佳内存,\n启用后偶尔卡顿属正常现象，进度条不显示实时进度。"))
        heatmap_layout.addLayout(multipro_layout)
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
        lifetime_layout.addWidget(InfoButton("双指数没做好，不要用"))
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
        size_layout.addWidget(InfoButton("在图像上直接点击即可选取，并在图像上会有显示"))
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
        """EM_iSCAT下的功能选择"""
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
        # stft_layout.addWidget(self.retransform_input)
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
        output_btn_layout = QHBoxLayout()
        self.EM_output_btn = QPushButton("时频变换结果快捷导出")
        output_btn_layout.addWidget(self.EM_output_btn)
        output_btn_layout.addWidget(InfoButton("""<div style='white-space: pre;'>这里是用于快速导出刚刚变换后的结果，导出还有其他方法：<br>1. 在图像显示区域显示后从右上角工具栏选择导出</div>"""))
        EM_iSCAT_layout1.addLayout(output_btn_layout)

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
        roi_signal_btn_layout = QHBoxLayout()
        roi_signal_btn_layout.addWidget(self.roi_signal_btn)
        roi_signal_btn_layout.addWidget(InfoButton("默认对数据流程有严格要求，如果绘制失败，请调到ROI模式重试"))
        # whole_cell_layout.addWidget(self.retransform_input)
        # whole_cell_layout.addWidget(self.retransform_btn)
        whole_cell_layout.addLayout(roi_signal_btn_layout)
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
        self.roi_distribution_btn = QPushButton("选区分布统计")
        Other_layout.addWidget(self.roi_distribution_btn)
        self.tDFT_btn2 = QPushButton("二维傅里叶变换")
        Other_layout.addWidget(self.tDFT_btn2)
        self.tDiFT_btn = QPushButton("二维傅里叶逆变换")
        Other_layout.addWidget(self.tDiFT_btn)
        self.atam_btn2 = QPushButton("累计时间振幅图")
        Other_layout.addWidget(self.atam_btn2)
        self.heartbeat_btn = QPushButton("心肌细胞跳动分析")
        Other_layout.addWidget(self.heartbeat_btn)
        self.basic_math_btn = QPushButton("基础运算器")
        Other_layout.addWidget(self.basic_math_btn)
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
        if self.fuction_select.currentIndex() in (3, 4):
            self.between_stack.setCurrentIndex(3)
            self.update_status('准备就绪')
            self.fps_input.setEnabled(True)
            self.time_step_input.setEnabled(True)
            self.time_unit_combo.setEnabled(True)

    def mode_switch_change(self, mode:int):
        """模式改变后"""
        self.mode = mode
        if mode == 0:
            self.mode_label.setText('ROI模式')
            self.mode_info.setToolTip("""<div style='white-space: pre;'>ROI模式:<br>每次处理前都<b>需要</b>选择ROI，<br>数据也<b>需要</b>选择，ROI需要与数据匹配</div>""")
        elif mode == 1:
            self.mode_label.setText('默认模式')
            self.mode_info.setToolTip("""<div style='white-space: pre;'>默认模式:<br>按标准处理流程执行的快速操作，<br>数据源和ROI会按照默认流程<b>自动</b>选择</div>""")
        elif mode == 2:
            self.mode_label.setText('自由模式')
            self.mode_info.setToolTip("""<div style='white-space: pre;'>自由模式:<br>每次处理前都<b>需要</b>选择数据，<br>数据自由选择，但可能会报错（无法处理）</div>""")
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
        cal_settings_edit = edit_menu.addAction("计算设置")
        cal_settings_edit.triggered.connect(self.calculation_set_edit_dialog)

        # 编辑菜单-绘图设置调整
        plt_settings_edit = edit_menu.addAction("绘图设置")
        plt_settings_edit.triggered.connect(self.plt_settings_edit_dialog)

        # 编辑菜单-缓存设置调整
        cache_settings_edit = edit_menu.addAction("缓存设置")
        cache_settings_edit.triggered.connect(self.cache_settings_edit_dialog)

        # 数据操作
        data_manipulation_menu = self.menu.addMenu("数据操作")
        # 数据操作——数据计算器
        data_calculator = data_manipulation_menu.addAction('数据计算器')
        data_calculator.triggered.connect(self.process_math)

        # 数据操作——数据切片器
        data_cropper = data_manipulation_menu.addAction('数据切片器')
        data_cropper.triggered.connect(self.data_crop)

        # 历史数据管理
        data_menu = self.menu.addMenu('历史数据')
        history_cache_manager = data_menu.addAction('历史与缓存管理')
        history_cache_manager.triggered.connect(self.history_cache_manager)
        # 清除历史
        data_history_clear = data_menu.addAction('本次历史清除')
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
        about_action.triggered.connect(lambda: QDesktopServices.openUrl(QUrl("https://github.com/CSSAcslin/LifeCalor")))

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
        self.status_label.setMinimumWidth(120)
        self.status_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.status_bar.addWidget(self.status_label, 2)
        # 鼠标悬停显示
        self.mouse_pos_label = QLabel("光标位置: x= -, y= -, t= -; 图像值: -, 实际值: -")
        self.mouse_pos_label.setMinimumWidth(260)
        self.mouse_pos_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.status_bar.addWidget(self.mouse_pos_label, 3)
        self._handle_hover = self.make_hover_handler()
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setMinimumWidth(160)
        self.progress_bar.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.status_bar.addWidget(self.progress_bar, 4)

        # 状态指示灯 (红绿灯)
        self.status_light = QLabel()
        self.status_light.setFixedWidth(24)
        self.status_light.setPixmap(QPixmap(":/icons/green_light.png").scaled(16, 16))
        self.status_bar.addPermanentWidget(self.status_light)

    """控制台相关"""
    def setup_console(self):
        """设置控制台停靠窗口"""
        self.console_dock = QDockWidget("控制台", self)
        self.console_dock.setObjectName("ConsoleDock")

        # 创建控制台部件
        self.console_widget = ConsoleWidget(self)
        # self.command_processor = CommandProcessor(self)

        self.console_dock.setWidget(self.console_widget)
        # self.addDockWidget(Qt.BottomDockWidgetArea, self.console_dock)
        # self.splitDockWidget(self.plot_dock, self.console_dock, Qt.Vertical)
        # self.resizeDocks([self.plot_dock, self.console_dock], [340, 60], Qt.Vertical)
        # 设置控制台特性
        # self.console_dock.setMinimumWidth(200)
        # self.console_dock.setMinimumHeight(50)
        self.console_dock.setFeatures(QDockWidget.DockWidgetMovable |
                                      QDockWidget.DockWidgetFloatable |
                                      QDockWidget.DockWidgetClosable)

        # self.console_dock.setVisible(False)

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
    def start_managed_export(self, data, output_dir, prefix, format_type, is_temporal, arg_dict):
        task = self.task_coordinator.create_task(f"导出 {prefix}", "export")
        task.start()
        self._export_task_id = task.task_id
        self.dat_thread.set_cancellation_token(task.token)
        task.cancel_callback = self.dat_thread.cancel
        self.task_coordinator.task_updated.emit(task)
        self.managed_export_signal.emit(data, output_dir, prefix, format_type, is_temporal, arg_dict)

    def _export_progress(self, current, total):
        task_id = getattr(self, "_export_task_id", None)
        if task_id and self.task_coordinator.registry.get(task_id) is not None:
            self.task_coordinator.progress(task_id, current, total, "正在导出数据")
        else:
            self.update_progress(current, total)

    def _export_finished(self, _files):
        task_id = getattr(self, "_export_task_id", None)
        if task_id:
            self.task_coordinator.complete(task_id, "数据导出完成")

    def _export_failed(self, message):
        task_id = getattr(self, "_export_task_id", None)
        if task_id:
            self.task_coordinator.fail(task_id, message)

    def _export_cancelled(self):
        task_id = getattr(self, "_export_task_id", None)
        if task_id:
            self.task_coordinator.cancelled(task_id, "数据导出已取消")

    def import_thread_open(self):
        """数据导入线程开启（.11.1版本加入）"""
        self.import_thread = QThread()
        self.imp_thread = ImportManager()
        self.imp_thread.moveToThread(self.import_thread)
        # 信号连接
        self.data_import_signal.connect(self.imp_thread.import_dispatch)
        self.imp_thread.update_status.connect(self.update_status)
        self.imp_thread.update_QMessageBox.connect(self.update_QMessageBox)
        self.imp_thread.processing_progress_signal.connect(self._import_progress)
        self.imp_thread.import_finished.connect(self.import_result)
        self.imp_thread.import_failed.connect(self._import_failed)
        self.imp_thread.import_cancelled.connect(self._import_cancelled)

    def data_thread_open(self):
        """图像数据操作线程（.10.10版本加入 ）和例外数据处理线程（.11.2版本加入）"""
        self.data_thread = QThread()
        self.dat_thread = DataManager()
        self.dat_thread.moveToThread(self.data_thread)

        self.image_display.image_export_signal.connect(self.start_managed_export)
        self.managed_export_signal.connect(self.dat_thread.export_data)
        self.dat_thread.data_progress_signal.connect(self._export_progress)
        self.mass_export_signal.connect(self.start_managed_export)
        self.roi_processed_signal.connect(self.dat_thread.ROI_processed)
        self.dat_thread.processed_result.connect(self.processed_result)
        self.dat_thread.export_finished.connect(self._export_finished)
        self.dat_thread.export_failed.connect(self._export_failed)
        self.dat_thread.export_cancelled.connect(self._export_cancelled)

        #
        self.process_thread = QThread()
        self.proc_thread = DataProcessor()
        self.proc_thread.moveToThread(self.process_thread)
        self.amend_data_signal.connect(self.proc_thread.amend_data)
        self.detect_bad_frames_auto_signal.connect(self.proc_thread.detect_bad_frames_auto)
        self.fix_bad_frames_signal.connect(self.proc_thread.fix_bad_frames)
        self.proc_thread.plot_singal.connect(self.graph_plot.handle_plot_signal)
        self.proc_thread.plot_series_signal.connect(self.graph_plot.handle_from_image)
        self.proc_thread.processing_error_signal.connect(lambda error: show_app_error(self, error))
        self.roi_value_distribution_signal.connect(self.proc_thread.get_roi_value_distribution)

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
        self.mass_data_processor.processing_progress_signal.connect(self._calculator_progress)
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
        self.tDiFT_signal.connect(self.mass_data_processor.twoD_inverse_fourier_transform)
        self.heartbeat_signal.connect(self.mass_data_processor.heartbeat_movement)
        self.calculator_signal.connect(self.mass_data_processor.calculation_operation)
        self.mass_data_processor.calculator_completed.connect(self._calculator_completed)
        self.mass_data_processor.calculator_failed.connect(self._calculator_failed)

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
        self.general_file_btn.clicked.connect(self.load_general_file)
        self.general_format_selector.currentIndexChanged.connect(
            lambda index: self.general_color_policy.setEnabled(index in (0, 2)))
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
        self.tDiFT_btn.clicked.connect(self.process_tDiFT)
        self.vector_signal_btn.clicked.connect(self.vectorROI_signal_show)
        self.select_frames_btn.clicked.connect(self.vectorROI_selection)
        self.diffusion_coefficient_btn.clicked.connect(self.result_display.plot_variance_evolution)
        self.roi_fast_btn.clicked.connect(self.fast_roi_result)
        self.heartbeat_btn.clicked.connect(self.process_heartbeat)
        self.basic_math_btn.clicked.connect(self.process_math)
        self.signal_extract.clicked.connect(self.process_signal_avg)
        self.roi_distribution_btn.clicked.connect(self.process_roi_value_distribution)
        self.export_image_btn.clicked.connect(self.export_image)
        self.export_data_btn.clicked.connect(self.export_data)
        # 成像绘制信号
        self.image_display.add_canvas_signal.connect(self.add_new_canvas)
        self.image_display.draw_result_signal.connect(self.draw_result)
        self.image_display.params_update_signal.connect(lambda params : self.tool_params.update(params))
        self.image_display.render_status_signal.connect(self.handle_render_status)
        # 时间滑块
        # self.time_slider.valueChanged.connect(self.image_display.update_time_slice)
        self.time_slider_vertical.valueChanged.connect(self.update_result_display)
        # # 连接控制台信号
        # self.command_processor.terminate_requested.connect(self.stop_calculation)
        # self.command_processor.save_config_requested.connect(self.save_config)
        # self.command_processor.load_config_requested.connect(self.load_config)
        # self.command_processor.clear_result_requested.connect(self.clear_result)
        # 结果区域信号
        self.result_display.tab_type_changed.connect(self._handle_result_tab)
        self.add_data_btn.clicked.connect(self.data_plot_add)
        self.reset_data_btn.clicked.connect(self.data_plot_clear)

    def disconnect_canvas_signal(self, signal, slot):
        return disconnect_display_canvas_signal(signal, slot)

    def canvas_signal_connect(self):
        return self.canvas_signal_binder.rebind_all()

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
        """通过时间戳查找父数据名称。"""
        # Prefer class-level history because restored items may not be focused yet.
        data_history = getattr(Data, 'history', None)
        if data_history is None and self.data is not None:
            data_history = getattr(self.data, 'history', None)
        if data_history is not None:
            for data in list(data_history):
                if data.timestamp == timestamp:
                    return data.name

        processed_history = getattr(ProcessedData, 'history', None)
        if processed_history is None and self.processed_data is not None:
            processed_history = getattr(self.processed_data, 'history', None)
        if processed_history is not None:
            for processed in list(processed_history):
                if processed.timestamp == timestamp:
                    return processed.name

        return '已恢复历史'

    def dispatch_import(self, import_type, filepath, params):
        task = self.task_coordinator.create_task(
            f"导入 {os.path.basename(filepath) or import_type}",
            "import",
        )
        task.start()
        self._import_task_id = task.task_id
        self.imp_thread.set_cancellation_token(task.token)
        task.cancel_callback = self.imp_thread.cancel
        self.task_coordinator.task_updated.emit(task)
        self.data_import_signal.emit(import_type, filepath, params)
        return task

    def _import_progress(self, current, total):
        task_id = getattr(self, "_import_task_id", None)
        if task_id and self.task_coordinator.registry.get(task_id) is not None:
            self.task_coordinator.progress(task_id, current, total, "正在导入数据")
        else:
            self.update_progress(current, total)

    def _import_failed(self, message):
        task_id = getattr(self, "_import_task_id", None)
        if task_id and self.task_coordinator.registry.get(task_id) is not None:
            self.task_coordinator.fail(task_id, message)

    def _import_cancelled(self):
        task_id = getattr(self, "_import_task_id", None)
        if task_id and self.task_coordinator.registry.get(task_id) is not None:
            self.task_coordinator.cancelled(task_id, "数据导入已取消")

    def load_general_file(self):
        format_index = self.general_format_selector.currentIndex()
        filters = {
            0: "支持的数据 (*.npy *.tif *.tiff);;所有文件 (*)",
            1: "NumPy 数据 (*.npy);;所有文件 (*)",
            2: "TIFF 图像 (*.tif *.tiff);;所有文件 (*)",
        }
        import_types = {0: "auto_file", 1: "npy", 2: "tiff_file"}
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择数据文件", self.settings.value("last_folder", ""), filters[format_index]
        )
        if not file_path:
            logging.info("用户取消选择通用数据文件")
            return False
        options = {
            **self.basic_params,
            "time_basis": "fps" if self.general_time_basis.currentIndex() == 1 else "time_step",
            "color_policy": "preserve" if self.general_color_policy.currentIndex() == 0 else "grayscale",
        }
        if options["time_basis"] == "fps":
            options["fps"] = self.fps_input.value()
            options["time_unit"] = "s"
        self.dispatch_import(import_types[format_index], file_path, options)
        return True

    def load_npy(self):
        """兼容旧调用；界面入口已统一迁移到左侧导入设置。"""
        previous = self.general_format_selector.currentIndex()
        self.general_format_selector.setCurrentIndex(1)
        try:
            return self.load_general_file()
        finally:
            self.general_format_selector.setCurrentIndex(previous)

    def load_tiff_folder(self):
        """加载TIFF文件夹(FS-iSCAT)"""
        self.time_step = float(self.time_step_input.value())
        folder_path = QFileDialog.getExistingDirectory(self, "选择TIFF图像文件夹",self.settings.value("last_folder", ""))
        if folder_path:
            logging.info(folder_path)
            self.update_status("已加载TIFF文件夹",'idle')
            current_group = self.group_selector.currentText()
            self.dispatch_import('tif_series', folder_path, {'current_group':current_group,**self.basic_params})

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
            self.dispatch_import('sif_folder', folder_path, {'normalize_type': self.method_combo.currentText(), **self.basic_params})

        elif not folder_path:
            self.update_status("文件夹选择已取消", 'idle')
            return

    def load_avi(self):
        """加载avi读取线程传递函数"""
        self.status_label.setText("正在处理数据...")
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

        self.dispatch_import('avi_EM', file_path, {'fps': self.fps_input.value(), **self.basic_params})
        self.update_param('EM','EM_fps', self.fps_input.value())

    def load_tiff_folder_EM(self):
        """加载TIFF文件夹(FS-iSCAT)"""
        self.status_label.setText("正在处理数据...")
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
                self.dispatch_import('tif_EM', folder_path, {'fps': self.fps_input.value(), **self.basic_params})
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
        task_id = getattr(self, "_import_task_id", None)
        if task_id and self.task_coordinator.registry.get(task_id) is not None:
            self.task_coordinator.complete(task_id, "数据导入完成")
        self.update_status("已加载文件", 'idle')
        # 成像显示
        self.load_image(origin_data=self.data)

    """画布设置相关"""
    def add_new_canvas(self, assign_data=None):
        """新建图像显示画布。"""
        return self.display_canvas_controller.add_new_canvas(assign_data)

    def load_image(self, data_type='original', other_params: str = None, origin_data=None):
        """加载图像显示画布。"""
        return self.display_canvas_controller.load_image(data_type, other_params, origin_data)

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
            value_text = format_hover_value(args['value'])
            origin_text = format_hover_value(args['origin'])
            text = f"光标位置: x={args['x']}, y={args['y']}, t={args['t']}; 图像值: {value_text}, 实际值: {origin_text}"
            self.mouse_pos_label.setText(text)
            self.mouse_pos_label.setToolTip(text)

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
        # if self.data is None or self.processed_data is None:
        #     logging.warning("无数据，请先加载数据文件")
        #     return
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

    def cache_settings_edit_dialog(self):
        """缓存设置。Legacy CacheSettingsDialog 入口转到统一历史与缓存管理。"""
        return self.history_cache_manager()

    def cleanup_array_cache_orphans(self):
        """清理当前历史和可恢复索引都未引用的临时缓存文件。"""
        cache_dir = self.tool_params.get('cache_directory') or self.default_cache_directory()
        store = HistoryManifestStore(Path(cache_dir))
        active_refs = collect_array_refs(Data.history) + collect_array_refs(ProcessedData.history)
        active_refs += array_refs_for_manifest(store.load())
        deleted = clear_array_cache(active_refs)
        removed_manifest_items = store.remove_missing_items()
        if deleted:
            logging.info(f"已自动清理孤立缓存文件 {deleted} 个")
        if removed_manifest_items:
            logging.info(f"已移除失效可恢复历史索引 {removed_manifest_items} 条")
        return deleted

    def clear_array_cache_files(self):
        """手动清除全部缓存文件，并同步清空可恢复历史索引。"""
        Data.clear_history(remove_cache=True)
        ProcessedData.clear_history(remove_cache=True)
        deleted = clear_array_cache()
        cache_dir = self.tool_params.get('cache_directory') or self.default_cache_directory()
        removed_manifest_items = HistoryManifestStore(Path(cache_dir)).clear_items()
        logging.info(f"已清除缓存文件 {deleted} 个，移除可恢复索引 {removed_manifest_items} 条")
        self.update_status("缓存已清除", 'idle')
        return deleted

    def start_calculation(self):
        """开始计算时调用此方法"""
        self.elapsed_timer.start()
        self.last_time = 0
        self.last_progress = 0
        self.last_percent = -1
        self.cached_remaining = "计算中..."

    """状态响应与更新"""
    def update_progress(self, current, total=None):
        """更新进度条，并把超大 byte 计数缩放到 QProgressBar 安全范围。"""
        if current == -1:
            self.progress_bar.reset()
            return

        scaled_current, scaled_total, current_percent = normalize_progress(current, total)
        if scaled_total is not None:
            self.progress_bar.setMaximum(scaled_total)

        if scaled_current == 0:
            self.start_calculation()
        self.progress_bar.setValue(scaled_current)

        elapsed_ms = self.elapsed_timer.elapsed()
        elapsed_sec = elapsed_ms / 1000.0

        if int(current_percent) > self.last_percent:
            if scaled_current > self.last_progress and elapsed_ms > self.last_time:
                progress_diff = scaled_current - self.last_progress
                time_diff = (elapsed_ms - self.last_time) / 1000.0
                speed = progress_diff / time_diff if time_diff > 0 else 0
                self.last_progress = scaled_current
                self.last_time = elapsed_ms
                if speed > 0 and scaled_total is not None:
                    remaining_sec = (scaled_total - scaled_current) / speed
                    self.cached_remaining = self.format_time(remaining_sec)
            self.last_percent = int(current_percent)

        elapsed_str = self.format_time(elapsed_sec)
        maximum = self.progress_bar.maximum()
        self.progress_bar.setFormat(
            f"进度: {scaled_current}/{maximum} "
            f"({current_percent:.1f}%) | "
            f"已用: {elapsed_str} | 预计剩余: {self.cached_remaining}"
        )

        if scaled_total is not None and scaled_total > 0 and scaled_current >= scaled_total:
            logging.info(f"计算完成，总耗时{elapsed_str}")
            self.update_status("进程任务完成,准备就绪", 'idle')
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

    def handle_render_status(self, status, message):
        update = render_status_update(status, message)
        if update is None:
            return
        text, state = update
        self.update_status(text, state)

    def _on_task_updated(self, task):
        """Reflect only the current foreground task in the compact status bar."""
        foreground = self.task_coordinator.registry.foreground()
        if foreground is not None and foreground.task_id != task.task_id:
            return
        self._displayed_task_id = task.task_id
        if task.status == TaskStatus.CANCELLING:
            self.update_status(f"正在中断: {task.name}", "working")
            return
        if task.status == TaskStatus.RUNNING:
            self.update_status(task.message or task.name, "working")
            if task.total:
                self.update_progress(task.current, task.total)

    def _on_task_finished(self, task):
        if self._displayed_task_id != task.task_id:
            return
        next_task = self.task_coordinator.registry.foreground()
        if next_task is not None and next_task.task_id != task.task_id:
            self._on_task_updated(next_task)
            return
        self._displayed_task_id = None
        if task.status == TaskStatus.FAILED:
            self.update_status(f"任务失败: {task.name}", "error")
            show_app_error(self, AppError("任务失败", task.error or task.name, stage=task.category, severity="error"))
        elif task.status == TaskStatus.CANCELLED:
            self.update_status(f"已中断: {task.name}", "idle")
        else:
            self.update_status(task.message or f"任务完成: {task.name}", "idle")
        self.update_progress(-1)

    def cancel_active_task(self):
        """Request cancellation without waiting in the GUI thread."""
        if self.task_coordinator.cancel_foreground_task():
            return True
        logging.info("当前没有可中断的前台任务")
        self.update_status("当前没有可中断任务", "idle")
        return False

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.cancel_active_task()
            event.accept()
            return
        super().keyPressEvent(event)

    def handle_logged_error(self, level, message):
        concise = str(message).splitlines()[0]
        show_app_error(self, AppError(
            "运行错误",
            concise,
            stage="日志错误桥",
            severity="critical" if level == "CRITICAL" else "error",
            details=str(message),
        ))

    def handle_unhandled_exception(self, exc_type, exc_value, exc_traceback):
        details = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        show_app_error(self, AppError(
            "程序发生未捕获错误",
            f"{exc_type.__name__}: {exc_value}\n详细信息已写入日志。",
            stage="未捕获异常",
            severity="critical",
            details=details,
            original=exc_value,
        ))

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
        elif working_status in ('error', 'failed'):
            logging.error(status, extra={"lifecalor_user_reported": True})
            light = "red_light.png"
        else:
            light = "red_light.png"
        self.status_light.setPixmap(QPixmap(f":/icons/{light}").scaled(16, 16))

    def update_QMessageBox(self, message_type, title, message):
        if message_type == "error":
            foreground = self.task_coordinator.registry.foreground()
            if foreground is not None:
                self.task_coordinator.fail(foreground.task_id, message)
            else:
                show_app_error(self, AppError(title, message, severity="error"))
                self.update_status(message, "error")
        elif message_type == "warning":
            logging.warning("%s: %s", title, message)
            self.update_status(message, "warning")
        else:
            logging.info("%s: %s", title, message)
            self.update_status(message, "idle")

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
            report_warning(
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
        self.ensure_task_thread_running("calc_thread", "calculation")
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
        self.ensure_task_thread_running("calc_thread", "calculation")
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
        self.ensure_task_thread_running("calc_thread", "calculation")
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

        self.ensure_task_thread_running("calc_thread", "calculation")
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
        self.ensure_task_thread_running("avi_thread", "em_processing")
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
                self.ensure_task_thread_running("avi_thread", "em_processing")
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
            self.ensure_task_thread_running("avi_thread", "em_processing")
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
            self.ensure_task_thread_running("avi_thread", "em_processing")
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
            self.ensure_task_thread_running("avi_thread", "em_processing")
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
        if data is None:
            return False
        mask = self.roi_selection(True)
        if mask is None:
            return
        if mask.shape != data.framesize:
            report_warning(self, "蒙版错误", "蒙版尺寸与数据不匹配")
            return
        self.ensure_task_thread_running("calc_thread", "calculation")
        self.update_status('计算进行中...', 'working')
        self.easy_process.emit(data,'avg',mask)

    def process_signal_avg(self):
        """广义信号平均"""
        aim_data = self.data_selection()
        if aim_data is None:
            return False
        self.ensure_task_thread_running("avi_thread", "em_processing")
        self.ensure_task_thread_running("calc_thread", "calculation")
        self.update_status('计算进行中...', 'working')
        self.easy_process.emit(aim_data,'avg',None)
        return None


    def _default_distribution_frame_index(self, data):
        if getattr(data, "ndim", 0) != 3:
            return 0
        for canvas in getattr(self.image_display, "display_canvas", []):
            canvas_data = getattr(canvas, "data", None)
            if getattr(canvas_data, "timestamp_inherited", None) == getattr(data, "timestamp", None):
                return max(0, min(int(getattr(canvas, "current_time_idx", 0)), data.timelength - 1))
        return 0

    def process_roi_value_distribution(self):
        """计算选区当前帧值分布并显示到 PlotGraph。"""
        aim_data = self.data_selection()
        if aim_data is None:
            return False
        mask = self.roi_selection(True)
        if mask is None:
            return False
        if mask.shape != aim_data.framesize:
            report_warning(self, "蒙版错误", "蒙版尺寸与数据不匹配")
            return False

        default_frame = self._default_distribution_frame_index(aim_data)
        try:
            frame = DataProcessor._distribution_frame(aim_data, default_frame)
            _, metadata = DataProcessor.value_distribution_from_frame(frame, mask, bins=1)
        except Exception as exc:
            show_app_error(self, AppError("选区分布统计失败", str(exc), stage="选区分布统计", severity="warning"))
            return False

        max_frame = aim_data.timelength - 1 if getattr(aim_data, "ndim", 0) == 3 else 0
        dialog = ValueDistributionDialog(metadata["min"], metadata["max"], max_frame=max_frame, default_frame=default_frame, parent=self)
        if not dialog.exec_():
            return False
        try:
            config = dialog.get_config()
        except ValueError as exc:
            report_warning(self, "参数错误", str(exc))
            return False

        frame_index = config["frame_index"]
        name = f"{aim_data.name}-frame{frame_index}-ROI值分布"
        self.update_status("选区分布统计中...", "working")
        self.roi_value_distribution_signal.emit(
            aim_data,
            mask,
            frame_index,
            config["bins"],
            config["value_range"],
            name,
        )
        return True

    def process_atam(self):
        """累计时间振幅图"""
        aim_data = self.data_selection()
        if aim_data is None:
            return False
        if aim_data.ndim not in [2, 3]:
            logging.info("数据无法被处理，请重选数据")
            return False
        self.ensure_task_thread_running("avi_thread", "em_processing")
        self.atam_signal.emit(aim_data)
        self.atam_btn.setEnabled(False)
        return True

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
                self.ensure_task_thread_running("avi_thread", "em_processing")
                self.tDgf_signal.emit(data,
                                      self.EM_params['scs_zoom'],
                                      self.EM_params['scs_thr'],
                                      self.EM_params['thr_known'])
                self.tDgf_btn.setEnabled(False)
        else:
            report_warning(self, "数据错误", "不支持的数据类型，请确认前序处理是否正确（是否确认ROI）")
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
                self.ensure_task_thread_running("avi_thread", "em_processing")
                self.sscs_signal.emit(data,
                                      self.EM_params['scs_zoom'],
                                      self.EM_params['scs_thr'],
                                      self.EM_params['thr_known'])
                self.sscs_btn.setEnabled(False)
        else:
            report_warning(self, "数据错误", "不支持的数据类型，请确认前序处理是否正确（是否确认ROI）")
            self.update_status("准备就绪", 'idle')
            return

    def process_tDFT(self):
        """二维傅里叶变换"""
        aim_data = self.data_selection()
        if aim_data is None:
            return False
        self.ensure_task_thread_running("avi_thread", "em_processing")
        self.update_status('二维傅里叶变换计算进行中...', 'working')
        self.tDFT_signal.emit(aim_data)
        return True

    def process_tDiFT(self):
        """二维傅里叶逆变换"""
        aim_data = self.data_selection()
        if aim_data is None:
            return False
        self.ensure_task_thread_running("avi_thread", "em_processing")
        self.update_status('二维傅里叶逆变换计算进行中...', 'working')
        self.tDiFT_signal.emit(aim_data)
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

    def calculator_sources(self):
        """Return current and historical data objects without duplicate entries."""
        sources = []
        seen = set()
        candidates = [self.data, self.processed_data, *list(Data.history), *list(ProcessedData.history)]
        for source in candidates:
            if source is None:
                continue
            identity = (source.__class__.__name__, getattr(source, "timestamp", None), getattr(source, "serial_number", id(source)))
            if identity in seen:
                continue
            seen.add(identity)
            sources.append(source)
        return sources

    def process_math(self):
        """Open or focus the non-modal multi-source calculation workspace."""
        existing = getattr(self, "data_calculator", None)
        if existing is not None:
            existing.show()
            existing.raise_()
            existing.activateWindow()
            return True

        dialog = DataCalculatorDialog(
            [],
            self,
            source_provider=self.calculator_sources,
            import_callback=self.load_general_file,
            metadata_defaults=self.basic_params,
        )
        self.data_calculator = dialog
        dialog.execute_requested.connect(self._submit_calculator_plan)
        dialog.closed.connect(self._calculator_closed)
        self.imp_thread.import_finished.connect(dialog.handle_import_finished)
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
        return True

    def _calculator_closed(self):
        dialog = getattr(self, "data_calculator", None)
        if dialog is None:
            return
        try:
            self.imp_thread.import_finished.disconnect(dialog.handle_import_finished)
        except (TypeError, RuntimeError):
            pass
        self.data_calculator = None

    def _submit_calculator_plan(self, plan):
        active_id = getattr(self, "_calculator_task_id", None)
        active = self.task_coordinator.registry.get(active_id) if active_id else None
        if active is not None and active.status in {
            TaskStatus.PENDING, TaskStatus.RUNNING, TaskStatus.CANCELLING,
        }:
            error = AppError(
                "多数据运算正忙",
                "已有一个多数据运算任务在执行，请等待完成后再提交",
                stage="多数据运算提交",
                severity="warning",
            )
            dialog = getattr(self, "data_calculator", None)
            if dialog is not None:
                dialog.set_execution_failed(error)
            show_app_error(self, error)
            return False
        self.ensure_task_thread_running("avi_thread", "em_processing")
        task = self.task_coordinator.create_task(
            "多数据运算", "calculation", foreground=True, cancellable=False
        )
        self._calculator_task_id = task.task_id
        self.task_coordinator.start(task.task_id, 1, "正在执行多数据运算")
        self.update_status("多数据运算中...", "working")
        self.calculator_signal.emit(plan)
        return True

    def _calculator_progress(self, current, total):
        task_id = getattr(self, "_calculator_task_id", None)
        if not task_id:
            return
        task = self.task_coordinator.registry.get(task_id)
        if task is not None:
            self.task_coordinator.progress(
                task_id, current, total, "正在执行多数据运算"
            )

    def _calculator_completed(self, result):
        task_id = getattr(self, "_calculator_task_id", None)
        if task_id and self.task_coordinator.registry.get(task_id) is not None:
            self.task_coordinator.complete(task_id, "多数据运算完成")
        dialog = getattr(self, "data_calculator", None)
        if dialog is not None:
            dialog.set_execution_finished(result)
        self._calculator_task_id = None
        self.update_status("多数据运算完成", "idle")

    def _calculator_failed(self, error):
        task_id = getattr(self, "_calculator_task_id", None)
        if task_id and self.task_coordinator.registry.get(task_id) is not None:
            self.task_coordinator.fail(task_id, error.message)
        self._calculator_task_id = None
        dialog = getattr(self, "data_calculator", None)
        if dialog is not None:
            dialog.set_execution_failed(error)
        show_app_error(self, error)
        self.update_status("多数据运算失败", "error")

    def data_crop(self):
        """数据空间切片器"""
        aim_data = self.data_selection()
        if aim_data is None:
            return False
        dialog = SpatialExtractor(aim_data)
        if dialog.exec_() == QDialog.Accepted:
            logging.info(f"数据切割完成, 切割后维度: {dialog.extracted_data.datashape}")
            self.processed_result(dialog.extracted_data)

    def data_selection(self, aim_type:str | list = 'all'):
        """数据选择代码（模式流程）"""
        return self.selection_controller.select_data(aim_type)

    def roi_selection(self, select = False):
        """ROI选择"""
        return self.selection_controller.select_roi(select)

    def data_pick(self, need_all=True):
        """数据选择"""
        return self.selection_controller.pick_data(need_all)

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
            process_type = data.get('type', '未知处理') if isinstance(data, dict) else '未知处理'
            error_message = data.get('error', '未知错误') if isinstance(data, dict) else str(data)
            show_app_error(self, AppError(
                "运算错误",
                f"在{process_type}处理中报错：\n{error_message}",
                stage=process_type,
                severity="warning",
                details=error_message,
            ))
            self.update_progress(-1) # 进度条重置
            return False
        foreground_task = self.task_coordinator.registry.foreground()
        if foreground_task is not None and foreground_task.category in {"calculation", "em_processing"}:
            self.task_coordinator.complete(foreground_task.task_id, "数据处理完成")
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
            case 'Basic_math':
                logging.info("对数据的基础运算完毕！")
            case 'data_cropped':
                logging.info("对数据的切片完成！")

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
                        except StopIteration: # 如果不行就从data里找
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
        """从树结构选择器中来，出新成像。"""
        return self.display_canvas_controller.upgrade_and_imaging(data, key)

    '''其他功能'''
    def is_thread_active(self, thread_name: str) -> bool:
        """检查指定名称的线程是否存在且正在运行"""
        thread = getattr(self, thread_name, None)
        return thread_is_active(thread, expected_type=QThread, is_deleted=sip.isdeleted)

    def ensure_task_thread_running(self, thread_name: str, task_key: str) -> bool:
        """启动兼容线程，并在统一任务注册表登记可取消的前台任务。"""
        thread = getattr(self, thread_name, None)
        worker_map = {
            "import": getattr(self, "imp_thread", None),
            "calculation": getattr(self, "cal_thread", None),
            "em_processing": getattr(self, "mass_data_processor", None),
            "export": getattr(self, "dat_thread", None),
        }
        worker = worker_map.get(task_key)
        if worker is not None and hasattr(worker, "abortion"):
            worker.abortion = False

        def request_cancel():
            if worker is not None:
                if hasattr(worker, "abortion"):
                    worker.abortion = True
                stop_method = getattr(worker, "stop", None)
                if callable(stop_method):
                    stop_method()
            if thread is not None and hasattr(thread, "requestInterruption"):
                thread.requestInterruption()

        task = self.task_coordinator.create_task(
            name={"import": "导入数据", "calculation": "寿命计算", "em_processing": "数据处理", "export": "导出数据"}.get(task_key, task_key),
            category=task_key,
            cancel_callback=request_cancel,
        )
        task.start()
        self._legacy_task_ids[task_key] = task.task_id
        self.task_coordinator.task_updated.emit(task)
        return ensure_thread_running(
            thread,
            self.task_states[task_key],
            expected_type=QThread,
            is_deleted=sip.isdeleted,
        )

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
        """停止指定后台线程。"""
        thread_map = {
            0: ("calc_thread", "calculation", "计算线程关闭"),
            1: ("avi_thread", "em_processing", "大数据处理线程关闭"),
        }
        if type not in thread_map:
            logging.warning(f"未知线程类型: {type}")
            return False
        thread_name, task_key, success_message = thread_map[type]
        try:
            stopped = stop_qthread(getattr(self, thread_name, None), expected_type=QThread, is_deleted=sip.isdeleted)
            if stopped:
                self.task_states[task_key].complete()
                logging.info(success_message)
            return stopped
        except Exception as e:
            self.task_states[task_key].fail(str(e))
            logging.error(f"线程退出错误{e}")
            return False

    def export_image(self):
        """导出热图为图片"""
        return self.export_controller.export_image()

    def export_data(self):
        """导出寿命数据"""
        return self.export_controller.export_data()

    def export_EM_data(self,result):
        """时频变换后目标频率下的结果导出"""
        return self.export_controller.export_em_data(result)

    def history_cache_manager(self):
        """统一历史与缓存管理。"""
        return self.history_controller.history_cache_manager()

    def data_history_view(self):
        """导入数据历史查看。"""
        return self.history_controller.data_history_view()

    def process_history_view(self):
        """处理数据历史查看。"""
        return self.history_controller.process_history_view()

    def load_cached_history_async(self, target, attr_name):
        """在线程中读取缓存数组，避免历史选择时阻塞 GUI。"""
        return self.history_controller.load_cached_history_async(target, attr_name)

    def data_history_clear(self):
        """本次历史清除：只清空内存历史，不删除已落盘缓存。"""
        if Data is not None:
            Data.clear_history(remove_cache=False)
            logging.info('本次导入数据历史已清除，已落盘缓存保留')
        else:
            logging.warning('没有数据可清除')
        if ProcessedData is not None:
            ProcessedData.clear_history(remove_cache=False)
            logging.info('本次处理数据历史已清除，已落盘缓存保留')

    '''以下控制台命令更新'''
    def stop_calculation(self):
        """兼容控制台入口：转发为非阻塞的全局任务取消请求。"""
        logging.warning("任务中断请求已接收")
        return self.cancel_active_task()

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
    app.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app.setAttribute(Qt.AA_UseHighDpiPixmaps)
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
