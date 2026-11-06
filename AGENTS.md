# LifeCalor 项目总览与开发交接

LifeCalor 是基于 PyQt5 的本地科学成像分析工具。超快成像工作流从衰减曲线提取载流子寿命、生成像素级寿命图并分析扩散。EM-iSCAT 工作流从图像序列提取 STFT/CWT 时频信号，支持全细胞和单通道分析。两条工作流共用多画布、ROI、数据计算器、历史缓存和导出。[用户操作手册](readme.md)

**交接基线（2026-11-05）**　源码版本 `1.1.1`，分支 `main`，提交 `a3bf420`。本机的依赖清单、README 安装步骤和 `.venv/` 忽略规则是该提交之后的工作区补充。后续接手时先运行 `git status --short --branch` 和 `git log -1 --oneline`，再判断本文哪些结论适用于当前版本。[版本常量](core/app_metadata.py) · [版本变更记录](updatinglog.md)

## 建立环境与启动

以下命令在仓库根目录用 PowerShell 执行。本机项目环境是 64 位 Python 3.12.14。新电脑装好 Python 3.12 后，执行前两行创建环境并安装依赖。已有项目 `.venv` 时，可直接从依赖检查开始。[固定版本清单](requirements.txt)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe core/MainWindow.py
```

`MainWindow.py` 作为脚本运行时会先进入 `launcher.main()`。启动器在创建 `QApplication` 前配置高 DPI 和 WebEngine 共享上下文，再加载主题、启动页和主窗口。硬件探测与更新检查安排在主窗口显示之后。修改启动顺序时可对照这三个入口。[脚本入口](core/MainWindow.py) · [启动器](core/launcher.py) · [主窗口装配](core/startup/bootstrap.py)

## 从导入到一张寿命热图

1. **文件进入 `Data`。** 通用导入支持 NPY、TIFF/OME-TIFF、AVI/MP4、Andor SIF 和 HDF5 数值数据集。`ImportManager._load_registered_file()` 先 `probe` 获取形状与 dtype，检查内存预算，再 `read` 取得 `ImportedPayload`。`data_origin` 保存科学分析值，`image_import` 可保留彩色显示数据，`time_point` 和 `parameters` 携带时间轴、尺度等信息。超快成像分组文件夹和 EM 专用导入另有入口。[导入调度](core/ImportManager.py) · [导入接口](core/importing/model.py)
2. **画布建立显示源并确认 ROI。** `ImagingData` 与 `DisplaySourceFactory` 把导入数据或处理结果接到画布。定量统计和 ROI 分析取科学数值，纯数据导出读取选定的数组或字段，带样式导出则逐帧应用显示范围与伪彩。三维数组可通过明确的 `T` 轴或唯一匹配的时间轴长度识别为视频，轴含义不明确时保留矩阵语义。[数据模型](core/DataManager.py) · [数据分类](core/dataio/classification.py) · [显示源](core/display/source.py)
3. **主窗口提交后台计算。** `MainWindow.distribution_analyze_start()` 收集数据、拟合模型、卷积和计算选项，经 `ProcessingController.begin()` 登记任务，再发出 `start_dis_cal_signal`。`calc_thread` 是 Qt 线程，`cal_thread` 是其中的 `CalculationThread` 工作对象。其 `distribution_analyze()` 调用寿命计算管线。[任务编排](core/processing/controller.py) · [寿命工作对象](core/LifetimeCalculator.py) · [寿命管线](core/compute/algorithms/lifetime_pipeline.py)
4. **`ProcessedData` 把结果带回界面。** 热图的 `type_processed` 是 `lifetime_distribution`。`data_processed` 保存主结果，`out_processed` 带回具名结果数组、`result_fields`、`active_result_field` 和 `compute` 执行信息。`MainWindow.processed_result()` 交给 `ResultRouter.route()` 分发。切换热图字段时读取已有结果，结果页的纯数据导出跟随当前字段。[结果路由](core/processing/result_router.py) · [结果展示](core/ResultDisplayWidget.py) · [字段联动测试](tests/test_lifetime_result_fields.py)

EM-iSCAT 共用上述任务与结果机制。对应的工作对象是 `MassDataProcessor`。它用起始若干帧的逐像素中位数估计背景，再按用户选择调用 STFT 或 CWT 管线。改动某个变换时，同时核对工作对象写回 `ProcessedData` 的结果轴和计算元数据。[EM 工作对象](core/DataProcessor.py) · [预处理](core/compute/algorithms/preprocess_pipeline.py) · [STFT 管线](core/compute/algorithms/stft_pipeline.py) · [CWT 管线](core/compute/algorithms/cwt_pipeline.py)

## 按更新内容定位代码

| 更新内容 | 首先看哪里 | 对应回归用例 |
| --- | --- | --- |
| 新增文件格式 | 在 [readers.py](core/importing/readers.py) 实现探测与读取，加入 [default_importer_registry](core/importing/registry.py)。界面的文件过滤器和显式格式入口在 `MainWindow.load_general_file()` 与 `ImportManager.type_all`。 | [导入注册与格式](tests/test_importer_registry.py) |
| 新增算法、修改精度或分块 | [内置算法规格](core/compute/registry.py) 定义轴、dtype、输出字段和后端处理函数。[规划器](core/compute/planner.py) 生成执行计划，[算法目录](core/compute/algorithms/) 实现各条管线。`core/AlgorithmRegistry.py` 是运行时可调用对象注册的兼容入口。 | [算法规格与分块](tests/test_compute_111_registry_and_blocks.py)、[计算契约](tests/test_compute_contracts.py) |
| 修改画布、播放、ROI 或 Anchor | [ImageDisplayWindow.py](core/ImageDisplayWindow.py) 处理画布交互，[display](core/display/) 负责显示源、帧渲染、缓存、时间轴和布局，[selection](core/selection/) 连接数据与选区。 | [显示生命周期](tests/test_canvas_lifecycle.py)、[时间轴与播放](tests/test_time_axis_and_playback.py)、[Anchor 统计](tests/test_anchor_distribution.py) |
| 计算器、切片或结果元数据 | [计算器引擎](core/calculator/engine.py) 用 AST 白名单验证并执行表达式，[元数据策略](core/calculator/metadata.py) 处理尺度和轴继承。[SpatialExtractor.py](core/SpatialExtractor.py) 是交互式切片入口。 | [表达式计算](tests/test_calculator_engine.py)、[计算器工作区](tests/test_calculator_workspace.py) |
| 大数组、历史或恢复 | [ArrayCache.py](core/ArrayCache.py) 管理 `.npy` 引用，[分块 I/O](core/dataio/npy.py) 负责读写，[历史清单](core/history/manifest.py) 与 [历史控制器](core/history/controller.py) 负责索引、恢复和清理。 | [分块读写](tests/test_chunked_npy_io.py)、[历史与缓存](tests/test_history_cache_manager.py) |
| 后台进度、取消或错误提示 | [tasks](core/tasks/) 管理任务 ID、状态和取消令牌，[处理控制器](core/processing/controller.py) 将操作排入工作线程，[diagnostics](core/diagnostics/) 统一错误、日志与弹窗。 | [任务上下文隔离](tests/test_processing_task_context.py)、[错误上报](tests/test_error_reporting.py) |
| 导出曲线、图像或视频 | [导出控制器](core/exporting/controller.py) 组织操作，[导出策略](core/exporting/policy.py) 与 [数据管理](core/DataManager.py) 处理格式、帧范围和科学数组。纯数据支持 NPY、HDF5、CSV 和 TIFF，带样式的图像和视频使用 [帧渲染器](core/display/renderer.py)。 | [导出流程](tests/test_export_workflow.py)、[逐帧 AVI](tests/test_styled_avi_export.py) |
| 统一选项、主题与资源 | [preferences](core/preferences/) 管理各页草稿、应用和冲突检测，[计算设置](core/compute/settings.py) 管理按算法覆盖，[appearance](core/appearance/) 提供主题和窗口装饰。修改图标时一并核对 `resources.qrc` 和 `resources_rc.py`。 | [选项窗口](tests/test_preferences_dialog.py)、[主题](tests/test_appearance.py) |

## 改动时需要保留的行为

**时频分析。** STFT 对选定频点的幅值取均值，输出时间轴对应变换窗口。实数单边谱对常规频点做两倍幅值校正，直流和偶数 FFT 的奈奎斯特频点使用原倍率。CWT 信号结果对各尺度的 `2 * abs(coef) / sqrt(scale)` 取均值，质量分析则保留“尺度 × 时间”二维谱。项目以 `PyWavelets==1.8.0` 及小波核 `precision=10` 为对照，升级依赖时应复核七种小波、实数和复数输入。[STFT 参考](core/compute/algorithms/stft.py) · [CWT 参考](core/compute/algorithms/cwt.py) · [数值回归](tests/test_compute_111_stage_a.py)

**寿命拟合。** CPU 数值参考使用 SciPy `curve_fit`、`trf`、受界参数和 `float64`。单指数热图输出 `lifetime_map`、`r_squared_map` 和 `fit_status`。双指数分别保存 `tau1_map`、`tau2_map`、`amplitude1_map`、`amplitude2_map`、`baseline_map`、`r_squared_map` 和 `fit_status`。拟合状态用于区分成功、输入异常、筛选拒绝和求解失败。[拟合模型与约束](core/compute/algorithms/lifetime.py) · [寿命管线回归](tests/test_lifetime_pipeline.py)

**任务与设备。** 每个计算任务在提交时取得设置与独立取消令牌，工作对象按队列取上下文。分块循环在安全点检查取消，进度单调递增，取消一个任务后同线程的下一项仍可运行。CUDA 由隔离 worker 执行，严格 GPU 与允许 CPU 回退是两种用户策略。核对执行结果时读取 `requested_backend`、`actual_backend`、精度、设备及回退原因。单曲线 CWT 质量分析在 Auto 下走 CPU。[任务上下文](core/tasks/context.py) · [CUDA worker](core/compute/worker.py) · [执行元数据](core/compute/model.py)

**内存、磁盘与旧数据。** 大数组可能是 `ArrayRef`。估算内存时使用实际 dtype 和 `nbytes`，计算按块读取，超阈值结果可写入磁盘。多字段计算按整体预算规划，写入失败或取消时清理整组未完成结果。恢复 `history_manifest.json` 时重建轻量引用并兼容旧时间轴。历史身份使用 `(kind, serial_number, timestamp)`，显示名和标签可单独更改。删除时保护仍被其他索引或画布引用的数组。[原子结果写入](core/compute/io.py) · [历史身份](core/history/annotations.py) · [缓存恢复](core/history/manifest.py)

**设置与显示状态。** 六类选项共用非模态窗口，应用或确定时只写已变更的字段，并检查同字段冲突。维护 `QSettings` 键时考虑已有用户配置。画布的物理时间轴、当前帧、ROI、伪彩和布局各有状态，修改默认设置时沿用相应控制器的更新范围。[选项应用逻辑](core/preferences/dialog.py) · [画布布局](core/display/layout_manager.py)

## 验证与发布

选择上表中与改动对应的用例先跑，再执行全套。下面以寿命管线为例。`benchmarks/benchmark_compute_11.py` 可用同一算法、规模和精度比较 CPU/GPU，`benchmark_data_path.py` 用于数据路径基准。

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_lifetime_pipeline.py" -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
$LASTEXITCODE
.\.venv\Scripts\python.exe benchmarks/benchmark_compute_11.py --algorithm stft --size small --backend cpu --repetitions 1
```

- **本机 CPU 基线。** `pip check` 已通过，主窗口已在离屏模式构建并显示。寿命管线的 7 项定向测试正常退出。全套发现 405 项，401 项通过，4 项目标 CUDA 用例按开关跳过。2026-11-05 复跑时，进程在 `OK (skipped=4)` 后出现 Windows 原生访问冲突 `0xC0000005`。启用 `faulthandler` 后记录为 `<no Python frame>`，具体组件待定位。记录全套结果时一并检查进程退出码。
- **目标 GPU 待验收。** STFT、CWT、单指数和双指数寿命热图已有 CUDA 路径。1.1.1 新增的 CWT 与寿命计算仍需在目标 NVIDIA 机器上做数值对照、真实数据和性能验收。配置匹配的 CuPy/CUDA 后，设置 `LIFECALOR_RUN_CUDA_TESTS=1` 运行四项专用测试，再按阶段 F 填写设备、结果与实测记录。[CUDA 数值测试](tests/test_compute_111_cuda_prototypes.py) · [阶段 F 验收表](docs/testing/lifecalor-1.1.1-stage-f-gpu-report.md)
- **打包。** PyInstaller 完整版使用 `core/LifeCalor.spec`，构建环境需提供 CuPy。CPU 精简版使用 `core/LifeCalor-CPU.spec`。两者均以 `core/MainWindow.py` 为入口。阶段 F 还要求在目标环境启动两个 EXE，检查资源、界面和计算后端。源码运行日志在 `core/carrier_lifetime.log`，打包版日志在 `%LOCALAPPDATA%\LifeCalor\carrier_lifetime.log`。[完整版配置](core/LifeCalor.spec) · [CPU 配置](core/LifeCalor-CPU.spec) · [构建说明](docs/packaging/lifecalor-compute-runtime.md)
- **后续版本线索。** 1.1.1 已有固定卷积的 CPU 分块实现和 CUDA 原型。上轮计划将自定义时空卷积的完整三维编辑工具列入 1.1.2。涉及该方向时，从现有卷积管线、边界处理及其回归开始核对需求。[卷积实现](core/compute/algorithms/convolution_pipeline.py) · [卷积回归](tests/test_compute_111_convolution.py) · [1.1.1 设计与实施记录](docs/superpowers/plans/2026-09-21-lifecalor-111-algorithms-and-options.md)

形成新版本时，将用户可见变化记入 `updatinglog.md`，将实机结果填入对应的 `docs/testing/` 报告，再更新本文的源码基线、已验证环境和剩余验收项。

2026-11-06 已完成全程序计算算法盘点与“分析与处理”菜单集成规划，包含八类能力、现有入口、参数与数据源联动、接入前修复项及验收条件。现有菜单“高级操作”集中坏帧处理、数据计算器和切片器，“分析与处理”集中科学分析任务。后续实现从该规划核对范围。[算法分类与菜单规划](docs/superpowers/plans/2026-11-06-lifecalor-analysis-menu.md)

2026-11-06 首期分析菜单已实施。高级操作集中坏帧处理、计算器和切片器，分析与处理复用共享输入确认、独立参数导航及原有任务管线。科学曲线带时间单位进入 PSD，双指数热图与区域拟合各有明确的模型控件。[菜单接入验收](docs/testing/lifecalor-analysis-menu.md)

本轮新增 60 项专项测试通过，最终全套 465 项中 461 项通过、4 项目标 CUDA 用例跳过，进程退出码为 0。早期复跑的 Windows 原生退出异常作为间歇性问题记录于验收文档。
