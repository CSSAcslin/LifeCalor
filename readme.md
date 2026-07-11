# Carrier Lifetime Calculator / LifeCalor

LifeCalor 是一个基于 PyQt5 的成像数据分析工具箱，核心代码位于 `core/`。当前主入口是 `core/MainWindow.py`，主要支持 TIFF/SIF/AVI 导入、ROI 交互、载流子寿命分析、EM-iSCAT 时频分析、结果绘图与导出。

## 运行入口

开发环境中从项目根目录运行：

```powershell
python core/MainWindow.py
```

程序启动时会读取 `core/style.qss`，并依赖 `core/resources_rc.py` 中编译好的 Qt 资源。

## GitHub 更新检查 Token

不要在源码中写入 GitHub PAT。若需要访问 GitHub release API，请在本机环境变量中设置：

```powershell
$env:LIFECALOR_GITHUB_TOKEN = "your-token"
```

代码会自动转换为 `Bearer <token>` 请求头；未设置时更新检查会以无认证方式请求。

## 核心目录

- `core/MainWindow.py`：PyQt5 主窗口入口和当前主要流程编排。
- `core/DataManager.py`：数据模型、图像转换、ROI 处理、导出。
- `core/ImportManager.py`：TIFF/SIF/AVI 数据导入。
- `core/DataProcessor.py`：EM-iSCAT、STFT/CWT、单通道、傅里叶等处理。
- `core/LifetimeCalculator.py`：寿命拟合、寿命热图、扩散、传热计算。
- `core/ImageDisplayWindow.py`：多画布、ROI、时间轴、图像交互。
- `core/ResultDisplayWidget.py`：结果图表展示。

## 打包入口

现有打包配置位于：

- `LifeCalor.spec`
- `core/LifeCalor.spec`
- `core/MainWindow.spec`

`build/`、`dist/`、`core/build/`、`core/dist/` 是生成产物，不应作为源码维护。

## 测试

当前地基测试使用标准库 `unittest`，无需额外安装 pytest：

```powershell
python -m unittest discover -s tests -v
```

## 维护与扩展地基

本仓库已开始把主窗口中的长期职责逐步抽出，当前新增的维护入口包括：

- `core/settings/parameter_store.py`：统一参数读取与类型转换，供 `MainWindow` 的 `QSettings` 参数组使用。
- `core/TaskState.py`：统一任务状态模型，状态值包括 `idle/running/cancelling/failed/completed`，后续导入、计算、EM 处理、导出线程都应逐步接入。
- `core/ProcessingBenchmark.py`：小型性能测量 helper，用于记录处理操作名称、耗时和输出 shape；后续 STFT/CWT 专项 benchmark 可直接复用。
- `core/AlgorithmRegistry.py`：新算法扩展注册表，新算法推荐以 `name + handler + description` 形式注册，handler 输入数据并返回处理结果，再由现有流程包装为 `ProcessedData`。
- `core/ThreadController.py`：线程活动检测和停止动作封装，`MainWindow` 通过它停止计算/EM 处理线程，并同步任务状态。
- `core/exporting/`：导出控制、保存流程与导出策略。
- `core/selection/`：数据焦点与 ROI 选择策略。
- `core/tasks/`：可扩展的多任务注册表、独立任务状态和非阻塞取消令牌。
- `core/dataio/`：NPY 与缓存数组的分块读写、连续进度和取消清理。
- `core/TaskController.py`：线程启动入口与任务状态 `running` 同步。

新增功能建议遵循：参数收集 -> 任务执行 -> `ProcessedData` -> 绘图/画布/导出。

## 大数组缓存与任务管理

LifeCalor 使用 NumPy `.npy` 作为大数组缓存格式。默认阈值为 512 MB，实际大小按 `ndarray.nbytes` 判断，因此 `uint8`、`float32`、`complex64` 等不同 dtype 都按真实字节数处理。缓存目录和阈值可以在“历史与缓存管理”中设置。

缓存读取和写入采用分块 I/O，能够显示连续进度，并通过统一任务取消令牌响应 Esc。写入先生成临时文件，成功后再原子替换目标文件；取消或失败不会留下可恢复索引或半成品。

`Data` / `ProcessedData` 保留完整 metadata，大数组历史使用磁盘引用。显示层按帧读取和渲染，不保存整段伪彩 RGBA 副本；画布样式导出会逐帧复用同一渲染器。

任务注册表可以同时记录多个独立任务。当前状态栏显示最近的前台任务，后续可在此基础上增加并发任务列表与逐任务进度面板。
