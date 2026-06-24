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
