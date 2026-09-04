# LifeCalor 下一阶段架构与交互优化规划

## 执行状态（2026-09-04）

- [x] 阶段 0：基线已固化为 `4331dec`，版本 `1.0.9`，198 项测试通过。
- [x] 阶段 1：统一错误总线、单弹窗队列、错误去重及后台任务错误接入完成。
- [x] 阶段 2：metadata-only 数据分类器和共享数据树“数据类型”列完成。
- [x] 阶段 3：画布右键数据详情与非模态详情窗口完成。
- [ ] 阶段 4-6：任务模型统一、导入扩展、显示封口与性能专项待后续执行。

## 1. 文档目标

本规划用于指导 LifeCalor 在多数据计算器基本可用之后的下一阶段开发。重点解决三个新需求：

1. 在画布区域右键查看当前画布及其来源数据的详细参数。
2. 为数据历史树中的所有数据和参数增加统一的数据形态分类。
3. 彻底解决未捕获异常和多行 traceback 被逐行弹窗的问题，并继续完成全局错误处理统一。

同时承接此前尚未完成的任务系统统一、MainWindow 高收益拆分、显示层封口、导入格式扩展和性能验证。计算器本轮不再重构，只保留回归测试和必要修复。

## 2. 当前基线

### 2.1 已完成并应保护的能力

- 大数组按可配置阈值落盘，默认阈值为 512 MB。
- `Data.data_origin`、`ProcessedData.data_processed`、`out_processed` 支持 `ArrayRef`。
- 缓存目录配置、手动清理、孤立缓存处理、强制落盘和跨启动历史恢复。
- 缓存读写、通用 NPY/TIFF 导入进入后台任务并支持连续进度。
- 显示层已有 `source / renderer / worker / service / render_controller / frame cache`。
- 大数据换帧、再次导入、伪彩显示和样式导出已经稳定。
- 数据树已抽为 `DataHistoryTreeWidget`，计算器和历史查看可复用。
- 多数据计算器支持预切片、广播、Shape 路径验证、后台执行和结果预览。
- 统一诊断已有 `AppError`、日志分级和基础弹窗去重。
- 当前自动化测试基线为 198 项通过。

### 2.2 当前未固化状态

- 阶段 0 基线提交为 `4331dec feat: stabilize multi-data calculator for 1.0.9`。
- 计算器验证一致性、任务重复提交防护、超大预览降采样、时间轴修复和代码清扫仍在工作区。
- 阶段 1-3 实现版本为 `1.0.10`；后续阶段继续按独立版本和提交推进。
- 提交前必须检查暂存区，不能把实验数据、临时脚本、打包产物或用户无关文件混入提交。

## 3. 总体原则

1. **科学数据只读优先**：详情查看和分类不得修改源对象，不得触发大数组整体加载。
2. **元数据优先**：Shape、dtype、axes、缓存状态应从对象字段或 `ArrayRef` 描述读取。
3. **一次错误一次事件**：日志可以多行，但同一根异常最多产生一个用户弹窗。
4. **日志与弹窗解耦**：`logging.error()` 只负责记录，不能天然等同于弹窗请求。
5. **兼容渐进迁移**：保留短期兼容入口，但所有新代码只使用新的分类器和错误总线。
6. **不破坏计算器**：数据树改动必须保持计算器选取 Data、ProcessedData 和 `out_processed` 的行为。
7. **每阶段可验证、可提交、可回滚**：不要把错误系统、树结构、任务并发混成一个无法定位的巨型提交。

## 4. 目标架构

```text
Data / ProcessedData / ArrayRef / parameter
                    |
                    v
          DataDescriptor + DataClassifier
             |                    |
             v                    v
   DataHistoryTreeWidget    CanvasDataDetailsDialog

worker / Qt slot / sys.excepthook / explicit validation
                    |
                    v
              ErrorBroker
       normalize -> deduplicate -> log once
                    |
          +---------+----------+
          |                    |
          v                    v
     status/task state     ErrorPresenter
                           single popup queue
```

## 5. 阶段 0：冻结并固化当前基线

### 目标

先把已经验证过的计算器及近期修复形成可靠基线，避免后续树和错误系统改造覆盖尚未提交的工作。

### 工作项

- 检查当前工作区差异，只保留项目相关源码和测试。
- 运行完整单元测试并记录基线数量。
- 由用户使用真实数据完成一次计算器烟测：
  - 普通 Data 与恢复的 ProcessedData 混合运算。
  - 选择 `out_processed` 数组。
  - `THW - HW`、指定帧切片、广播和 transpose。
  - 结果继续显示、裁剪、缓存、恢复和导出。
- 将当前计算器后续修复作为独立提交固化。
- 不修改版本号，除非用户明确要求。

### 验收标准

- 完整测试通过。
- 打开、关闭、重新打开计算器不会产生任务串位。
- 当前提交不包含数据文件、日志、打包目录和实验脚本。

## 6. 阶段 1：重建错误呈现链路

该阶段优先级最高，因为当前逐行弹窗会严重阻断用户操作。

### 6.1 已确认根因

当前错误链包含两条会相互叠加的路径：

1. `sys.stderr` 被 `StreamLogger(logging.ERROR)` 接管。traceback 写入 stderr 时通常按多行或多次 `write()` 输出。
2. 每条 ERROR 日志进入 `ConsoleHandler.emit()`，随后通过 `legacy_error` 调用 `MainWindow.handle_logged_error()`，每一行都会构造一次 `AppError` 并尝试弹窗。
3. 同一未捕获异常还会进入 `sys.excepthook`，再由 `handle_unhandled_exception()` 上报一次完整错误。
4. 现有 `_RECENT_POPUPS` 只对完全相同的标题、消息和阶段做短时间去重；traceback 每行文本不同，因此无法合并。

结论：不能仅延长去重时间，必须取消“任意 ERROR 日志自动转成弹窗”的设计。

### 6.2 新增错误模型

建议在 `core/diagnostics/` 中形成以下模块：

- `model.py`
  - `AppError` 或重命名后的 `ErrorEvent`。
  - 字段至少包含：`error_id`、`title`、`message`、`stage`、`severity`、`details`、`original`、`context`、`task_id`、`timestamp`、`popup_policy`。
- `broker.py`
  - 单一错误入口 `ErrorBroker.report(event)`。
  - 负责标准化、指纹生成、去重、日志一次写入和任务失败联动。
- `presenter.py`
  - 管理唯一弹窗队列。
  - 任意时刻最多显示一个错误弹窗。
  - 相同指纹错误在合并窗口内累计次数，不重复弹窗。
- `reporter.py`
  - 保留 `report_warning/report_error/report_exception/show_app_error` 兼容函数。
  - 内部全部转发给 `ErrorBroker`，不能各自再次写日志或直接创建多个 `QMessageBox`。

### 6.3 明确日志和 UI 规则

- `info`：只写日志，可更新状态栏，不弹窗。
- `warning`：写日志并更新状态栏，不弹窗。
- `error`、`critical`：写日志，由明确的结构化错误事件请求弹窗。
- 单纯调用 `logging.error()` 不产生弹窗。
- 未捕获异常产生一个 `critical ErrorEvent`，消息只显示异常类型和简述，完整 traceback 只进入详情和日志。
- worker 中已经记录过的异常不得在 MainWindow 再记录第二次。
- 每个错误事件生成稳定 `error_id`，弹窗和日志都显示该 ID，便于用户反馈。
- 同一错误短时间重复触发时：
  - 日志保留首次完整 traceback。
  - 后续只记录“重复 N 次”的汇总。
  - UI 不重复弹窗。

### 6.4 修复 stderr 和未捕获异常入口

- 修改 `ConsoleUtils.StreamLogger`：
  - 对 stderr 内容进行缓冲，不把 traceback 每一行独立升级成 UI 错误。
  - 写入日志的 record 标记为 `ui_silent=True` 或 `lifecalor_user_reported=True`。
- 修改 `ConsoleHandler.emit()`：
  - 删除或停用 `ERROR -> legacy_error` 的自动桥接。
  - ConsoleHandler 只负责控制台展示，不负责决定是否弹窗。
- `sys.excepthook` 只构造一个完整 `ErrorEvent`。
- 增加 `threading.excepthook`，捕获纯 Python 后台线程未处理异常。
- Qt worker 和多进程任务继续通过失败信号返回结构化错误，不依赖全局 hook 猜测。
- 弹窗关闭过程中发生的新错误进入队列，不允许递归弹窗。

### 6.5 迁移旧错误路径

优先处理 `DataProcessor.py` 中仍然存在的：

```python
processed_result.emit({"type": "...", "error": str(exc)})
```

迁移规则：

- 成功信号只承载成功结果。
- 失败使用专用 `failed(AppError)` 信号或统一任务失败信号。
- `MainWindow.processed_result()` 不再判断错误字典。
- 导入、缓存、显示、计算、导出逐步接入同一错误总线。
- 每个错误包含阶段、数据名、Shape、dtype、来源格式和 task ID；获取这些字段时不能解析大数组。

### 6.6 错误弹窗 UI

- 默认只显示：标题、简短原因、错误 ID。
- 提供“查看详情”折叠区域，展示阶段、数据上下文和 traceback。
- 提供“复制详情”和“打开日志目录”按钮。
- 不把 traceback 整段塞进主消息文本。
- 连续错误采用队列或“还有 N 个错误”提示，不堆叠多个模态窗口。

### 6.7 自动测试

新增或扩展：

- `tests/test_error_broker.py`
- `tests/test_error_reporting.py`
- `tests/test_error_severity.py`
- `tests/test_console_error_bridge.py`

必须覆盖：

- 一段 20 行 traceback 只产生一个弹窗事件。
- stderr 分多次 `write()` 输出仍只形成一个未捕获异常事件。
- 相同异常连续触发 100 次只弹一次，并记录重复次数。
- 两个不同错误按顺序显示，不叠加窗口。
- warning 不弹窗。
- `logging.error()` 被记录但不自动弹窗。
- 显式 `report_error()` 恰好弹一次。
- worker 失败后任务状态、进度条和按钮恢复。

### 6.8 验收标准

- 任意一次未捕获异常最多弹出一个错误窗口。
- 不再出现需要逐个关闭 traceback 每一行的情况。
- 日志仍保存完整 traceback，且不丢失 info/warning/error。
- 同一错误不会被 worker、日志桥、MainWindow 和全局 hook 重复记录。

## 7. 阶段 2：统一数据形态分类

### 7.1 新建纯逻辑分类模块

建议创建 `core/dataio/classification.py`，不把规则写死在 Qt 控件中。

建议数据结构：

```python
class DataCategory(Enum):
    SCALAR = "标量"
    VECTOR = "向量"
    LINEAR = "线性"
    IMAGE = "图片"
    VIDEO = "视频"
    MATRIX_3D = "三维矩阵"
    HIGH_DIMENSIONAL = "高维数据"
    STRUCTURED = "结构化参数"
    UNKNOWN = "未知"

@dataclass(frozen=True)
class DataDescriptor:
    category: DataCategory
    shape: tuple[int, ...]
    dtype: str
    axes: str
    reason: str
    inferred: bool = False
```

分类函数只接收 shape、dtype、axes 和少量语义提示，不接收或遍历完整大数组。

### 7.2 分类规则

按以下顺序判断，保证类别互斥：

1. **标量**
   - Python/NumPy 数值、布尔、字符串或 0 维数组。
2. **线性**
   - 一维数据并且存在与长度匹配的独立坐标轴，如 `time_point`。
   - 或 metadata 明确标记为线性曲线/频数曲线。
   - 典型用途为可以直接进入 PlotGraph 的序列。
3. **向量**
   - 其他一维 ndarray、ArrayRef、list 或 tuple。
   - 表示参数向量、频率轴、坐标向量等，不擅自判断为时间曲线。
4. **图片**
   - 二维矩阵且两个维度都大于 1。
   - `(1, N)`、`(N, 1)` 默认按线性处理。
5. **视频**
   - 三维矩阵，并且标准化 axes 中恰好包含一个 `T` 轴。
   - T 不要求位于第 0 轴；例如 `THW`、`HTW` 都属于视频。
6. **三维矩阵**
   - 三维矩阵，但 axes 中没有 T。
7. **高维数据**
   - 四维及以上数组，暂不强行归入视频。
8. **结构化参数**
   - dict、dataclass 或无法安全扁平化的复合对象。
9. **未知**
   - 缺少足够元数据且无法安全判断。

### 7.3 axes 获取和推断规则

axes 的读取优先级：

1. 当前字段自己的 descriptor/metadata。
2. `scientific_axes`。
3. `source_axes`。
4. `display_axes`。
5. Data/ProcessedData 自身的 `time_point` 与 Shape 匹配关系。

若三维对象没有 axes，但 `time_point.size` 只与其中一个轴长度匹配，可以分类为“视频（推断）”，同时将 `inferred=True`，Tooltip 必须说明推断依据。若存在多个等长轴导致歧义，则归为“三维矩阵”，不能静默猜测 T 轴。

兼容旧 metadata 中的 `Y/X` 命名，但 UI 统一显示 `H/W`。

### 7.4 数据树改造

修改 `core/widget/DataTreeWidget.py`：

- 当前“类型”列改名为“来源/处理类型”，继续显示：原始格式、处理方法、ndarray、ArrayRef 等。
- 新增“数据形态”列，显示：标量、向量、线性、图片、视频、三维矩阵等。
- 建议列顺序：
  1. 名称 / Key
  2. 来源 / 处理类型
  3. 数据形态
  4. Shape & 大小
  5. 数值范围
  6. 创建时间 / 值
  7. 操作
- `DataTreeEntry` 增加 `descriptor` 或 `category/axes` 字段。
- Data、ProcessedData、`out_processed`、parameters 和嵌套参数全部调用同一分类器。
- 数组范围继续只使用对象已有 `datamin/datamax`；不能为了树显示调用 `min/max` 扫描 ndarray 或 mmap。
- ArrayRef 只读取 `shape/dtype/nbytes` 描述，不调用 `load()`。
- 大 list/dict 只显示项目数量；嵌套参数采用延迟展开或限制初始深度，避免生成窗口时卡顿。
- 所有截断文本必须有完整 Tooltip。

### 7.5 复用分类结果

- `ExtraDialog.DataTreeViewDialog` 的“线性绘图/图像显示/直接导出”按钮由分类结果决定，不再重复写 ndim 判断。
- 计算器 source picker 继续复用同一树，不改变可选数据规则。
- 后续导入预检、缓存管理和画布详情也使用同一 descriptor。
- 不允许不同窗口各自维护一套 `ndim == ...` 分类规则。

### 7.6 自动测试

新增：

- `tests/test_data_classification.py`
- `tests/test_data_tree_classification.py`

覆盖：

- Python 标量和 0D ndarray。
- 一维普通向量。
- 有匹配 time_point 的一维线性数据。
- `(1, N)`、`(N, 1)`。
- 普通 HW 图片。
- THW、HTW 和 WHT 视频。
- 无 T axes 的三维矩阵。
- axes 缺失但 time_point 唯一匹配的推断视频。
- axes 歧义时不误判视频。
- 4D 数据和嵌套 dict。
- ArrayRef 分类期间不调用 `load()`。
- 大数组分类期间不调用 `min/max/np.asarray()`。

### 7.7 验收标准

- 所有树节点的数据形态列都有明确内容。
- 同一对象在历史树、计算器选择器和画布详情中的分类一致。
- 打开包含大量恢复缓存数据的树不会加载缓存数组。
- 分类规则有 Tooltip 解释，推断结果明确标注。

## 8. 阶段 3：画布右键数据详情

### 8.1 交互入口

修改 `SubImageDisplayWidget.show_context_menu()`，在现有“同步播放”和“导出画布”之外新增：

- `查看数据详情...`

建议放在同步播放后、导出前，并用分隔线区分查看与操作命令。

### 8.2 详情窗口

建议新增 `core/display/data_details.py` 或 `core/widget/DataDetailsDialog.py`。

窗口应为只读、非模态窗口，同一画布只保留一个实例；重复点击时刷新、置顶并激活，不叠加多个窗口。

推荐布局：

- 顶部摘要：数据名、数据形态、Shape、dtype、axes、缓存状态。
- “数据来源”页：
  - Data/ProcessedData。
  - 导入格式或处理类型。
  - timestamp、timestamp_inherited、serial number。
  - 来源名称和处理链 provenance。
- “参数”页：
  - `parameters`。
  - `out_processed` 中非大数组参数。
  - 数组字段只显示 descriptor，不加载内容。
- “画布状态”页：
  - canvas ID、当前帧、总帧数、当前真实时间值。
  - 当前伪彩名称、是否启用、显示上下限。
  - ROI/anchor/vector 是否存在及摘要。
  - 当前渲染状态。
- 底部命令：复制摘要、复制完整参数、刷新、关闭。

### 8.3 数据来源与生命周期

- 优先通过 `ImagingData.parent_data` 弱引用取得原始 Data/ProcessedData。
- 弱引用已失效时仍必须显示 ImagingData 在创建时保存的来源摘要，不能报错。
- 建议在 `ImagingData.create_image()` 时生成小型不可变 `source_descriptor`，只保存名称、类型、Shape、dtype、axes、时间戳和 metadata 摘要，不保存数组副本。
- 详情窗口不能访问 `image_backup` 全量内容，也不能重新计算全局 min/max。
- 当前帧值、时间和伪彩状态从 canvas 当前状态读取。

### 8.4 参数展示约束

- 标量直接显示值。
- 向量、线性、图片、视频和三维矩阵显示分类、Shape、dtype、axes、内存/缓存大小。
- ArrayRef 可显示缓存文件名和状态，但默认不展示完整绝对路径；“打开缓存目录”沿用缓存管理入口。
- 超长字符串截断并提供 Tooltip/复制全文。
- dict 使用树结构并限制自动展开深度。
- 参数读取失败应在窗口内显示“无法读取该字段”，记录 warning，不弹 error 窗口。

### 8.5 自动测试

新增 `tests/test_canvas_data_details.py`，覆盖：

- Data 和 ProcessedData 画布详情。
- out_processed 来源画布。
- 二维图片、视频和非时间三维矩阵分类。
- 恢复缓存 ArrayRef 不被加载。
- parent_data 弱引用失效后仍可查看摘要。
- 伪彩、当前帧和时间值正确刷新。
- 重复打开同一画布详情不会创建多个窗口。
- 关闭画布时详情窗口安全关闭，不残留 Qt 对象。

### 8.6 手工验收

- 右键任意画布，详情动作可见且不会使伪彩消失。
- 播放视频时打开详情，点击刷新可看到当前帧变化。
- 大数据画布打开详情无明显停顿，内存不出现大幅增长。
- 删除画布后详情窗口不崩溃。

## 9. 阶段 4：任务和线程模型继续统一

错误系统完成后再进行，错误事件应携带 task ID。

### 工作项

- 以 `TaskCoordinator/TaskRegistry` 为唯一任务状态来源，逐步移除旧 `TaskState`。
- 将寿命计算、STFT/CWT、ROI、傅里叶等耗时流程全部注册为独立任务。
- Esc 只调用 cancellation token，不在 GUI 线程执行 `quit/wait`。
- 多进程算法在分块或行循环中检查取消标志，并可靠关闭 Pool/shared memory。
- 明确哪些 QThread 常驻、哪些按任务创建；统一 finished/failed/cancelled 清理模板。
- 计算器暂时保持单任务串行，后续任务面板稳定后再考虑并发。
- 建立多任务面板：任务名、来源数据、进度、状态、耗时、取消按钮、错误详情。
- 只有资源互不冲突的任务允许并行；同一数据写任务和同一 canvas 渲染更新需要串行化。

### 验收标准

- 任何任务失败都能由 task ID 找到对应日志和错误。
- Esc 不冻结 GUI。
- 任务取消后无线程、进程、临时文件或共享内存残留。
- 多个任务的进度互不覆盖。

## 10. 阶段 5：MainWindow 和结果路由高收益拆分

只做收益明确的拆分，不追求机械减少行数。

### 工作项

- 将任务提交、完成、失败和取消编排抽为 `ProcessingController`。
- 将 `processed_result()` 的大型 match 分发改成结果处理注册表。
- 将 DataProcessor 旧错误字典全部移除后，删除对应兼容分支。
- 将 ExtraDialog 中缓存、导出、参数编辑、数据查看对话框逐类迁出。
- 逐步替换 MainWindow 的星号导入，明确依赖来源。
- 清理已确认无引用的注释实现、空方法和重复 ndim 判断。

### 验收标准

- MainWindow 只保留 UI 装配、顶层信号和少量应用级协调。
- 新算法不需要修改 MainWindow 大型 match 才能接入。
- 每次拆分都有现有流程回归测试。

## 11. 阶段 6：导入扩展与性能验证

### 导入扩展

- 将旧 AVI/SIF 逐步迁入 `ImporterRegistry` 数据契约。
- 增加 OME-TIFF，读取并保留可靠 axes 和物理单位。
- 增加 HDF5/H5 数据集选择器，读取前显示 Shape、dtype 和预计大小。
- 支持多页彩色 TIFF 和明确的四维数据，不静默猜轴。
- 批量导入时显示每个文件的独立任务状态。

### 性能验证

- 建立固定的小、中、大数据 benchmark。
- 记录导入、缓存、历史恢复、首帧显示、换帧、STFT/CWT 和导出的耗时及峰值内存。
- 增加全局内存预算，避免多个各自低于 512 MB 的数组同时挤满内存。
- 审查算法是否因 `np.asarray`、`copy`、切片连续化等操作意外整体加载 ArrayRef。
- 完成打包后 EXE 烟测。

## 12. 推荐提交顺序

1. `chore: seal calculator stabilization baseline`
2. `refactor: centralize error reporting and popup queue`
3. `refactor: migrate processing failures to structured errors`
4. `feat: add metadata-only data classification`
5. `feat: show data categories in shared history tree`
6. `feat: add canvas data details dialog`
7. `refactor: unify task state and result routing`
8. 后续导入格式和多任务面板分别提交。

每次提交只暂存明确文件。不得使用 `git add .`。

## 13. 完整回归清单

### 自动测试

- 完整 unittest 测试集全部通过。
- 新增错误队列、数据分类、树分类和画布详情测试。
- `python -m compileall -q core` 通过。
- `git diff --check` 通过。

### 真实数据烟测

- 导入小型 AVI、NPY、灰度 TIFF、Palette TIFF。
- 导入并恢复超过缓存阈值的大数据。
- 首帧显示、连续换帧、播放、伪彩、ROI 和导出。
- 右键查看 Data、ProcessedData、恢复数据和 out_processed 画布详情。
- 在历史树中检查每种数据形态及 Tooltip。
- 人为触发一个普通错误和一个未捕获异常，确认每个根异常只出现一个弹窗。
- 连续触发相同错误，确认不会形成弹窗循环。
- 取消导入、缓存读取和计算，确认 GUI 不冻结。

## 14. 本轮明确不做

- 不重写已经基本可用的多数据计算器。
- 不为了分类或详情窗口加载大数组。
- 不在错误系统稳定前直接开放任意任务并行。
- 不把 warning 改成弹窗。
- 不进行全量 UI 视觉重设计。
- 不自动更新版本号或更新日志版本标题。

## 15. 完成定义

本规划阶段 1-3 完成时，必须满足：

1. 画布右键可以稳定查看当前画布及源数据完整参数摘要。
2. 所有共享数据树统一展示数据形态，且不因缓存数据产生阻塞。
3. 一个根异常只产生一个可管理的错误弹窗，traceback 不再逐行弹出。
4. warning 不弹窗，error/critical 可追踪到日志、数据上下文和任务 ID。
5. 计算器、缓存恢复、显示和导出没有行为回归。

阶段 4-6 属于后续架构完善，不应阻塞前三项新需求交付，但应沿用同一数据 descriptor、ErrorBroker 和 TaskCoordinator 契约。
