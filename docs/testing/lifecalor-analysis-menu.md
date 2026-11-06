# 分析与处理菜单首期验收

日期为 2026-11-06，源码基线为 `a3bf420 / 1.1.1`，验收对象为当前工作树的未发布改动。

## 实施范围

- 新增“分析与处理”菜单，包含 18 个图像/序列分析动作及共享的曲线分析动作。
- “高级操作”包含坏点处理、数据计算器和数据切片器，两类菜单按用途分工。
- 菜单与左面板共享输入确认与提交，确认实际 Data/ProcessedData 身份和匹配 ROI，参数导航独立于导入下拉框。
- 区域与像素寿命各有明确模型控件，传热输入提示主寿命字段。扩散及心肌的延迟提交保留动作 ID 与输入源摘要。
- 修复坏帧异步返回及新结果发布、ROI 统计和整型精度、时间平均图、HW/1HW/THW FFT 往返、扩散有效帧、光流输入与取消检查。
- 曲线模式与峰工具同步，PSD 校验采样及输入语义，峰面积使用当前 NumPy 支持的接口。
- 曲线输入有效性按数据快照缓存，重复展开菜单只更新状态与提示，实际执行时重新校验数据。
- 科学时序通过共享 helper 传递时间轴与单位，标记已物理化的轴。ROI 统计及全场均值到 PSD 的 ps 单位链路增加端到端回归。

## 环境与命令

测试源代码来自当前工作树。解释器复用主仓库的 `D:/coding/L/LifeCalor/.venv/Scripts/python.exe`，Python 3.12.14，主要版本记录在 [requirements.txt](../../requirements.txt)。独立配置环境后，在仓库根目录可用以下命令复现。

```powershell
.\.venv\Scripts\python.exe -B -m unittest tests.test_analysis_menu tests.test_analysis_worker_numerics tests.test_bad_frame_dialog tests.test_curve_analysis_actions
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
$LASTEXITCODE
```

## 结果

| 验证项 | 记录 |
| --- | --- |
| 菜单、输入、上下文及延迟提交 | 13 项通过 |
| 工作对象与科学时轴 | 27 项通过 |
| 坏帧异步对话框 | 8 项通过 |
| 曲线分析、菜单刷新缓存及 ps 单位端到端 PSD | 12 项通过 |
| 上述专项合计 | 60 项通过，独立进程退出码 0 |
| 实际主窗口 | 启动、菜单展开渲染、参数导航、导入选择保持及独立模型控件 smoke 通过，独立进程退出码 0 |
| 完整自动化测试 | 465 项，461 项通过、4 项目标 CUDA 测试跳过，断言汇总为 `OK (skipped=4)` |
| 最终完整测试进程结束 | 退出码 0，正常结束 |

ps 端到端用例输入为 128 点、16 Hz 正弦，时间坐标以 ps 表示。分别经过 ROI 统计和全场均值的实际结果路由后，Welch PSD 的峰频率均为 16 Hz。

## 剩余验收

本轮早期的完整回归曾在汇总后复现既有 Windows 原生访问冲突，faulthandler 显示 `<no Python frame>`，相关进程超时后停止。最终 465 项回归正常退出。该间歇性退出问题保留历史记录并继续追踪，新增专项与主窗口 smoke 的正常退出记录列于上表。

目标 NVIDIA 设备的真实 CUDA 数值、性能及打包验收继续按 [阶段 F](lifecalor-1.1.1-stage-f-gpu-report.md) 执行。本轮验证了 CPU 工作流和可用入口，四项实机 CUDA 用例按环境开关跳过。

本轮改动登记在 [更新日志](../../updatinglog.md) 的开发中条目，能力边界及后续候选见 [菜单规划](../superpowers/plans/2026-11-06-lifecalor-analysis-menu.md)。
