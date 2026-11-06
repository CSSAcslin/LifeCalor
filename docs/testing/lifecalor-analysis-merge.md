# 分析菜单与 1.1.2 合并验收

日期为 2026-11-06。应用源码版本保持 `1.1.2`，本次集成的两条基线如下。

| 基线 | 提交 | 已有历史验收 |
| --- | --- | --- |
| 主线 1.1.2：稳定性、GPU 与时空卷积 | `1798dde` | [1.1.2 验收记录](lifecalor-1.1.2-acceptance.md) |
| 首期分析菜单与算法修复 | `72e74ba` | [菜单接入验收](lifecalor-analysis-menu.md) |

两份历史报告中的 437 项和 465 项分别来自各自的源码基线，合并后的结果在本文独立记录。

后续正式发布包含收尾修复和重新构建的两种 EXE，最终 518 项回归及产物验收见 [1.1.2 正式发布验收](lifecalor-1.1.2-release.md)。本文的 497 项统计和合并前 EXE 状态保留为历史记录。

## 集成范围

- 菜单顺序保持“选项 → 控制台 → 分析与处理 → 高级操作 → 历史数据 → 使用指南 → 检查更新”。
- 分析菜单及左面板复用输入确认、匹配 ROI、参数导航和原有任务提交。
- 高级操作保留坏点处理、数据计算器、数据切片器和时空卷积四个入口。时空卷积沿用 1.1.2 的原有入口，分析菜单不增加重复动作。
- 保留主线退出控制器、卷积任务和结果路由、CUDA 与 STFT 数值修复，并保留分析菜单分支的算法及时间轴修复。
- 固定运行依赖沿用主线 1.1.2，CUDA 与构建清单继续保留。

## 合并后验证

解释器复用主仓库 `.venv`（Python 3.12.14），测试源码来自当前集成工作树，GPU 为 RTX 5070 Ti。

| 验证项 | 结果 |
| --- | --- |
| 菜单与输入确认、坏帧窗口、曲线分析及工作对象数值专项 | 60 项通过，进程退出码 0 |
| 时空卷积、任务与结果路由、退出生命周期回归 | 已包含在完整回归中，通过 |
| 默认模式完整回归 | 497 项：488 项通过，9 项目标 GPU 用例跳过；`OK (skipped=9)`，9.752 秒，进程退出码 0 |
| 目标 GPU 完整回归 | `LIFECALOR_RUN_CUDA_TESTS=1`，497 项全部通过，0 跳过；25.892 秒，进程退出码 0 |
| 实际主窗口 | 启动、菜单顺序、高级操作四项、独立寿命模型控件、卷积窗口打开/关闭及主窗正常关闭均通过，进程退出码 0 |
| 依赖一致性 | `pip check` 通过 |
| Git 冲突标记、空白检查 | 通过 |

可在主仓库根目录复现：

```powershell
Remove-Item Env:LIFECALOR_RUN_CUDA_TESTS -ErrorAction SilentlyContinue
.\.venv\Scripts\python.exe -B -X faulthandler -m unittest discover -s tests -v
$LASTEXITCODE
$env:LIFECALOR_RUN_CUDA_TESTS = '1'
$env:CUPY_CACHE_DIR = Join-Path $env:TEMP 'lifecalor-merge-cupy-cache'
.\.venv\Scripts\python.exe -B -X faulthandler -m unittest discover -s tests -v
$LASTEXITCODE
```

最终日志位于 `%TEMP%\lifecalor-merge-final-cpu.log` 和 `%TEMP%\lifecalor-merge-final-cuda.log`。诊断过程曾有一次画布用例收尾停顿，手动中止后未在诊断及最终回归中复现，原因尚未确认。另一次 GPU 回归由 120 秒诊断定时器终止，堆栈停在 CuPy 写默认编译缓存的 `tempfile._mkstemp_inner`；显式指定上述可写临时缓存目录后，全套正常完成。以上中间运行不计作最终通过结果。

## 打包与实机记录

[1.1.2 EXE 原始记录](package-2026-11-06.json)对应合并分析菜单之前的产物，不能作为本次合并后 EXE 的验收。已有 EXE 尚不包含新菜单，本次源码合并不更新这些历史二进制记录。重新打包后需另行记录产物 SHA256、菜单界面、包内计算和退出结果。

RTX 5070 Ti 的既有数值、small/medium 性能及真实 AVI/TIFF 验收保留于[目标机报告](2026-11-06-rtx5070ti-validation.md)。带明确时间轴的真实寿命衰减序列、完整 CUDA 卷积工作流和更大规模的性能实测仍是后续验收项。
