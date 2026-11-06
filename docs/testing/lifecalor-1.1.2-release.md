# LifeCalor 1.1.2 正式发布验收

日期为 2026-11-06。应用版本 `1.1.2`，发布标签 `v1.1.2`。最终测试及 EXE 构建的源码基线为 `6652905f33840794b30997ed5935dd2b3cc0c9de`，其后的发布整理仅更新文档及验收记录。

本版合并主线的稳定性、GPU 和时空卷积更新，以及“分析与处理”菜单。发布收尾另修复更新包选择、EXE 重启路径、寿命与热传导的时间轴重复缩放，以及全局 Qt 消息回调的退出解绑。完整用户变化见[更新日志](../../updatinglog.md)。

## 最终自动回归

环境为 Windows x64、Python 3.12.14、RTX 5070 Ti，使用仓库固定依赖。完整测试保持原生 Qt 日志与退出回归启用，没有替换生产函数。运行时暂停 EXE 的并行 GUI 验收，60 秒定时堆栈没有触发。

| 检查 | 结果 | unittest 用时 | 进程退出码 |
| --- | --- | --- | --- |
| `pip check` | 依赖一致 | — | 0 |
| 默认完整回归 | 518 项，509 通过、9 项目标 GPU 用例跳过 | 21.593 秒 | 0 |
| 开启目标 GPU 的完整回归 | 518 项全部通过，0 跳过 | 29.828 秒 | 0 |

测试前后 `core/`、`tests/`、依赖清单和 `tools/` 的内容快照相同。原始统计及快照摘要见[完整回归记录](regression-1.1.2-release.json)。

本次新增 21 项回归：10 项更新资产与重启路径、9 项物理/旧帧时间轴和已知寿命、2 项 Qt 消息处理器恢复与退出解绑。已知寿命 `5 ps` 在界面步长 `0.5` 下仍拟合为 `5 ps`；区域、热图和热传导三条入口均核对，旧帧序号仍按本次步长换算。

## 发布包

使用 `core/LifeCalor-CPU.spec` 与 `core/LifeCalor.spec` 标准构建，不使用归档补丁。两个 EXE 均检查内嵌版本、分析菜单模块、更新器辅助函数和科学时间轴引用。

发布资产为 `LifeCalor-v1.1.2.exe`、`LifeCalor_CPU-v1.1.2.exe` 和 `SHA256SUMS.txt`。完整包包含 CPU/CUDA，CPU 精简包排除 CuPy/CUDA。实际包内 worker 不继承源码虚拟环境的模块路径。

CPU 资产名使用下划线，使 GitHub 的资产排序先返回完整版，兼容旧版只选择首个安装包并按完整版文件名重启的更新器。新版更新器同时识别下划线和连字符的 CPU 标记；已用两种旧程序名及实际远端资产顺序核对选择结果。

最终四轮独立验收全部正常退出，原始版本、模块断言、计算结果、大小及 SHA256 见[发布包原始记录](package-1.1.2-release.json)。

| 产物与检查 | 结果 | 验证脚本退出码 |
| --- | --- | --- |
| CPU 包内 worker 与默认 2 秒关闭 | CUDA 排除正确，主窗口显示与关闭正常 | 0 |
| CUDA 包内 worker 与默认 2 秒关闭 | RTX 5070 Ti 自检、STFT 与单/双指数寿命通过，主窗口显示与关闭正常 | 0 |
| CUDA 主窗显示后 0 秒关闭 | 正常退出 | 0 |
| CUDA 主窗显示后 8 秒关闭 | 正常退出 | 0 |

两个标准构建进程退出码也均为 0。CUDA 冻结 worker 的 STFT 最大绝对误差为 `5.96e-8`，已知单指数寿命 `2.5` 及双指数 `(2.5, 12)` 拟合正确。CPU 包为 `239157578` 字节，CUDA 包为 `1343542785` 字节；公开资产与构建产物使用相同字节内容，上传后再核对 GitHub 提供的 SHA256。

复验命令如下。第三、四行分别覆盖启动后立即关闭和后台自检完成后关闭。

```powershell
.\.venv\Scripts\python.exe tools/verify_package.py dist/LifeCalor-CPU.exe --backend cpu
.\.venv\Scripts\python.exe tools/verify_package.py dist/LifeCalor.exe --backend gpu
.\.venv\Scripts\python.exe tools/verify_package.py dist/LifeCalor.exe --backend gpu --window-only --close-delay 0
.\.venv\Scripts\python.exe tools/verify_package.py dist/LifeCalor.exe --backend gpu --window-only --close-delay 8
```

## 中间诊断与结论边界

收尾过程中，旧基线 `e168b52` 的两轮完整 offscreen 测试分别停在 Qt 原生 `dock.hide()` 和 `empty_widget.setVisible()`。已保留外部 Python 堆栈和定时堆栈，两轮均由测试工具结束，未计入通过结果。同一四画布关闭用例在 offscreen 下连续 20 次通过，在真实 Windows 平台的 20 个独立进程也全部通过。临时禁用 Qt 消息回调的一次诊断完成算法及画布用例，但原生日志用例因诊断开关而预期失败，该轮也不计正式验收。

旧 CUDA EXE 首轮正常关闭曾返回 `0xC0000005`，日志显示工作线程已退出、应用日志已解绑，未取得故障模块。随后三轮针对本次自启进程的 ProcDump 监控未复现异常。全局 Qt Python 消息回调未恢复是审查确认的生命周期缺口，已在 `6652905` 修复；现有证据不足以将上述所有原生现象归为同一根因。最终源码的完整默认/GPU 回归均正常退出，以本报告及最终包记录为准。

中间日志与堆栈保存在本机 `.venv/release-1.1.2/`，包括 `tests-cuda-final-stalled-stack.txt`、`tests-cuda-final-diagnostic.log`、`application-after-cuda-failure.log` 和 ProcDump 监控记录。开发阶段历史记录保留于[原 1.1.2 验收](lifecalor-1.1.2-acceptance.md)和[菜单合并验收](lifecalor-analysis-merge.md)。

## 真实数据和后续验收

用户授权目录 `G:/Data/2026` 的真实 AVI/TIFF 只读验证沿用[RTX 5070 Ti 报告](2026-11-06-rtx5070ti-validation.md)，两份源文件的大小、修改时间及 SHA256 前后相同。真实 AVI 容器报告 989 帧、实际解码 491 帧，导入器现会提示差异；STFT/CWT 使用实际解码帧，未补造缺失帧。

Auto 根据 small/medium 实测保守使用 CPU，仍可显式选择 GPU；时空卷积工作台正式执行使用 CPU。真实载流子寿命还需要带明确采样时间的衰减序列，CUDA 卷积完整工具流程与更大规模性能/显存峰值仍待后续验收。本轮没有将单页 TIFF 当作真实寿命序列。
