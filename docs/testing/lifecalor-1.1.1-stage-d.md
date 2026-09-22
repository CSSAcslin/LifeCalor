# LifeCalor 1.1.1 阶段 D 验证记录

日期：2026-09-22

## 已完成的代码路径

- 单指数和双指数热图共享具名多结果流水线。
- 双指数输出：tau1、tau2、amplitude1、amplitude2、baseline、R² 和 fit_status；tau1 是默认主显示字段，不计算两个 tau 的平均值。
- 结果标签页可即时切换具名字段，切换不重新拟合；当前字段同步进入 DataFrame 导出。
- 多结果按总估算大小决定内存/落盘，每个字段独立 ArrayRef，整组提交或整组清理。
- CUDA 求解器使用 float64、解析 Jacobian、下界变量变换、批量阻尼最小二乘和接受/拒绝步。
- 每条曲线保留独立有效样本掩码、峰位/相关性预筛、收敛状态、tau 与 R² 后筛选。
- GPU 任务支持隔离进程、取消、OOM 递归缩块、严格模式和干净 CPU 回退。
- 真实 GPU 任务成功后记录本次启动的 lifetime_single/lifetime_double 验证状态。

## 本机验证

- 阶段 A-D 聚焦回归：117 项通过，4 项跳过。
- NumPy 镜像直接验证了 GPU 求解器核心的单/双指数解析样例、边界与收敛逻辑。
- 模拟 worker 验证 GPU 分发、双指数具名结果和失败后 CPU 重启。
- 离屏 Qt 测试验证结果字段切换及当前 DataFrame 更新。
- 大结果测试验证三个单指数输出分别落盘；通用输出集合另覆盖提交与清理。

## 目标 GPU 验收

在目标电脑运行：

```powershell
$env:LIFECALOR_RUN_CUDA_TESTS='1'
$env:QT_QPA_PLATFORM='offscreen'
python -m unittest tests.test_compute_111_cuda_prototypes
```

然后分别用真实数据执行单指数与双指数热图，核对：

1. 任务面板显示实际 GPU、float64、设备和块大小。
2. tau1/tau2、R² 与 fit_status 可切换、恢复和导出。
3. CPU/GPU 使用同一输入时的有效像素数、失败状态分布和参数误差。
4. 运行中取消后无半成品；下一任务可正常启动。
5. 允许回退与严格 GPU 模式行为符合设置。

## 尚未完成边界

- 目标 GPU 上的数值、峰值显存、冷/热启动和实际性能尚未由本机替代验证。
- 若目标设备解析自检未达到规划阈值，能力自检会失败并禁止自动选择 GPU，不会静默放宽误差。
- 阶段 E 统一选项和阶段 F 发布工作仍未完成，应用版本保持 1.1.0。
