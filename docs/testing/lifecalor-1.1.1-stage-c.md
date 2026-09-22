# LifeCalor 1.1.1 阶段 C 验证记录

日期：2026-09-22

## 已完成的代码路径

- 当前七种 UI 小波共用已冻结的 PyWavelets 积分核、差分、截边和尺度归约契约。
- CWT 信号计算支持 CPU 与隔离 CUDA worker，携带 task/attempt/block 标识。
- GPU OOM 最多递归缩小空间块三次；允许回退时从干净 CPU 输出重启，严格 GPU 模式直接报错。
- CPU 基线逐尺度计算并立即归约，避免完整 S×T×H×W 系数体。
- 质量分析保留 S×T 谱，不误用信号计算的尺度均值。
- 成功的真实 GPU CWT 任务写入本次启动的 verified_algorithms；设置界面不要求每次重复自检。
- 非有限输入在尚未定义等价 GPU 局部传播前拒绝 GPU；允许回退时使用 CPU 参考路径。

## 本机验证

- 阶段 A-C 聚焦回归：110 项通过，3 项跳过。
- 跳过项需要设置 `LIFECALOR_RUN_CUDA_TESTS=1` 并在目标 NVIDIA/CuPy 环境运行。
- 测试覆盖 CPU/PyWavelets 系数、七种小波、实复 dtype、分块、落盘、模拟 GPU 成功、GPU 失败回退、质量谱轴与 STFT 回归。

## 目标 GPU 验收命令

在目标 GPU 电脑的项目环境中运行：

```powershell
$env:LIFECALOR_RUN_CUDA_TESTS='1'
$env:QT_QPA_PLATFORM='offscreen'
python -m unittest tests.test_compute_111_cuda_prototypes
```

随后使用真实数据分别运行：

1. CWT 信号计算：morl 与 cmor，小/大数据各一次，确认任务面板实际后端为 GPU。
2. CWT 质量分析：显式 GPU，确认结果仍为尺度×时间二维谱。
3. 取消一次运行中的大 CWT；随后重新提交任务，确认 worker 和缓存可继续使用。
4. 在允许回退与严格 GPU 两种策略下各制造一次资源不足，确认行为不同且只有一条聚合错误。

## 未完成边界

- 当前开发机未替代用户的目标 NVIDIA 设备，因此阶段 C 只能标记“代码完成、待实机验收”。
- 尚未记录目标设备上的冷启动/热执行、峰值显存和 CPU/GPU 性能对比。
- 应用版本仍为 1.1.0，阶段 C 不单独发布。
