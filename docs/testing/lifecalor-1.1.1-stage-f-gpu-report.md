# LifeCalor 1.1.1 阶段 F：目标 GPU 验收记录

> 本文件是待填写的实机报告，不代表 CUDA 已验收。测试必须在最终使用的 NVIDIA 电脑和最终打包环境上执行。

## 1. 环境

- 日期：
- LifeCalor commit：
- Windows 版本：
- GPU：
- NVIDIA 驱动：
- CUDA runtime/toolkit：
- Python：
- CuPy：
- 打包方式：源码 / `LifeCalor.exe`

## 2. CUDA 科学一致性

在项目根目录执行：

```powershell
$env:LIFECALOR_RUN_CUDA_TESTS = "1"
H:\Newera\programing\.venv\Scripts\python.exe tests\test_compute_111_cuda_prototypes.py
```

预期共 4 项通过：

- 当前七种 CWT 小波的 CPU/CUDA 结果满足单精度误差门槛。
- CWT 质量分析保留尺度×时间二维谱。
- 单/双指数模型与解析 Jacobian 满足双精度误差门槛。
- 受界 CUDA 求解器恢复已知单/双指数寿命。

结果：

```text
粘贴 unittest 摘要与失败信息
```

## 3. 同契约性能基准

先执行一次小数据确认，再运行中型数据。每条命令的 JSON 建议保存到独立文件：

```powershell
python benchmarks/benchmark_compute_11.py --algorithm stft --size medium --backend gpu --repetitions 5 --output reports/stft-gpu.json
python benchmarks/benchmark_compute_11.py --algorithm cwt --size medium --backend gpu --repetitions 5 --output reports/cwt-gpu.json
python benchmarks/benchmark_compute_11.py --algorithm lifetime_single --size medium --backend gpu --precision double --repetitions 5 --output reports/lifetime-single-gpu.json
python benchmarks/benchmark_compute_11.py --algorithm lifetime_double --size medium --backend gpu --precision double --repetitions 5 --output reports/lifetime-double-gpu.json
```

使用相同算法、尺寸、精度和重复次数，将 `--backend gpu` 改为 `--backend cpu` 保存 CPU 对照。比较 `actual_backend`、`compute_dtype`、`output_shape`、`output_fields`、冷启动和热运行中位数。`peak_device_bytes` 与细分阶段当前为 `null`，不能据此声称显存峰值或内核耗时。

| 算法 | CPU 热中位数 | GPU 热中位数 | 实际后端 | 数值通过 | 备注 |
| --- | ---: | ---: | --- | --- | --- |
| STFT |  |  |  |  |  |
| CWT |  |  |  |  |  |
| 单指数寿命 |  |  |  |  |  |
| 双指数寿命 |  |  |  |  |  |

## 4. 真实数据工作流

- [ ] 导入真实大数据，画布显示、切帧和 ROI 正常。
- [ ] CWT 显式选择 GPU，任务面板显示实际 GPU，取消后可继续运行其他任务。
- [ ] CWT 结果与同精度 CPU 结果抽样对照。
- [ ] 单指数寿命显式选择 GPU，检查寿命图、R²、fit_status 和导出。
- [ ] 双指数寿命显式选择 GPU，逐项检查 tau1、tau2、两个 amplitude、baseline、R²、fit_status。
- [ ] 大输出落盘后可恢复、切换字段、继续显示和导出。
- [ ] GPU OOM 或后端失败时，严格 GPU 模式只报告一次错误；允许回退时任务面板和 metadata 明确记录 CPU 回退。
- [ ] 连续计算、取消和重开选项至少 10 次，无持续增长的 worker、映射、RAM 或 VRAM。

## 5. 打包验收

```powershell
pyinstaller --clean --noconfirm core/LifeCalor.spec
pyinstaller --clean --noconfirm core/LifeCalor-CPU.spec
```

- [ ] 完整版在目标 NVIDIA 电脑自检通过，STFT/CWT/单指数/双指数均实际调用 CUDA。
- [ ] CPU-only 版在未安装 CuPy 的电脑可启动，GPU 选项禁用，CPU 工作流正常。
- [ ] 两版均显示应用图标、启动页、浅色/深色主题和统一选项窗口。
- [ ] 打包程序日志中记录实际设备、后端、精度、块计划、回退和失败原因。

## 6. 最终结论

- CUDA 科学一致性：通过 / 不通过
- CUDA 性能证据：GPU 更快 / CPU 更快 / 规模相关 / 尚无结论
- CPU-only 打包：通过 / 不通过
- 完整 CUDA 打包：通过 / 不通过
- 允许发布 1.1.1：是 / 否
- 遗留问题：
