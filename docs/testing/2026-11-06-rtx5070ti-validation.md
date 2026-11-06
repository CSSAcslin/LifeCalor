# RTX 5070 Ti 目标机计算验收记录

记录日期为 2026-11-06。验证基于 `a3bf420` 后的当前工作区，包含本轮双指数 CUDA 求解和 STFT 双精度修复。本记录覆盖合成信号、两份真实 AVI/TIFF 的小 ROI 数值一致性、隔离任务控制和端到端性能采样。EXE 工作流的验收另行记录。

## 环境与运行范围

| 项目 | 本机记录 |
| --- | --- |
| 操作系统 | Windows 11，10.0.26200 |
| GPU | NVIDIA GeForce RTX 5070 Ti，驱动查询总显存 16303 MiB |
| NVIDIA 驱动 | 616.56，CUDA driver API 13040 |
| Python / CuPy | Python 3.12.14，CuPy 14.2.0 |
| 实际 CUDA runtime | 13020，即 13.2，以 `cuda_self_test()` 返回值为准 |
| CUDA 缓存 | 仓库内 `.venv/cupy-cache` |
| CPU 基准配置 | `cpu_workers=1` |

环境、自检及各组误差的原始数据保存在 [accuracy.json](gpu-2026-11-06/accuracy.json)。

## 数值与任务控制

开启目标设备开关后，`*cuda*.py` 共 12 项通过，退出码为 0。其中包含原有的四项设备自检、五项目标工作流检查及三项可在 CPU 上运行的求解核心回归。STFT 精度专项两项和既有 STFT 管线六项也均通过。

| 检查 | 覆盖范围 | 本机结果 |
| --- | --- | --- |
| STFT | float32/float64/complex64/complex128，直流、常规频点及频率边缘 | 单精度最大绝对误差 `1.1921e-7`，双精度最大绝对误差 `2.2205e-16` |
| CWT | 当前七种小波，各测上述四种 dtype | 单精度最大绝对误差 `7.1526e-7`，双精度最大绝对误差 `1.7209e-15` |
| CWT 质量分析 | 保留尺度 × 时间二维谱并对照 CPU | 既有目标设备用例通过 |
| 单指数寿命 | 异构 2×3 像素批量，含常量和 NaN 输入 | 有效像素各字段最大绝对误差 `1.6267e-12`，全部状态与 CPU 相同 |
| 双指数寿命 | 异构 2×3 像素批量，七个具名输出及无效输入状态 | 有效像素各字段最大绝对误差 `1.1895e-11`，全部状态与 CPU 相同 |
| 请求取消 | 取消已排队的隔离 GPU 请求 | 抛出 `TaskCancelled`，隔离进程已结束且连接关闭 |
| OOM 重试及落盘 | 受控注入 OOM 后缩小空间块，末端块实际由 GPU 计算 | 保持 `actual_backend=gpu`，无 CPU 回退，提交的 NPY 与 CPU 参考一致 |

误差是本报告所列确定性信号上的绝对误差。STFT 单精度按 `rtol=1e-5, atol=1e-7` 检查，双精度按 `rtol=1e-11, atol=1e-12` 检查。CWT 单精度按 `rtol=5e-4, atol=5e-6` 检查，双精度按 `rtol=1e-10, atol=1e-12` 检查。异构寿命有效像素按 `rtol=5e-4, atol=1e-5` 检查所有具名数值字段，同时逐项检查 `fit_status`。

### 本轮发现并修复的问题

原批量 LM 求解可能接受过大的 softplus 坐标步长，令第二寿命分量陷入下界附近。在无噪声样例中，真实 `(tau1, tau2)=(2.5,12)` 曾返回约 `(2.33915,10)`，仍标记为成功。现在按每个样本的相对坐标长度缩放整条 LM 步，保持求解方向，避免一次更新令变换导数饱和。CPU 可运行的回归与 GPU 异构批量检查都能验证修复。[求解器](../../core/compute/backends/lifetime_cuda.py) · [核心回归](../../tests/test_lifetime_cuda_solver_core.py)

原 CPU STFT 在频谱归约后固定转换为 float32，使 double 策略发生中间舍入。现在 `stft_frequency_trace()` 接受输出 dtype，旧调用默认 float32，CPU 后端显式传递计划 dtype。新增的解析幅值测试可分辨小于 float32 分辨率的幅值差。[STFT 参考](../../core/compute/algorithms/stft.py) · [精度回归](../../tests/test_stft_precision.py)

## 相同计算配置的 CPU/GPU 性能采样

四个算法均使用 `compatibility` 精度。STFT/CWT 使用 float32，寿命使用 float64。每种算法、规模和后端重复五次。表中的耗时为后四次的中位数，记录来自现有基准脚本。CPU/GPU 的计算 dtype、输入及输出形状逐项一致，所有 GPU 记录均实际使用 GPU，`fallback_reason` 为空。

| 算法 | 输入 shape | CPU 中位数 / 秒 | GPU 中位数 / 秒 | 原始记录 |
| --- | --- | ---: | ---: | --- |
| STFT small | 256×16×16 | 0.002364 | 1.851782 | [CPU](gpu-2026-11-06/stft-small-cpu.json) / [GPU](gpu-2026-11-06/stft-small-gpu.json) |
| STFT medium | 1024×64×64 | 0.125438 | 1.872086 | [CPU](gpu-2026-11-06/stft-medium-cpu.json) / [GPU](gpu-2026-11-06/stft-medium-gpu.json) |
| CWT small | 256×16×16 | 0.014996 | 2.318452 | [CPU](gpu-2026-11-06/cwt-small-cpu.json) / [GPU](gpu-2026-11-06/cwt-small-gpu.json) |
| CWT medium | 1024×64×64 | 0.822582 | 3.368902 | [CPU](gpu-2026-11-06/cwt-medium-cpu.json) / [GPU](gpu-2026-11-06/cwt-medium-gpu.json) |
| 单指数 small | 80×8×8 | 0.044134 | 1.871019 | [CPU](gpu-2026-11-06/lifetime_single-small-cpu.json) / [GPU](gpu-2026-11-06/lifetime_single-small-gpu.json) |
| 单指数 medium | 160×24×24 | 0.398396 | 1.882392 | [CPU](gpu-2026-11-06/lifetime_single-medium-cpu.json) / [GPU](gpu-2026-11-06/lifetime_single-medium-gpu.json) |
| 双指数 small | 80×8×8 | 0.114209 | 1.954344 | [CPU](gpu-2026-11-06/lifetime_double-small-cpu.json) / [GPU](gpu-2026-11-06/lifetime_double-small-gpu.json) |
| 双指数 medium | 160×24×24 | 1.203803 | 1.927869 | [CPU](gpu-2026-11-06/lifetime_double-medium-cpu.json) / [GPU](gpu-2026-11-06/lifetime_double-medium-gpu.json) |

在以上规模和当前执行方式下，CPU 的端到端耗时均更低。每次 GPU 管线执行会建立并关闭隔离进程，耗时包含导入、CUDA 上下文、数据传输及结果回收。重复执行中位数仍包含这些成本。更大的实验规模、不同并行配置及持久 worker 的表现需要独立测量。

现有 JSON 的 `peak_device_bytes` 和细分阶段时间仍为 `null`。这些记录可用于评估端到端耗时，显存峰值和纯内核耗时需要另加设备端测量。

## 真实 AVI/TIFF 小 ROI 验证

源文件通过通用 `ImporterRegistry` 以只读方式读取。算法输出、缓存和 ROI 副本均保存到项目 `.venv/real-data-validation/`。两份源文件在操作前后逐一比较大小、`mtime_ns` 和 SHA256，所有字段一致。[完整记录与源文件指纹](gpu-2026-11-06/real-data.json)

| 源文件 | 文件大小 | 导入结果 | 实际分析范围 |
| --- | ---: | --- | --- |
| `G:/Data/2026/0114/no-150-b_1.avi` | 5,214,382 字节 | 实际解码 `491×40×88`，uint8 灰度 | 帧 `[0,491)`、行 `[4,36)`、列 `[28,60)`，得到 `491×32×32` |
| `G:/Data/2026/0115/1/1-aft_1_1.tif` | 8,364,134 字节 | 单页 `2040×2048`，uint16，轴 YX | 行 `[956,1084)`、列 `[960,1088)`，得到 `128×128` |

AVI 容器报告 989 帧，OpenCV 实际解码到 491 帧。此次测试使用上述实际 491 帧，差异的来源仍待采集记录或文件结构进一步核实。通用视频导入现已记录 `reported_frame_count`、`decoded_frame_count` 并产生对应的 `import_warning`，时间轴长度跟随实际解码结果。视频容器给出的 99 FPS 用作本次数值对照的算法采样率，物理采样率以实验采集记录为准。[短解码回归](../../tests/test_video_import_counts.py)

AVI 的原始分析值为 uint8 数字灰度。公共 CPU EM 预处理以前 20 帧的逐像素中位数作为背景，结果为 float32 相对对比度 `(I-background)/background`。分块预处理与整块参考的最大绝对误差为 0。随后分别使用显式 CPU 和严格 GPU 执行相同 ROI 的 STFT/CWT，两个 GPU 任务的 `actual_backend` 都为 `gpu`，回退原因为空。

| 真实 AVI 算法 | 参数 | 输出 shape | CPU/GPU 最大绝对误差 |
| --- | --- | --- | ---: |
| STFT | Hann 窗长 64，overlap 32，nfft 64，目标 10 Hz、范围 4 Hz | `17×32×32`，float32 | `9.3133e-10` |
| CWT | morl，8 个尺度，目标 10 Hz、范围 4 Hz | `491×32×32`，float32 | `4.6567e-9` |

TIFF 文件提供二维数字强度值，未给出物理空间单位或采集时间序列。此处仅验证图像 ROI 上的单位核与 `3×3` 均值核卷积。CPU 使用分块管线，GPU 使用已有卷积原型，均采用 reflect 全局边界。单位核最大绝对误差为 0，均值核最大绝对误差为 `0.0078125`，相对于参考峰值 `54155.5547` 为约 `1.44e-7`，输出均为 float32 数字强度。真实寿命参数对照仍需要带明确时间轴的衰减序列样本。

### 真实寿命序列的元数据补查

随后对 `G:/Data/2026` 的 28 个日期目录做了有界元数据核对，共检查 90 个目录、13,256 个文件名。每个日期最多核对 30 个目录和 5,000 个文件名，`0306` 与 `0310` 达到文件名上限，其余日期在所遍历范围内结束。此次补查只读取目录项、TIFF 头和少量伴随文件头。[检查范围与候选记录](gpu-2026-11-06/lifetime-data-screening.json)

在各个含 TIFF 的已访问目录中选取头部样本，共检查 35 份 TIFF 元数据。样本均为二维 YX、uint16，OME 和 ImageJ 时间轴信息均为空。`0306`、`0310` 与 `0828` 下可见编号 TIFF 文件序列，但序号尚不足以确定采集延迟、帧间隔及其物理单位。

检查到的 18 个伴随候选均为 CSV。表头包含 `EM_pre`、`r_stft`、`cwt`、`ROI_avg` 或提取曲线的 `_x/_y` 标记，部分文件含采样频率标签。所读表头与前几行提供了处理曲线的数值横轴，未建立这些横轴与载流子衰减采集延迟、物理时间单位的对应关系。

本次范围内尚未建立可用于真实载流子寿命验收的样本身份和标定时间轴，因此保留该项待验。继续验收需要确定原始衰减序列的位置、时间单位，以及逐帧时间列表或明确的时间步长。

## 复现命令与剩余验收

以下命令从仓库根目录的 PowerShell 执行，使用已配置的 NVIDIA 驱动、CuPy 和 CUDA 运行环境。

```powershell
$env:CUPY_CACHE_DIR = Join-Path $PWD ".venv/cupy-cache"
$env:LIFECALOR_RUN_CUDA_TESTS = '1'
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "*cuda*.py" -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_stft_precision.py" -v
.\.venv\Scripts\python.exe benchmarks/benchmark_compute_11.py --algorithm cwt --size medium --backend gpu --repetitions 5
```

仍需补齐大体积全量数据、真实寿命时间序列、字段切换、交互导出、持久缓存恢复和连续交互流程，并在最终 CPU/CUDA EXE 上执行对应工作流。OOM 检查采用受控错误注入，物理显存耗尽场景与长期资源稳定性留待专门压力验收。后续可沿用 [阶段 F 清单](lifecalor-1.1.1-stage-f-gpu-report.md) 继续记录。
