# LifeCalor 计算运行时与 EXE 分发

## 结论

LifeCalor 不在运行中的 EXE 内远程执行 pip install cupy。默认发布使用 LifeCalor.spec，包含构建环境中已经安装的完整 CPU/CUDA 后端；LifeCalor-CPU.spec 是额外提供的不含 CuPy/CUDA 后端的精简构建。

建议发布两种构建：

| 构建 | 适用环境 | 计算后端 | 建议形式 |
| --- | --- | --- | --- |
| LifeCalor 默认完整包 | 已安装兼容 NVIDIA 驱动的 Windows 机器 | CPU + CUDA | LifeCalor.spec |
| LifeCalor CPU 精简包 | 所有 Windows 机器 | CPU | LifeCalor-CPU.spec |

## 为什么不在 EXE 中远程下载 CuPy

- CuPy、CUDA 运行库和显卡驱动存在严格的版本匹配关系，运行时临时安装无法保证可复现。
- 企业代理、离线环境、权限和杀毒软件都可能中断下载，科学计算任务不应依赖一次临时联网安装。
- 冻结程序的临时解压目录不是可维护的 Python 环境，不能把 pip install 当作可靠更新机制。
- GPU 发行包必须在目标 NVIDIA 机器上完成启动、自检、STFT 科学一致性和大数据稳定性验收。

## 构建规则

- LifeCalor.spec 保持为默认完整构建配置，不因 CPU 精简包而修改。
- LifeCalor-CPU.spec 直接派生自默认配置，仅排除 CuPy、CUDA 后端及相关可选运行库，并使用独立产物名 LifeCalor-CPU。

### 默认完整包

1. 使用独立、锁定版本的构建环境安装对应 CUDA 主版本的 CuPy wheel。
2. 使用 PyInstaller hook 收集 CuPy 的动态库、子模块和分发元数据。
3. GPU 包优先使用 onedir，避免大型 CUDA 动态库在每次启动时重复解压。
4. 构建产物首次启动后自动执行隔离 CUDA 快速自检；自检失败时禁用 GPU 选项并回退 CPU。
### CPU 精简包

1. 明确排除 CuPy、CUDA 后端和相关可选运行库。
2. 检测到 NVIDIA GPU 但安装包没有 CUDA 后端时，只显示诊断信息，不提供在线安装按钮。

CuPy 官方安装说明：
https://docs.cupy.dev/en/stable/install.html

PyInstaller hook 说明：
https://pyinstaller.org/en/stable/hooks.html

## 自动资源策略

- CPU：以物理核心为基准，默认使用约 80%，并根据任务提交时的系统负载降低并行度。
- 内存：根据任务提交时的可用内存计算，至少保留 2 GB 或总内存的 20%。
- GPU：仅选择已通过隔离自检且支持当前算法的设备，至少保留 1 GB 或总显存的 15%。
- 手动模式：CPU 工作数、工作内存上限和显存比例分别可关闭自动分配并独立设置。
- Auto 后端：只有算法和设备均通过验证时选择 GPU；其他情况使用 CPU。
- 每个任务的请求后端、实际后端、设备、精度、资源预算、分块和回退原因同时写入日志及任务面板。

## 发布验收

### CPU 精简包

- 无 NVIDIA GPU 的机器可启动。
- 未安装 CuPy 的机器不弹出依赖错误。
- GPU 选项不可选择，CPU 算法和任务面板正常。
- 日志记录 CPU、内存检测结果和每个任务的执行计划。

### 默认完整包

- 安装包不依赖用户本地 Python 环境。
- 启动探测不阻塞主界面。
- CUDA 快速自检通过后，仅开放已经验证的算法。
- STFT CPU/GPU 输出满足既定误差阈值。
- GPU 显存不足时按策略缩块或回退，日志和任务面板显示实际执行路径。
- EXE 冷启动、第二次启动和大数据任务均无 CUDA DLL 缺失。
