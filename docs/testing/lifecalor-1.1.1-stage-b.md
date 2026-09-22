# LifeCalor 1.1.1 阶段 B 验证记录

日期：2026-09-22

## 已实现

- `compute/registry.py` 是内置算法 ID、轴、输出、执行模式和可信 CPU/CUDA handler 的唯一来源。
- 原 `AlgorithmRegistry.py` 继续提供第三方/运行时 callable 注册兼容接口，不再维护第二份内置能力表。
- 设置算法键、精度契约和规划器由注册表派生；保留原有算法 ID 与 QSettings 键。
- `BlockPlan` 明确核心区、读取区、本地裁剪、前后 halo 和仅限全局源边界的 padding。
- CUDA worker 使用通用 `execute` 消息；STFT 已迁移，旧 `.stft()` API 仍可用。
- 增加固定 HW/THW 时空卷积：same 输出、卷积语义、显式 origin、reflect/constant 等边界、无自动归一化。
- 增加具名多结果 NPY 输出集合，供双指数寿命等多数组结果使用。

## 验证

运行完整的 1.1.1 阶段 A+B 聚焦 unittest 测试集。

结果：106 项通过，2 项跳过。跳过项是必须在目标 CUDA 设备上显式启用的阶段 A 数学原型测试。

额外检查：

- 阶段 B 修改文件通过 `py_compile`。
- `git diff --check` 通过。
- 卷积覆盖二维、THW、偶数核、非对称核、偏移 origin、reflect/constant 和不整除分块。
- 原 STFT CPU、GPU 模拟回退、OOM 递归缩块、落盘与 ArrayRef 测试通过。

## 尚未宣称完成

- 固定卷积 CUDA handler 仍是原型，不进入设备能力表，也没有用户菜单。
- CWT CUDA 与单/双指数寿命 CUDA 尚未完成阶段 C/D 的目标设备数值验收。
- 通用入口目前完成注册、消息、分块和输出地基；业务 UI 仍按后续阶段逐项迁移。
- 应用版本保持 1.1.0，未写发布日志、未创建 1.1.1 发布提交。
