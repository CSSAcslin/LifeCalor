from __future__ import annotations

import math

import numpy as np

from memory.budget import array_nbytes

from .model import (
    ALGORITHM_CONTRACTS,
    BackendPreference,
    CapabilityStatus,
    ComputePlan,
    ComputeRequest,
    DeviceCapability,
    PrecisionDescription,
    PrecisionPolicy,
    ResourceBudget,
)


DISK_OUTPUT_THRESHOLD_BYTES = 512 * 1024 * 1024


def _contract_name(algorithm: str) -> str:
    return "lifetime" if algorithm.startswith("lifetime") else algorithm


def resolve_precision(algorithm: str, input_dtype, policy: PrecisionPolicy) -> PrecisionDescription:
    policy = PrecisionPolicy(policy)
    dtype = np.dtype(input_dtype)
    contract_name = _contract_name(algorithm)
    if contract_name not in ALGORITHM_CONTRACTS:
        raise ValueError(f"未知计算算法: {algorithm}")
    contract = ALGORITHM_CONTRACTS[contract_name]
    if dtype.kind == "c" and not contract.accepts_complex:
        raise ValueError(f"{algorithm} 不接受复数输入，请先明确选择实部、幅值或功率")

    if policy is PrecisionPolicy.COMPATIBILITY:
        if contract_name == "lifetime":
            compute = accumulator = output = "float64"
        else:
            compute = "complex128" if dtype == np.dtype("complex128") else (
                "complex64" if dtype.kind == "c" else "float32"
            )
            accumulator = "float64" if dtype.itemsize > 8 else "float32"
            output = contract.compatibility_output_dtype
    elif policy is PrecisionPolicy.PRESERVE_INPUT:
        if dtype.kind in "iu":
            compute = accumulator = "float64"
        elif dtype.kind == "c":
            compute = dtype.name
            accumulator = "complex128" if dtype.itemsize > 8 else "complex64"
        else:
            compute = dtype.name
            accumulator = "float64" if dtype.itemsize > 4 else "float32"
        output = (
            "float64" if np.dtype(compute).itemsize > 8 or compute == "float64" else "float32"
        )
    elif policy is PrecisionPolicy.SINGLE:
        if contract_name == "lifetime":
            raise ValueError("寿命拟合尚未验证单精度求解，当前请使用兼容现有或双精度")
        compute = "complex64" if dtype.kind == "c" else "float32"
        accumulator = output = "float32"
    else:
        compute = "complex128" if dtype.kind == "c" else "float64"
        accumulator = "complex128" if dtype.kind == "c" else "float64"
        output = "float64"

    return PrecisionDescription(
        policy=policy,
        input_dtype=dtype.name,
        compute_dtype=compute,
        accumulator_dtype=accumulator,
        output_dtype=output,
    )

def normalize_progress(current: int, total: int, maximum: int = 10000) -> tuple[int, int]:
    """Map arbitrarily large counters onto a stable Qt-friendly range."""
    total = max(0, int(total))
    current = max(0, int(current))
    maximum = max(1, int(maximum))
    if total == 0:
        return 0, 0
    scaled = min(maximum, (min(current, total) * maximum) // total)
    return int(scaled), maximum


def _select_backend(
    request: ComputeRequest,
    capabilities: tuple[DeviceCapability, ...],
    default_backend: BackendPreference,
    allow_cpu_fallback: bool,
) -> tuple[BackendPreference, str, str, DeviceCapability | None]:
    requested = request.backend
    if requested is BackendPreference.FOLLOW_GLOBAL:
        requested = BackendPreference(default_backend)
    algorithm = request.algorithm
    usable_gpu = next(
        (
            device for device in capabilities
            if device.kind == "gpu"
            and device.status is CapabilityStatus.AVAILABLE
            and algorithm in device.supported_algorithms
        ),
        None,
    )
    if requested is BackendPreference.CPU:
        return requested, "cpu", "用户指定 CPU", None
    if requested is BackendPreference.AUTO:
        if usable_gpu is not None:
            return requested, "gpu", f"自动策略选择已通过自检的设备 {usable_gpu.name}", usable_gpu
        return requested, "cpu", "尚无该设备与算法的可信性能档案，自动模式保守使用 CPU", None
    if usable_gpu is not None:
        return requested, "gpu", f"使用设备 {usable_gpu.name}", usable_gpu
    if allow_cpu_fallback:
        return requested, "cpu", "GPU 后端不可用或该算法尚未验证，已按设置回退 CPU", None
    raise RuntimeError("请求了严格 GPU 执行，但当前没有可用且已验证的 GPU 后端")


def _output_shape(request: ComputeRequest) -> tuple[int, ...]:
    configured = request.parameters.get("output_shape")
    if configured:
        shape = tuple(int(value) for value in configured)
    elif _contract_name(request.algorithm) == "lifetime" and len(request.shape) == 3:
        shape = request.shape[1:]
    else:
        shape = request.shape
    if not shape or any(value <= 0 for value in shape):
        raise ValueError(f"无效输出尺寸: {shape}")
    return shape


def _plan_cpu_chunk(request, precision, output_shape, host_limit_bytes):
    limit = int(host_limit_bytes)
    if limit <= 0:
        raise MemoryError("主机工作内存预算必须大于 0")
    input_itemsize = np.dtype(precision.compute_dtype).itemsize
    output_itemsize = np.dtype(precision.output_dtype).itemsize
    reserve = min(64 * 1024 * 1024, max(1, limit // 8))
    available = max(1, limit - reserve)

    if len(request.shape) == 3:
        time_length, height, width = request.shape
        output_time = output_shape[0] if len(output_shape) == 3 else 1
        default_per_pixel = (
            time_length * input_itemsize * 2
            + output_time * output_itemsize
        )
        working_per_pixel = max(
            1,
            int(request.parameters.get(
                "workspace_bytes_per_spatial_item", default_per_pixel
            )),
        )
        max_pixels = available // working_per_pixel
        configured_max = int(request.parameters.get("max_spatial_items", 0) or 0)
        if configured_max > 0:
            max_pixels = min(max_pixels, configured_max)
        if max_pixels < 1:
            raise MemoryError(
                "当前内存预算连一个像素时间序列计算块都无法容纳，请提高工作内存预算"
            )
        columns = min(width, max_pixels)
        rows = min(height, max(1, max_pixels // columns))
        chunk_shape = (time_length, rows, columns)
        count = math.ceil(height / rows) * math.ceil(width / columns)
        peak = min(limit, reserve + rows * columns * working_per_pixel)
    elif len(request.shape) > 3:
        raise ValueError(f"有界计算暂不支持超过三维的输入: {request.shape}")
    else:
        tail = math.prod(request.shape[1:]) if len(request.shape) > 1 else 1
        item_working = tail * input_itemsize * 2 + tail * output_itemsize
        rows = max(1, min(request.shape[0], available // max(1, item_working)))
        if item_working > available:
            raise MemoryError("当前内存预算连一个最小计算块都无法容纳")
        chunk_shape = (rows, *request.shape[1:])
        count = math.ceil(request.shape[0] / rows)
        peak = min(limit, reserve + rows * item_working)
    return tuple(int(value) for value in chunk_shape), int(count), int(peak)

def plan_compute(
    request: ComputeRequest,
    budget: ResourceBudget,
    *,
    capabilities: tuple[DeviceCapability, ...] = (),
    default_backend: BackendPreference = BackendPreference.AUTO,
    allow_cpu_fallback: bool = True,
    disk_output_threshold_bytes: int = DISK_OUTPUT_THRESHOLD_BYTES,
) -> ComputePlan:
    precision = resolve_precision(request.algorithm, request.dtype, request.precision)
    output_shape = _output_shape(request)
    input_bytes = array_nbytes(request.shape, request.dtype)
    output_bytes = array_nbytes(output_shape, precision.output_dtype)
    requested, actual, reason, selected_device = _select_backend(
        request, tuple(capabilities), default_backend, allow_cpu_fallback
    )
    planning_limit = int(budget.host_limit_bytes)
    if actual == "gpu":
        if int(budget.device_limit_bytes) <= 0:
            raise MemoryError("GPU 计划缺少有效显存预算")
        planning_limit = min(planning_limit, int(budget.device_limit_bytes))
    chunk_shape, chunk_count, planned_peak = _plan_cpu_chunk(
        request, precision, output_shape, planning_limit
    )
    peak_host = min(int(budget.host_limit_bytes), planned_peak)
    output_to_disk = output_bytes >= int(disk_output_threshold_bytes) or output_bytes > budget.host_limit_bytes // 2
    if output_to_disk and budget.disk_free_bytes and output_bytes > budget.disk_free_bytes:
        raise OSError(
            f"计算结果预计需要 {output_bytes} 字节，当前缓存磁盘可用空间不足"
        )
    peak_device = 0
    if actual == "gpu":
        peak_device = min(int(budget.device_limit_bytes), planned_peak)
    return ComputePlan(
        request=request,
        requested_backend=requested,
        actual_backend=actual,
        backend_reason=reason,
        precision=precision,
        input_bytes=input_bytes,
        output_shape=output_shape,
        output_bytes=output_bytes,
        chunk_shape=chunk_shape,
        chunk_count=chunk_count,
        peak_host_bytes=peak_host,
        peak_device_bytes=peak_device,
        output_to_disk=output_to_disk,
        cpu_workers=max(1, int(budget.cpu_workers)),
        host_budget_bytes=max(0, int(budget.host_limit_bytes)),
        device_budget_bytes=max(0, int(budget.device_limit_bytes)),
        selected_device=(selected_device.name if selected_device is not None else "CPU"),
    )
