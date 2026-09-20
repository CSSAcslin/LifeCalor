from __future__ import annotations

import math
import os
from dataclasses import dataclass

from .capabilities import HardwareSnapshot
from .model import CapabilityStatus


MIB = 1024 ** 2
GIB = 1024 ** 3


@dataclass(frozen=True)
class ResourceRecommendation:
    cpu_workers: int
    host_memory_limit_mb: int
    gpu_memory_percent: int
    selected_device: str = ""
    reasons: tuple[str, ...] = ()


def _fallback_counts():
    logical = max(1, int(os.cpu_count() or 1))
    try:
        import psutil

        physical = int(psutil.cpu_count(logical=False) or logical)
        memory = psutil.virtual_memory()
        return (
            physical,
            logical,
            float(psutil.cpu_percent(interval=None)),
            int(memory.total),
            int(memory.available),
        )
    except Exception:
        return logical, logical, 0.0, 0, 0


def recommend_resources(snapshot: HardwareSnapshot | None) -> ResourceRecommendation:
    """Choose conservative per-task limits while leaving the desktop responsive."""
    if snapshot is None:
        physical, logical, cpu_percent, ram_total, ram_available = _fallback_counts()
        devices = ()
    else:
        physical = int(snapshot.physical_cores or snapshot.logical_cores or 1)
        logical = int(snapshot.logical_cores or physical or 1)
        cpu_percent = float(snapshot.cpu_percent or 0.0)
        ram_total = int(snapshot.ram_total_bytes or 0)
        ram_available = int(snapshot.ram_available_bytes or 0)
        devices = tuple(snapshot.devices)

    base_workers = max(1, min(logical, int(math.floor(max(1, physical) * 0.8))))
    if cpu_percent >= 80:
        workers = max(1, base_workers // 2)
        cpu_reason = f"CPU 当前负载 {cpu_percent:.0f}%，自动降低并行度"
    elif cpu_percent >= 60:
        workers = max(1, int(math.floor(base_workers * 0.75)))
        cpu_reason = f"CPU 当前负载 {cpu_percent:.0f}%，保留更多前台余量"
    else:
        workers = base_workers
        cpu_reason = f"使用约 80% 物理核心，保留系统响应余量"

    if ram_available > 0:
        reserve = max(2 * GIB, int(ram_total * 0.2))
        usable = max(256 * MIB, ram_available - reserve)
        host_bytes = max(256 * MIB, min(usable, int(ram_available * 0.65)))
        memory_reason = "按当前可用内存保留至少 2 GB 或总内存 20%"
    else:
        host_bytes = 4096 * MIB
        memory_reason = "内存状态不可用，使用 4 GB 安全默认值"

    gpu_percent = 70
    selected_device = ""
    available_gpus = [
        device for device in devices
        if device.kind == "gpu" and device.status is CapabilityStatus.AVAILABLE
    ]
    if available_gpus:
        device = max(available_gpus, key=lambda item: int(item.free_memory_bytes or 0))
        selected_device = device.device_id
        total = max(1, int(device.total_memory_bytes or 0))
        free = max(0, int(device.free_memory_bytes or 0))
        reserve = max(GIB, int(total * 0.15))
        usable = max(0, free - reserve)
        gpu_percent = min(75, max(10, int(usable * 100 / total)))
        gpu_reason = f"选择可用显存最多的 {device.name}，并保留至少 1 GB 或 15% 显存"
    else:
        gpu_reason = "没有已通过自检的 GPU，GPU 配额暂不启用"

    return ResourceRecommendation(
        cpu_workers=workers,
        host_memory_limit_mb=max(256, int(host_bytes // MIB)),
        gpu_memory_percent=gpu_percent,
        selected_device=selected_device,
        reasons=(cpu_reason, memory_reason, gpu_reason),
    )