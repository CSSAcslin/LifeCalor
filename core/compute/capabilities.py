from __future__ import annotations

import importlib.util
import json
import os
import platform
import subprocess
import time
import threading
from dataclasses import dataclass, replace

from .model import CapabilityStatus, DeviceCapability


@dataclass(frozen=True)
class HardwareSnapshot:
    captured_at: float
    cpu_name: str
    physical_cores: int
    logical_cores: int
    cpu_percent: float | None
    ram_total_bytes: int
    ram_available_bytes: int
    devices: tuple[DeviceCapability, ...]
    probe_error: str = ""



_HARDWARE_CACHE = None
_HARDWARE_CACHE_LOCK = threading.RLock()


def get_cached_snapshot():
    with _HARDWARE_CACHE_LOCK:
        return _HARDWARE_CACHE


def set_cached_snapshot(snapshot):
    global _HARDWARE_CACHE
    with _HARDWARE_CACHE_LOCK:
        _HARDWARE_CACHE = snapshot
    return snapshot


def reconcile_probe_snapshot(snapshot):
    """Keep session validation results while refreshing cheap hardware data."""
    if snapshot is None:
        return None
    with _HARDWARE_CACHE_LOCK:
        cached = _HARDWARE_CACHE
        if cached is None:
            return snapshot
        previous = {device.device_id: device for device in cached.devices}
        devices = []
        for fresh in snapshot.devices:
            validated = previous.get(fresh.device_id)
            if validated is None or validated.status not in (
                CapabilityStatus.AVAILABLE,
                CapabilityStatus.FAILED,
            ):
                devices.append(fresh)
                continue
            devices.append(replace(
                validated,
                name=fresh.name or validated.name,
                vendor=fresh.vendor or validated.vendor,
                total_memory_bytes=(
                    fresh.total_memory_bytes or validated.total_memory_bytes
                ),
                free_memory_bytes=(
                    fresh.free_memory_bytes or validated.free_memory_bytes
                ),
                driver=fresh.driver or validated.driver,
                backend=validated.backend or fresh.backend,
            ))
        return replace(snapshot, devices=tuple(devices))


def update_cached_device_capability(capability):
    """Atomically publish a device self-test result for the current session."""
    global _HARDWARE_CACHE
    with _HARDWARE_CACHE_LOCK:
        _HARDWARE_CACHE = merge_device_capability(_HARDWARE_CACHE, capability)
        return _HARDWARE_CACHE


def mark_algorithm_verified(device_id, algorithm):
    """Record that a real GPU task completed in this application session."""
    global _HARDWARE_CACHE
    device_id = str(device_id)
    algorithm = str(algorithm)
    with _HARDWARE_CACHE_LOCK:
        snapshot = _HARDWARE_CACHE
        if snapshot is None:
            return None
        devices = []
        verified_device = None
        for device in snapshot.devices:
            if device.device_id != device_id:
                devices.append(device)
                continue
            supported = tuple(dict.fromkeys((*device.supported_algorithms, algorithm)))
            verified = tuple(dict.fromkeys((*device.verified_algorithms, algorithm)))
            verified_device = replace(
                device,
                status=CapabilityStatus.AVAILABLE,
                detail=(
                    "CUDA 可用；已完成实际任务验证："
                    + ", ".join(name.upper() for name in verified)
                ),
                supported_algorithms=supported,
                verified_algorithms=verified,
            )
            devices.append(verified_device)
        if verified_device is None:
            return None
        _HARDWARE_CACHE = replace(
            snapshot,
            devices=tuple(devices),
            captured_at=time.time(),
        )
        return verified_device


def refresh_runtime_usage(snapshot):
    """Refresh cheap CPU/RAM counters without probing display drivers again."""
    if snapshot is None:
        return None
    try:
        import psutil

        memory = psutil.virtual_memory()
        return replace(
            snapshot,
            captured_at=time.time(),
            cpu_percent=float(psutil.cpu_percent(interval=None)),
            ram_total_bytes=int(memory.total),
            ram_available_bytes=int(memory.available),
        )
    except Exception:
        return snapshot


def merge_device_capability(snapshot, capability):
    if snapshot is None:
        snapshot = HardwareSnapshot(
            time.time(), "未知 CPU", 0, int(os.cpu_count() or 1), None, 0, 0, ()
        )
    previous = next(
        (
            device for device in snapshot.devices
            if device.device_id == capability.device_id
        ),
        None,
    )
    if previous is not None and capability.status is CapabilityStatus.AVAILABLE:
        capability = replace(
            capability,
            verified_algorithms=tuple(dict.fromkeys((
                *previous.verified_algorithms,
                *capability.verified_algorithms,
            ))),
        )
    devices = [
        device for device in snapshot.devices
        if device.device_id != capability.device_id
    ]
    devices.append(capability)
    return replace(snapshot, devices=tuple(devices), captured_at=time.time())

def _hidden_startupinfo():
    if os.name != "nt":
        return None
    info = subprocess.STARTUPINFO()
    info.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    info.wShowWindow = subprocess.SW_HIDE
    return info


def _nvidia_devices(timeout: float) -> list[DeviceCapability]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.free,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            startupinfo=_hidden_startupinfo(),
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if result.returncode != 0:
        return []
    cupy_present = importlib.util.find_spec("cupy") is not None
    devices = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            total = int(float(parts[2])) * 1024 * 1024
            free = int(float(parts[3])) * 1024 * 1024
        except ValueError:
            total = free = 0
        detail = (
            "已检测到 CUDA Python 后端；等待本次启动的 CUDA 自检"
            if cupy_present
            else "已检测到 NVIDIA GPU，但未安装可选 CUDA Python 后端"
        )
        devices.append(DeviceCapability(
            device_id=f"nvidia:{parts[0]}",
            kind="gpu",
            name=parts[1],
            vendor="NVIDIA",
            status=(
                CapabilityStatus.UNAVAILABLE
                if cupy_present else CapabilityStatus.UNSUPPORTED
            ),
            total_memory_bytes=total,
            free_memory_bytes=free,
            driver=parts[4],
            backend="CuPy/CUDA" if cupy_present else "",
            detail=detail,
            supported_algorithms=(),
        ))
    return devices


def _windows_display_devices(timeout: float) -> list[DeviceCapability]:
    if os.name != "nt":
        return []
    script = (
        "Get-CimInstance Win32_VideoController | "
        "Select-Object PNPDeviceID,Name,AdapterCompatibility,AdapterRAM,DriverVersion | "
        "ConvertTo-Json -Compress"
    )
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            startupinfo=_hidden_startupinfo(),
        )
        if result.returncode != 0 or not result.stdout.strip():
            return []
        payload = json.loads(result.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return []
    rows = payload if isinstance(payload, list) else [payload]
    devices = []
    for index, row in enumerate(rows):
        name = str(row.get("Name") or "未知图形设备")
        vendor = str(row.get("AdapterCompatibility") or "")
        try:
            memory = max(0, int(row.get("AdapterRAM") or 0))
        except (TypeError, ValueError):
            memory = 0
        devices.append(DeviceCapability(
            device_id=str(row.get("PNPDeviceID") or f"display:{index}"),
            kind="gpu",
            name=name,
            vendor=vendor,
            status=CapabilityStatus.UNSUPPORTED,
            total_memory_bytes=memory,
            driver=str(row.get("DriverVersion") or ""),
            detail="已检测到图形设备；本版本尚未适配该设备的计算后端",
        ))
    return devices


def probe_hardware(timeout: float = 3.0) -> HardwareSnapshot:
    """Collect hardware information without importing an accelerator runtime."""
    errors = []
    try:
        import psutil

        physical = int(psutil.cpu_count(logical=False) or 0)
        logical = int(psutil.cpu_count(logical=True) or os.cpu_count() or 1)
        cpu_percent = float(psutil.cpu_percent(interval=None))
        memory = psutil.virtual_memory()
        ram_total = int(memory.total)
        ram_available = int(memory.available)
    except Exception as exc:
        physical = 0
        logical = int(os.cpu_count() or 1)
        cpu_percent = None
        ram_total = ram_available = 0
        errors.append(f"CPU/RAM: {type(exc).__name__}: {exc}")

    devices = _nvidia_devices(timeout)
    if not devices:
        devices = _windows_display_devices(timeout)
    cpu_name = platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "未知 CPU")
    return HardwareSnapshot(
        captured_at=time.time(),
        cpu_name=cpu_name,
        physical_cores=physical,
        logical_cores=logical,
        cpu_percent=cpu_percent,
        ram_total_bytes=ram_total,
        ram_available_bytes=ram_available,
        devices=tuple(devices),
        probe_error="; ".join(errors),
    )
