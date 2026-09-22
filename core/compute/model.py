from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping

import numpy as np


COMPUTE_CONTRACT_VERSION = "1.1.1-a1"


class PrecisionPolicy(str, Enum):
    COMPATIBILITY = "compatibility"
    PRESERVE_INPUT = "preserve_input"
    SINGLE = "single"
    DOUBLE = "double"


class BackendPreference(str, Enum):
    FOLLOW_GLOBAL = "follow_global"
    AUTO = "auto"
    CPU = "cpu"
    GPU = "gpu"


class CapabilityStatus(str, Enum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"


class FrozenMapping(Mapping):
    """Small pickle-safe read-only mapping for spawned compute workers."""

    def __init__(self, values=None):
        self._values = dict(values or {})

    def __getitem__(self, key):
        return self._values[key]

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)

    def __reduce__(self):
        return FrozenMapping, (self._values,)

def _freeze_value(value):
    if isinstance(value, Mapping):
        return FrozenMapping({str(key): _freeze_value(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze_value(item) for item in value)
    return value


@dataclass(frozen=True)
class ComputeRequest:
    """Immutable task snapshot. Building one must never resolve an ArrayRef."""

    task_id: str
    attempt_id: int
    algorithm: str
    data_id: str
    shape: tuple[int, ...]
    dtype: str
    axes: str
    source: Any = field(repr=False, compare=False)
    parameters: Mapping[str, Any] = field(default_factory=dict)
    backend: BackendPreference = BackendPreference.AUTO
    precision: PrecisionPolicy = PrecisionPolicy.COMPATIBILITY

    def __post_init__(self):
        object.__setattr__(self, "shape", tuple(int(value) for value in self.shape))
        object.__setattr__(self, "dtype", np.dtype(self.dtype).name)
        object.__setattr__(self, "parameters", _freeze_value(dict(self.parameters or {})))
        object.__setattr__(self, "backend", BackendPreference(self.backend))
        object.__setattr__(self, "precision", PrecisionPolicy(self.precision))
        if not self.task_id or not self.data_id:
            raise ValueError("计算请求必须包含 task_id 和 data_id")
        if not self.shape or any(value <= 0 for value in self.shape):
            raise ValueError(f"无效数据尺寸: {self.shape}")


@dataclass(frozen=True)
class PrecisionDescription:
    policy: PrecisionPolicy
    input_dtype: str
    compute_dtype: str
    accumulator_dtype: str
    output_dtype: str


@dataclass(frozen=True)
class DeviceCapability:
    device_id: str
    kind: str
    name: str
    vendor: str = ""
    status: CapabilityStatus = CapabilityStatus.UNAVAILABLE
    total_memory_bytes: int = 0
    free_memory_bytes: int = 0
    driver: str = ""
    backend: str = ""
    detail: str = ""
    supported_algorithms: tuple[str, ...] = ()
    verified_algorithms: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResourceBudget:
    host_limit_bytes: int
    device_limit_bytes: int = 0
    disk_free_bytes: int = 0
    cpu_workers: int = 1


@dataclass(frozen=True)
class AlgorithmContract:
    algorithm: str
    input_axes: str
    output_axes: str
    accepts_complex: bool
    compatibility_output_dtype: str
    reduction: str


@dataclass(frozen=True)
class ComputePlan:
    request: ComputeRequest
    requested_backend: BackendPreference
    actual_backend: str
    backend_reason: str
    precision: PrecisionDescription
    input_bytes: int
    output_shape: tuple[int, ...]
    output_bytes: int
    chunk_shape: tuple[int, ...]
    chunk_count: int
    peak_host_bytes: int
    peak_device_bytes: int = 0
    output_to_disk: bool = False
    cpu_workers: int = 1
    host_budget_bytes: int = 0
    device_budget_bytes: int = 0
    selected_device: str = "CPU"


def compatibility_metadata(algorithm, input_dtype, **details):
    """Return serializable provenance for the frozen 1.1 compatibility path."""
    from .registry import ALGORITHM_CONTRACTS

    contract = ALGORITHM_CONTRACTS[algorithm]
    metadata = {
        "contract_version": COMPUTE_CONTRACT_VERSION,
        "algorithm": algorithm,
        "precision_policy": PrecisionPolicy.COMPATIBILITY.value,
        "input_dtype": np.dtype(input_dtype).name,
        "output_dtype": contract.compatibility_output_dtype,
        "contract": asdict(contract),
    }
    metadata.update(details)
    return metadata

def execution_metadata(plan, **details):
    """Return serializable provenance for a planned compute execution."""
    metadata = {
        "contract_version": COMPUTE_CONTRACT_VERSION,
        "algorithm": plan.request.algorithm,
        "requested_backend": plan.requested_backend.value,
        "actual_backend": plan.actual_backend,
        "backend_reason": plan.backend_reason,
        "precision_policy": plan.precision.policy.value,
        "input_dtype": plan.precision.input_dtype,
        "compute_dtype": plan.precision.compute_dtype,
        "accumulator_dtype": plan.precision.accumulator_dtype,
        "output_dtype": plan.precision.output_dtype,
        "chunk_shape": tuple(plan.chunk_shape),
        "chunk_count": int(plan.chunk_count),
        "peak_host_bytes": int(plan.peak_host_bytes),
        "peak_device_bytes": int(plan.peak_device_bytes),
        "output_to_disk": bool(plan.output_to_disk),
        "attempt_id": int(plan.request.attempt_id),
        "cpu_workers": int(plan.cpu_workers),
        "host_budget_bytes": int(plan.host_budget_bytes),
        "device_budget_bytes": int(plan.device_budget_bytes),
        "selected_device": str(plan.selected_device),
    }
    metadata.update(details)
    return metadata

def format_bytes(value):
    value = max(0, int(value or 0))
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{value} B"
        value /= 1024


def plan_execution_details(plan):
    """Return one canonical, UI/log friendly description of an execution plan."""
    chunk = "x".join(str(value) for value in plan.chunk_shape)
    resource_parts = [
        f"CPU {int(plan.cpu_workers)}",
        f"RAM {format_bytes(plan.host_budget_bytes)}",
    ]
    if plan.device_budget_bytes:
        resource_parts.append(f"VRAM {format_bytes(plan.device_budget_bytes)}")
    resource_parts.append(f"块 {chunk} x {int(plan.chunk_count)}")
    return {
        "attempt_id": int(plan.request.attempt_id),
        "algorithm": str(plan.request.algorithm),
        "requested_backend": plan.requested_backend.value,
        "actual_backend": str(plan.actual_backend),
        "backend_reason": str(plan.backend_reason),
        "precision": plan.precision.policy.value,
        "device": str(plan.selected_device or "CPU"),
        "cpu_workers": int(plan.cpu_workers),
        "host_budget_bytes": int(plan.host_budget_bytes),
        "device_budget_bytes": int(plan.device_budget_bytes),
        "chunk_shape": tuple(int(value) for value in plan.chunk_shape),
        "chunk_count": int(plan.chunk_count),
        "resource_summary": " · ".join(resource_parts),
    }


def format_plan_log(plan):
    details = plan_execution_details(plan)
    return (
        "计算计划: algorithm={algorithm} attempt={attempt_id} "
        "backend={requested_backend}->{actual_backend} device={device} "
        "precision={precision} cpu_workers={cpu_workers} "
        "host_budget={host_budget} device_budget={device_budget} "
        "chunk={chunk}x{chunk_count} reason={backend_reason}"
    ).format(
        **details,
        host_budget=format_bytes(details["host_budget_bytes"]),
        device_budget=format_bytes(details["device_budget_bytes"]),
        chunk="x".join(str(value) for value in details["chunk_shape"]),
    )
