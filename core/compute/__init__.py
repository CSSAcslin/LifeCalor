"""Shared compute contracts used by CPU and optional accelerator backends."""

from .model import (
    ALGORITHM_CONTRACTS,
    COMPUTE_CONTRACT_VERSION,
    AlgorithmContract,
    BackendPreference,
    CapabilityStatus,
    ComputePlan,
    ComputeRequest,
    DeviceCapability,
    PrecisionDescription,
    PrecisionPolicy,
    ResourceBudget,
    compatibility_metadata,
    execution_metadata,
    format_plan_log,
    plan_execution_details,
)
from .planner import normalize_progress, plan_compute, resolve_precision
from .settings import ComputePreferences, ComputeSettingsStore

__all__ = [
    "ALGORITHM_CONTRACTS",
    "COMPUTE_CONTRACT_VERSION",
    "AlgorithmContract",
    "BackendPreference",
    "CapabilityStatus",
    "ComputePlan",
    "ComputePreferences",
    "ComputeRequest",
    "ComputeSettingsStore",
    "DeviceCapability",
    "PrecisionDescription",
    "PrecisionPolicy",
    "ResourceBudget",
    "compatibility_metadata",
    "execution_metadata",
    "format_plan_log",
    "plan_execution_details",
    "normalize_progress",
    "plan_compute",
    "resolve_precision",
]
