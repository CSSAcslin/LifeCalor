from __future__ import annotations

import os
from dataclasses import dataclass

from .model import BackendPreference, PrecisionPolicy
from .registry import user_algorithm_keys


ALGORITHM_KEYS = user_algorithm_keys()


@dataclass(frozen=True)
class ComputePreferences:
    default_backend: BackendPreference = BackendPreference.AUTO
    precision: PrecisionPolicy = PrecisionPolicy.COMPATIBILITY
    allow_cpu_fallback: bool = True
    auto_cpu: bool = True
    auto_memory: bool = True
    auto_gpu: bool = True
    cpu_workers: int = 1
    host_memory_limit_mb: int = 4096
    gpu_memory_percent: int = 70
    preferred_device: str = ""
    algorithm_backends: tuple[tuple[str, BackendPreference], ...] = ()
    algorithm_precisions: tuple[tuple[str, PrecisionPolicy], ...] = ()

    def backend_for(self, algorithm: str) -> BackendPreference:
        overrides = dict(self.algorithm_backends)
        selected = overrides.get(algorithm, BackendPreference.FOLLOW_GLOBAL)
        return self.default_backend if selected is BackendPreference.FOLLOW_GLOBAL else selected

    def precision_for(self, algorithm: str) -> PrecisionPolicy:
        return dict(self.algorithm_precisions).get(algorithm, self.precision)


def _as_bool(value, default=True):
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _enum_value(enum_type, value, default):
    try:
        return enum_type(str(value))
    except (ValueError, TypeError):
        return default


class ComputeSettingsStore:
    PREFIX = "compute"

    def __init__(self, settings):
        self.settings = settings

    def _key(self, name):
        return f"{self.PREFIX}/{name}"

    def load(self) -> ComputePreferences:
        logical = max(1, int(os.cpu_count() or 1))
        worker_default = min(logical, 8)
        legacy_workers = self.settings.value("cal_set/cpu_workers", worker_default)
        try:
            cpu_workers = int(self.settings.value(self._key("cpu_workers"), legacy_workers))
        except (TypeError, ValueError):
            cpu_workers = worker_default
        cpu_workers = min(logical, max(1, cpu_workers))

        try:
            host_limit = int(self.settings.value(self._key("host_memory_limit_mb"), 4096))
        except (TypeError, ValueError):
            host_limit = 4096
        try:
            gpu_percent = int(self.settings.value(self._key("gpu_memory_percent"), 70))
        except (TypeError, ValueError):
            gpu_percent = 70

        overrides = []
        precision_overrides = []
        for algorithm in ALGORITHM_KEYS:
            value = self.settings.value(
                self._key(f"algorithms/{algorithm}/backend"),
                BackendPreference.FOLLOW_GLOBAL.value,
            )
            overrides.append((
                algorithm,
                _enum_value(BackendPreference, value, BackendPreference.FOLLOW_GLOBAL),
            ))
            precision_value = self.settings.value(
                self._key(f"algorithms/{algorithm}/precision"), ""
            )
            if str(precision_value or ""):
                precision_overrides.append((
                    algorithm,
                    _enum_value(
                        PrecisionPolicy,
                        precision_value,
                        PrecisionPolicy.COMPATIBILITY,
                    ),
                ))

        return ComputePreferences(
            default_backend=_enum_value(
                BackendPreference,
                self.settings.value(self._key("default_backend"), BackendPreference.AUTO.value),
                BackendPreference.AUTO,
            ),
            precision=_enum_value(
                PrecisionPolicy,
                self.settings.value(self._key("precision"), PrecisionPolicy.COMPATIBILITY.value),
                PrecisionPolicy.COMPATIBILITY,
            ),
            allow_cpu_fallback=_as_bool(
                self.settings.value(self._key("allow_cpu_fallback"), True), True
            ),
            auto_cpu=_as_bool(self.settings.value(self._key("auto_cpu"), True), True),
            auto_memory=_as_bool(self.settings.value(self._key("auto_memory"), True), True),
            auto_gpu=_as_bool(self.settings.value(self._key("auto_gpu"), True), True),
            cpu_workers=cpu_workers,
            host_memory_limit_mb=max(256, host_limit),
            gpu_memory_percent=min(90, max(10, gpu_percent)),
            preferred_device=str(self.settings.value(self._key("preferred_device"), "") or ""),
            algorithm_backends=tuple(overrides),
            algorithm_precisions=tuple(precision_overrides),
        )

    def save(self, preferences: ComputePreferences) -> ComputePreferences:
        normalized = ComputePreferences(
            default_backend=BackendPreference(preferences.default_backend),
            precision=PrecisionPolicy(preferences.precision),
            allow_cpu_fallback=bool(preferences.allow_cpu_fallback),
            auto_cpu=bool(preferences.auto_cpu),
            auto_memory=bool(preferences.auto_memory),
            auto_gpu=bool(preferences.auto_gpu),
            cpu_workers=min(max(1, int(os.cpu_count() or 1)), max(1, int(preferences.cpu_workers))),
            host_memory_limit_mb=max(256, int(preferences.host_memory_limit_mb)),
            gpu_memory_percent=min(90, max(10, int(preferences.gpu_memory_percent))),
            preferred_device=str(preferences.preferred_device or ""),
            algorithm_backends=tuple(
                (key, BackendPreference(value))
                for key, value in preferences.algorithm_backends
                if key in ALGORITHM_KEYS
            ),
            algorithm_precisions=tuple(
                (key, PrecisionPolicy(value))
                for key, value in preferences.algorithm_precisions
                if key in ALGORITHM_KEYS
            ),
        )
        self.settings.setValue(self._key("default_backend"), normalized.default_backend.value)
        self.settings.setValue(self._key("precision"), normalized.precision.value)
        self.settings.setValue(self._key("allow_cpu_fallback"), normalized.allow_cpu_fallback)
        self.settings.setValue(self._key("auto_cpu"), normalized.auto_cpu)
        self.settings.setValue(self._key("auto_memory"), normalized.auto_memory)
        self.settings.setValue(self._key("auto_gpu"), normalized.auto_gpu)
        self.settings.setValue(self._key("cpu_workers"), normalized.cpu_workers)
        self.settings.setValue(self._key("host_memory_limit_mb"), normalized.host_memory_limit_mb)
        self.settings.setValue(self._key("gpu_memory_percent"), normalized.gpu_memory_percent)
        self.settings.setValue(self._key("preferred_device"), normalized.preferred_device)
        for algorithm, backend in normalized.algorithm_backends:
            self.settings.setValue(
                self._key(f"algorithms/{algorithm}/backend"), backend.value
            )
        precision_overrides = dict(normalized.algorithm_precisions)
        for algorithm in ALGORITHM_KEYS:
            key = self._key(f"algorithms/{algorithm}/precision")
            precision = precision_overrides.get(algorithm)
            if precision is None:
                self.settings.setValue(key, "")
            else:
                self.settings.setValue(key, precision.value)
        self.settings.sync()
        return normalized
