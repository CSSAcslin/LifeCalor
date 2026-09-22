from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Callable

from .model import AlgorithmContract, COMPUTE_CONTRACT_VERSION


class ExecutionMode(str, Enum):
    TIME_TRACES = "time_traces"
    NEIGHBORHOOD_BLOCK = "neighborhood_block"
    MULTI_OUTPUT = "multi_output"


@dataclass(frozen=True)
class OutputSpec:
    name: str
    axes: str
    dtype: str
    semantic: str = ""
    primary: bool = False


@dataclass(frozen=True)
class AlgorithmSpec:
    algorithm_id: str
    display_name: str
    input_axes: str
    accepts_complex: bool
    compatibility_output_dtype: str
    reduction: str
    execution_mode: ExecutionMode
    outputs: tuple[OutputSpec, ...]
    cpu_handler: str
    cuda_handler: str | None = None
    prototype_cuda_handler: str | None = None
    configurable: bool = True
    contract_version: str = COMPUTE_CONTRACT_VERSION


def _output(name, axes, dtype, semantic="", primary=False):
    return OutputSpec(name, axes, dtype, semantic, primary)


_BUILTIN_SPECS = (
    AlgorithmSpec("stft", "短时傅里叶变换", "THW", True, "float32", "mean amplitude across selected frequency bins", ExecutionMode.TIME_TRACES, (_output("amplitude", "T'HW", "float32", primary=True),), "stft_block", cuda_handler="stft_block"),
    AlgorithmSpec("cwt", "连续小波变换", "THW", True, "float32", "mean(2 * abs(coefficients) / sqrt(scale), axis=scale)", ExecutionMode.TIME_TRACES, (_output("amplitude", "THW", "float32", primary=True),), "cwt_block", cuda_handler="cwt_block"),
    AlgorithmSpec("cwt_quality", "CWT 质量分析", "T", True, "float32", "absolute CWT coefficient spectrum", ExecutionMode.TIME_TRACES, (_output("spectrum", "ST", "float32", primary=True),), "cwt_quality_trace", cuda_handler="cwt_quality_trace", configurable=False),
    AlgorithmSpec("em_preprocess", "EM 预处理", "THW", False, "float32", "exact leading-frame median background normalization", ExecutionMode.TIME_TRACES, (_output("processed", "THW", "float32", primary=True),), "em_preprocess_block"),
    AlgorithmSpec("lifetime_single", "单指数寿命拟合", "THW", False, "float64", "one fitted lifetime and R-squared per pixel", ExecutionMode.MULTI_OUTPUT, (_output("lifetime_map", "HW", "float64", primary=True), _output("r_squared_map", "HW", "float64"), _output("fit_status", "HW", "int16")), "lifetime_single_block", cuda_handler="lifetime_fit_block"),
    AlgorithmSpec("lifetime_double", "双指数寿命拟合", "THW", False, "float64", "two fitted lifetimes, amplitudes, baseline, R-squared and status per pixel", ExecutionMode.MULTI_OUTPUT, (_output("tau1_map", "HW", "float64", primary=True), _output("tau2_map", "HW", "float64"), _output("amplitude1_map", "HW", "float64"), _output("amplitude2_map", "HW", "float64"), _output("baseline_map", "HW", "float64"), _output("r_squared_map", "HW", "float64"), _output("fit_status", "HW", "int16")), "lifetime_double_block", cuda_handler="lifetime_fit_block"),
    AlgorithmSpec("fft2", "二维傅里叶变换", "...HW", True, "complex64", "two-dimensional Fourier transform over the spatial axes", ExecutionMode.NEIGHBORHOOD_BLOCK, (_output("spectrum", "...HW", "complex64", primary=True),), "fft2_block", cuda_handler="fft2_block"),
    AlgorithmSpec("spatiotemporal_convolution", "时空卷积", "HW|THW", True, "float32", "same-shape convolution with explicit boundary and origin semantics", ExecutionMode.NEIGHBORHOOD_BLOCK, (_output("convolved", "HW|THW", "float32", primary=True),), "spatiotemporal_convolution_block", prototype_cuda_handler="spatiotemporal_convolution_block", configurable=False),
)

_SPEC_BY_ID = MappingProxyType({spec.algorithm_id: spec for spec in _BUILTIN_SPECS})
if len(_SPEC_BY_ID) != len(_BUILTIN_SPECS):
    raise RuntimeError("内置算法标识重复")


def list_algorithm_specs():
    return _BUILTIN_SPECS


def get_algorithm_spec(algorithm_id):
    try:
        return _SPEC_BY_ID[str(algorithm_id)]
    except KeyError as exc:
        raise ValueError(f"未知计算算法: {algorithm_id}") from exc


def user_algorithm_keys():
    return tuple(spec.algorithm_id for spec in _BUILTIN_SPECS if spec.configurable)


def algorithm_contracts():
    contracts = {spec.algorithm_id: AlgorithmContract(spec.algorithm_id, spec.input_axes, spec.outputs[0].axes, spec.accepts_complex, spec.compatibility_output_dtype, spec.reduction) for spec in _BUILTIN_SPECS}
    single = contracts["lifetime_single"]
    contracts["lifetime"] = AlgorithmContract("lifetime", single.input_axes, single.output_axes, False, "float64", single.reduction)
    return MappingProxyType(contracts)


ALGORITHM_CONTRACTS = algorithm_contracts()


@dataclass(frozen=True)
class LegacyAlgorithmSpec:
    name: str
    handler: Callable[..., Any]
    description: str = ""


class AlgorithmRegistry:
    """Compatibility registry for third-party/runtime callable registration."""

    def __init__(self):
        self._algorithms = {}

    def register(self, name, handler, description=""):
        if not name or not isinstance(name, str):
            raise ValueError("Algorithm name must be a non-empty string")
        if name in self._algorithms:
            raise ValueError(f"Algorithm already registered: {name}")
        if not callable(handler):
            raise TypeError("Algorithm handler must be callable")
        self._algorithms[name] = LegacyAlgorithmSpec(name, handler, description)

    def list_names(self):
        return sorted(self._algorithms)

    def describe(self, name):
        return self._algorithms[name].description

    def run(self, name, data, **kwargs):
        return self._algorithms[name].handler(data, **kwargs)
