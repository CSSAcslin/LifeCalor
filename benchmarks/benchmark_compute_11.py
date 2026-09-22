"""Repeatable LifeCalor 1.1.1 CPU/CUDA compute benchmark."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from compute.algorithms import stft_frequency_trace
from compute.algorithms.cwt_pipeline import run_cwt_pipeline
from compute.algorithms.lifetime_pipeline import run_lifetime_pipeline
from compute.algorithms.stft_pipeline import run_stft_pipeline
from compute.model import (
    BackendPreference,
    CapabilityStatus,
    ComputeRequest,
    PrecisionPolicy,
    ResourceBudget,
)
from compute.planner import plan_compute, resolve_precision
from compute.worker import probe_cuda_capability_isolated


TIME_FREQUENCY_SIZES = {
    "small": (256, 16, 16),
    "medium": (1024, 64, 64),
    "large": (4096, 128, 128),
}
LIFETIME_SIZES = {
    "small": (80, 8, 8),
    "medium": (160, 24, 24),
    "large": (240, 48, 48),
}
ALGORITHMS = ("stft", "cwt", "lifetime_single", "lifetime_double")


class RssSampler:
    def __init__(self, interval=0.01):
        self.process = psutil.Process(os.getpid())
        self.interval = float(interval)
        self.start_bytes = 0
        self.peak_bytes = 0
        self._stop = threading.Event()
        self._thread = None

    def __enter__(self):
        self.start_bytes = self.process.memory_info().rss
        self.peak_bytes = self.start_bytes
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def _sample(self):
        while not self._stop.wait(self.interval):
            self.peak_bytes = max(
                self.peak_bytes, self.process.memory_info().rss
            )

    def __exit__(self, *_args):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.peak_bytes = max(
            self.peak_bytes, self.process.memory_info().rss
        )

    @property
    def delta_bytes(self):
        return max(0, self.peak_bytes - self.start_bytes)


def _time_frequency_source(shape):
    fps = 256
    times = np.arange(shape[0], dtype=np.float32) / fps
    trace = (
        np.sin(2 * np.pi * 16 * times)
        + 0.25 * np.sin(2 * np.pi * 32 * times)
    )
    return np.broadcast_to(trace[:, None, None], shape).copy(), times, fps


def _lifetime_source(shape, model_type):
    times = np.linspace(0.0, 30.0, shape[0], dtype=np.float64)
    if model_type == "double":
        trace = (
            50.0 * np.exp(-times / 3.0)
            + 20.0 * np.exp(-times / 15.0)
            + 2.0
        )
    else:
        trace = 50.0 * np.exp(-times / 7.5) + 2.0
    return np.broadcast_to(trace[:, None, None], shape).copy(), times, None


def _source(algorithm, size):
    if algorithm.startswith("lifetime_"):
        model_type = algorithm.removeprefix("lifetime_")
        return _lifetime_source(LIFETIME_SIZES[size], model_type)
    return _time_frequency_source(TIME_FREQUENCY_SIZES[size])


def _device_index(capability):
    if capability is None:
        return 0
    try:
        return int(str(capability.device_id).rsplit(":", 1)[-1])
    except (TypeError, ValueError):
        return 0


def _plan(
    algorithm,
    source,
    parameters,
    backend,
    precision,
    capability,
    cpu_workers,
):
    request = ComputeRequest(
        task_id=f"benchmark-{algorithm}",
        attempt_id=0,
        algorithm=algorithm,
        data_id="synthetic",
        shape=source.shape,
        dtype=source.dtype,
        axes="THW",
        source=source,
        parameters=parameters,
        backend=BackendPreference(backend),
        precision=PrecisionPolicy(precision),
    )
    device_limit = 0
    capabilities = ()
    if capability is not None and capability.status is CapabilityStatus.AVAILABLE:
        device_limit = int(capability.free_memory_bytes * 0.7)
        capabilities = (capability,)
    return plan_compute(
        request,
        ResourceBudget(
            host_limit_bytes=2 * 1024 ** 3,
            device_limit_bytes=device_limit,
            disk_free_bytes=100 * 1024 ** 3,
            cpu_workers=max(1, int(cpu_workers)),
        ),
        capabilities=capabilities,
        allow_cpu_fallback=False,
        disk_output_threshold_bytes=512 * 1024 ** 2,
    )


def _stft_run(source, fps, plan, cache_dir, device_index):
    precision_info = resolve_precision(
        "stft", source.dtype, PrecisionPolicy(plan[1])
    )
    window = np.hanning(128)
    frequencies, times, magnitude, _ = stft_frequency_trace(
        np.zeros(source.shape[0], dtype=np.float32),
        fs=fps,
        window=window,
        nperseg=128,
        noverlap=96,
        nfft=128,
        target_freq=16.0,
    )
    params = {
        "fps": fps,
        "window": window,
        "window_size": 128,
        "noverlap": 96,
        "nfft": 128,
        "target_freq": 16.0,
        "scale_range": 0.0,
    }
    parameters = {
        "output_shape": (magnitude.shape[-1], *source.shape[1:]),
        "workspace_bytes_per_spatial_item": (
            source.shape[0] * np.dtype(precision_info.compute_dtype).itemsize
            + 2
            * frequencies.size
            * times.size
            * np.dtype("complex64").itemsize
        ),
        "max_spatial_items": 512,
    }
    plan = _plan(
        "stft", source, parameters, plan[0], plan[1], plan[2], plan[3]
    )
    result = run_stft_pipeline(
        plan,
        params,
        cache_dir=cache_dir,
        allow_cpu_fallback=False,
        device_index=device_index,
    )
    return result, tuple(result.output.shape), ()


def _cwt_run(source, fps, plan, cache_dir, device_index):
    params = {
        "target_freq": 16.0,
        "scale_range": 4.0,
        "total_scales": 8,
        "wavelet": "morl",
        "fps": fps,
    }
    parameters = {
        "output_shape": source.shape,
        "workspace_bytes_per_spatial_item": (
            8 * source.shape[0] * np.dtype("complex64").itemsize
        ),
        "max_spatial_items": 256,
    }
    compute_plan = _plan(
        "cwt", source, parameters, plan[0], plan[1], plan[2], plan[3]
    )
    result = run_cwt_pipeline(
        compute_plan,
        params,
        cache_dir=cache_dir,
        allow_cpu_fallback=False,
        device_index=device_index,
    )
    return result, tuple(result.output.shape), ()


def _lifetime_run(
    algorithm, source, times, plan, cache_dir, device_index
):
    model_type = algorithm.removeprefix("lifetime_")
    output_multiplier = 7 if model_type == "double" else 3
    parameters = {
        "output_shape": source.shape[1:],
        "output_multiplier": output_multiplier,
        "workspace_bytes_per_spatial_item": source.shape[0] * 32,
        "max_spatial_items": 256,
    }
    compute_plan = _plan(
        algorithm, source, parameters, plan[0], plan[1], plan[2], plan[3]
    )
    result = run_lifetime_pipeline(
        compute_plan,
        data_type="central positive",
        time_points=times,
        fit_params={
            "from_start_cal": True,
            "r_squared_min": 0.4,
            "peak_range": (0, source.shape[0] - 1),
            "tau_range": (1e-3, 100.0),
        },
        model_type=model_type,
        cpu_workers=max(1, int(plan[3])),
        allow_cpu_fallback=False,
        device_index=device_index,
        cache_dir=cache_dir,
    )
    fields = tuple(result.named_outputs)
    return result, tuple(result.lifetime_map.shape), fields


def run_once(
    algorithm,
    source,
    times,
    fps,
    backend,
    precision,
    cache_dir,
    capability,
    cpu_workers,
):
    plan_args = (backend, precision, capability, cpu_workers)
    device_index = _device_index(capability)
    started = time.perf_counter()
    with RssSampler() as memory:
        if algorithm == "stft":
            result, output_shape, fields = _stft_run(
                source, fps, plan_args, cache_dir, device_index
            )
        elif algorithm == "cwt":
            result, output_shape, fields = _cwt_run(
                source, fps, plan_args, cache_dir, device_index
            )
        else:
            result, output_shape, fields = _lifetime_run(
                algorithm,
                source,
                times,
                plan_args,
                cache_dir,
                device_index,
            )
    elapsed = time.perf_counter() - started
    return {
        "seconds": elapsed,
        "peak_rss_delta_bytes": memory.delta_bytes,
        "output_shape": output_shape,
        "output_fields": fields,
        "plan": result.plan,
        "fallback_reason": result.fallback_reason,
    }


def run(algorithm, size, backend, precision, repetitions, cpu_workers=1):
    source, times, fps = _source(algorithm, size)
    capability = None
    if backend == "gpu":
        capability = probe_cuda_capability_isolated(timeout=45.0)
        if capability.status is not CapabilityStatus.AVAILABLE:
            raise RuntimeError(f"CUDA 自检未通过: {capability.detail}")
        if algorithm not in capability.supported_algorithms:
            raise RuntimeError(f"CUDA 自检未声明支持算法: {algorithm}")

    runs = []
    with tempfile.TemporaryDirectory() as directory:
        for _ in range(max(1, repetitions)):
            runs.append(
                run_once(
                    algorithm,
                    source,
                    times,
                    fps,
                    backend,
                    precision,
                    Path(directory),
                    capability,
                    cpu_workers,
                )
            )

    plan = runs[-1]["plan"]
    timings = [item["seconds"] for item in runs]
    warm_timings = timings[1:] or timings
    return {
        "schema_version": 2,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "algorithm": algorithm,
        "size": size,
        "input_shape": source.shape,
        "input_dtype": str(source.dtype),
        "requested_backend": backend,
        "actual_backend": plan.actual_backend,
        "precision_policy": precision,
        "compute_dtype": plan.precision.compute_dtype,
        "output_shape": runs[-1]["output_shape"],
        "output_fields": runs[-1]["output_fields"],
        "chunk_shape": plan.chunk_shape,
        "chunk_count": plan.chunk_count,
        "cpu_workers": plan.cpu_workers,
        "device": {
            "name": capability.name if capability is not None else "CPU",
            "backend": capability.backend if capability is not None else "CPU",
            "driver": capability.driver if capability is not None else "",
            "total_memory_bytes": (
                capability.total_memory_bytes if capability is not None else None
            ),
        },
        "cold_start_seconds": timings[0],
        "warm_seconds": warm_timings,
        "warm_median_seconds": statistics.median(warm_timings),
        "peak_rss_delta_bytes": max(
            item["peak_rss_delta_bytes"] for item in runs
        ),
        "peak_device_bytes": None,
        "phase_seconds": {
            "io": None,
            "host_to_device": None,
            "kernel_or_solver": None,
            "device_to_host": None,
            "total": timings,
        },
        "fallback_reason": runs[-1]["fallback_reason"],
        "notes": (
            "GPU peak memory and phase timing are null until worker-side "
            "instrumentation is available; total time includes pipeline I/O."
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", choices=ALGORITHMS, default="stft")
    parser.add_argument("--size", choices=TIME_FREQUENCY_SIZES, default="small")
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument(
        "--precision",
        choices=("compatibility", "preserve_input", "single", "double"),
        default="compatibility",
    )
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--cpu-workers", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output
    payload = run(**{key: value for key, value in vars(args).items() if key != "output"})
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    print(text)
