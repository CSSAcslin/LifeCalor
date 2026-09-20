"""Repeatable LifeCalor 1.1 STFT/CWT compute benchmark."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from compute.algorithms import stft_frequency_trace
from compute.algorithms.cwt_pipeline import run_cwt_pipeline
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


SIZES = {
    "small": (256, 16, 16),
    "medium": (1024, 64, 64),
    "large": (4096, 128, 128),
}


def _source(shape):
    fps = 256
    times = np.arange(shape[0], dtype=np.float32) / fps
    trace = np.sin(2 * np.pi * 16 * times) + 0.25 * np.sin(2 * np.pi * 32 * times)
    return np.broadcast_to(trace[:, None, None], shape).copy(), fps


def _plan(algorithm, source, parameters, backend, precision, cache_dir, capability):
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
            cpu_workers=1,
        ),
        capabilities=capabilities,
        allow_cpu_fallback=False,
        disk_output_threshold_bytes=512 * 1024 ** 2,
    )


def run_once(algorithm, source, fps, backend, precision, cache_dir, capability):
    precision_info = resolve_precision(algorithm, source.dtype, PrecisionPolicy(precision))
    if algorithm == "stft":
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
                + 2 * frequencies.size * times.size * np.dtype("complex64").itemsize
            ),
            "max_spatial_items": 512,
        }
        plan = _plan("stft", source, parameters, backend, precision, cache_dir, capability)
        started = time.perf_counter()
        result = run_stft_pipeline(
            plan,
            params,
            cache_dir=cache_dir,
            allow_cpu_fallback=False,
        )
    else:
        if backend == "gpu":
            raise ValueError("CWT CUDA 尚未开放，基准只接受 CPU")
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
        plan = _plan("cwt", source, parameters, backend, precision, cache_dir, None)
        started = time.perf_counter()
        result = run_cwt_pipeline(plan, params, cache_dir=cache_dir)
    elapsed = time.perf_counter() - started
    return elapsed, tuple(result.output.shape), result.plan


def run(algorithm, size, backend, precision, repetitions):
    source, fps = _source(SIZES[size])
    capability = None
    if backend == "gpu":
        capability = probe_cuda_capability_isolated(timeout=15.0)
        if capability.status is not CapabilityStatus.AVAILABLE:
            raise RuntimeError(f"CUDA 自检未通过: {capability.detail}")
    timings = []
    output_shape = None
    plan = None
    with tempfile.TemporaryDirectory() as directory:
        for _ in range(max(1, repetitions)):
            elapsed, output_shape, plan = run_once(
                algorithm,
                source,
                fps,
                backend,
                precision,
                Path(directory),
                capability,
            )
            timings.append(elapsed)
    return {
        "algorithm": algorithm,
        "size": size,
        "input_shape": source.shape,
        "input_dtype": str(source.dtype),
        "backend": plan.actual_backend,
        "precision": precision,
        "output_shape": output_shape,
        "chunk_shape": plan.chunk_shape,
        "chunk_count": plan.chunk_count,
        "seconds": timings,
        "median_seconds": statistics.median(timings),
        "device": capability.name if capability is not None else "CPU",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", choices=("stft", "cwt"), default="stft")
    parser.add_argument("--size", choices=SIZES, default="small")
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument(
        "--precision",
        choices=("compatibility", "preserve_input", "single", "double"),
        default="compatibility",
    )
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    print(json.dumps(run(**vars(args)), ensure_ascii=False, indent=2))
