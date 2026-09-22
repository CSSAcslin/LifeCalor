import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from ArrayCache import ArrayRef
from compute.algorithms import stft_frequency_trace
from compute.algorithms.stft_pipeline import run_stft_pipeline
from compute.backends.cpu import stft_block_cpu
from compute.model import (
    BackendPreference,
    CapabilityStatus,
    ComputeRequest,
    DeviceCapability,
    PrecisionPolicy,
    ResourceBudget,
)
from compute.planner import plan_compute
from compute.worker import CudaBackendError


def fixture():
    fps = 64
    time = np.arange(64) / fps
    trace = np.sin(2 * np.pi * 8 * time).astype(np.float32)
    source = np.empty((64, 4, 5), dtype=np.float32)
    for y in range(4):
        for x in range(5):
            source[:, y, x] = trace * (1 + y + x)
    params = {
        "fps": fps,
        "window": np.hanning(32),
        "window_size": 32,
        "noverlap": 16,
        "nfft": 32,
        "target_freq": 8.0,
        "scale_range": 0.0,
    }
    _, times, magnitude, _ = stft_frequency_trace(
        trace,
        fs=fps,
        window=params["window"],
        nperseg=32,
        noverlap=16,
        nfft=32,
        target_freq=8.0,
    )
    return source, params, times.size


def make_plan(source, out_length, *, backend=BackendPreference.CPU, threshold=10**9,
              capability=None):
    request = ComputeRequest(
        task_id="stft-test",
        attempt_id=0,
        algorithm="stft",
        data_id="source",
        shape=source.shape,
        dtype=source.dtype,
        axes="THW",
        source=source,
        parameters={
            "output_shape": (out_length, source.shape[1], source.shape[2]),
            "workspace_bytes_per_spatial_item": 100_000,
        },
        backend=backend,
        precision=PrecisionPolicy.COMPATIBILITY,
    )
    return plan_compute(
        request,
        ResourceBudget(
            host_limit_bytes=1024 * 1024,
            device_limit_bytes=1024 * 1024,
            disk_free_bytes=1024 ** 3,
        ),
        capabilities=(capability,) if capability else (),
        allow_cpu_fallback=True,
        disk_output_threshold_bytes=threshold,
    )


class _FailingCudaWorker:
    def __init__(self, _device_index):
        pass

    def execute(self, *args, **kwargs):
        raise CudaBackendError("simulated backend failure")

    def close(self):
        pass


class _SplittingCudaWorker:
    calls = 0

    def __init__(self, _device_index):
        pass

    def execute(
        self, algorithm_id, *, block, params, compute_dtype, output_dtype, token,
        task_id, attempt_id, block_id,
    ):
        self.assert_algorithm = algorithm_id
        type(self).calls += 1
        if block.shape[1] * block.shape[2] > 1:
            raise CudaBackendError("simulated OOM", out_of_memory=True)
        return stft_block_cpu(
            block, params, compute_dtype=compute_dtype, output_dtype=output_dtype
        )

    def close(self):
        pass


class StftPipelineTests(unittest.TestCase):
    def test_bounded_cpu_tiles_match_trace_reference(self):
        source, params, out_length = fixture()
        plan = make_plan(source, out_length)
        self.assertGreater(plan.chunk_count, 1)

        with tempfile.TemporaryDirectory() as directory:
            result = run_stft_pipeline(plan, params, cache_dir=directory)

        expected = np.empty(result.output.shape, dtype=np.float32)
        for y in range(source.shape[1]):
            for x in range(source.shape[2]):
                _, _, magnitude, _ = stft_frequency_trace(
                    source[:, y, x],
                    fs=params["fps"], window=params["window"],
                    nperseg=params["window_size"], noverlap=params["noverlap"],
                    nfft=params["nfft"], target_freq=params["target_freq"],
                )
                expected[:, y, x] = magnitude
        np.testing.assert_allclose(result.output, expected, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(
            result.whole_mean, expected.mean(axis=(1, 2)), rtol=1e-6, atol=1e-7
        )

    def test_large_output_is_committed_as_array_ref(self):
        source, params, out_length = fixture()
        plan = make_plan(source, out_length, threshold=1)
        with tempfile.TemporaryDirectory() as directory:
            result = run_stft_pipeline(plan, params, cache_dir=directory)
            self.assertIsInstance(result.output, ArrayRef)
            self.assertTrue(result.output.path.exists())
            self.assertEqual(np.load(result.output.path).shape, plan.output_shape)

    def test_gpu_failure_restarts_cleanly_on_cpu(self):
        source, params, out_length = fixture()
        capability = DeviceCapability(
            "nvidia:0", "gpu", "Test GPU",
            status=CapabilityStatus.AVAILABLE,
            supported_algorithms=("stft",),
        )
        plan = make_plan(
            source, out_length, backend=BackendPreference.GPU, capability=capability
        )
        with tempfile.TemporaryDirectory() as directory, patch(
            "compute.algorithms.stft_pipeline.CudaWorkerClient", _FailingCudaWorker
        ):
            result = run_stft_pipeline(
                plan, params, cache_dir=directory, allow_cpu_fallback=True
            )
        self.assertEqual(result.plan.actual_backend, "cpu")
        self.assertIn("回退 CPU", result.fallback_reason)

    def test_strict_gpu_failure_does_not_fall_back(self):
        source, params, out_length = fixture()
        capability = DeviceCapability(
            "nvidia:0", "gpu", "Test GPU",
            status=CapabilityStatus.AVAILABLE,
            supported_algorithms=("stft",),
        )
        plan = make_plan(
            source, out_length, backend=BackendPreference.GPU, capability=capability
        )
        with tempfile.TemporaryDirectory() as directory, patch(
            "compute.algorithms.stft_pipeline.CudaWorkerClient", _FailingCudaWorker
        ):
            with self.assertRaises(CudaBackendError):
                run_stft_pipeline(
                    plan, params, cache_dir=directory, allow_cpu_fallback=False
                )
    def test_gpu_oom_retries_smaller_tiles_without_changing_backend(self):
        source, params, out_length = fixture()
        capability = DeviceCapability(
            "nvidia:0", "gpu", "Test GPU",
            status=CapabilityStatus.AVAILABLE,
            supported_algorithms=("stft",),
        )
        plan = make_plan(
            source, out_length, backend=BackendPreference.GPU, capability=capability
        )
        _SplittingCudaWorker.calls = 0
        with tempfile.TemporaryDirectory() as directory, patch(
            "compute.algorithms.stft_pipeline.CudaWorkerClient", _SplittingCudaWorker
        ):
            result = run_stft_pipeline(plan, params, cache_dir=directory)
        self.assertEqual(result.plan.actual_backend, "gpu")
        self.assertGreater(_SplittingCudaWorker.calls, plan.chunk_count)

    def test_importing_compute_stack_does_not_import_cupy(self):
        self.assertNotIn("cupy", sys.modules)


if __name__ == "__main__":
    unittest.main()
