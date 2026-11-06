"""Target-device checks for numerical cases beyond the quick CUDA self-test."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

CORE = Path(__file__).resolve().parents[1] / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from compute.algorithms.cwt import SUPPORTED_CWT_WAVELETS
from compute.algorithms.lifetime_pipeline import _fit_lifetime_block
from compute.algorithms.stft_pipeline import run_stft_pipeline
from compute.backends.cpu import cwt_block_cpu, stft_block_cpu
from compute.backends.cuda import cwt_block_cuda, stft_block_cuda
from compute.backends.lifetime_cuda import lifetime_fit_block_cuda
from compute.model import (
    BackendPreference, CapabilityStatus, ComputeRequest, DeviceCapability,
    PrecisionPolicy, ResourceBudget,
)
from compute.planner import plan_compute
from compute.worker import CudaBackendError, CudaWorkerClient
from tasks.model import CancellationToken, TaskCancelled


RUN_CUDA = os.environ.get("LIFECALOR_RUN_CUDA_TESTS") == "1"


@unittest.skipUnless(RUN_CUDA, "set LIFECALOR_RUN_CUDA_TESTS=1 on the target NVIDIA machine")
class CudaTargetWorkflowTests(unittest.TestCase):
    def test_stft_real_complex_and_edge_bins_match_cpu(self):
        fps = 64
        times = np.arange(128) / fps
        for dtype in ("float32", "float64", "complex64", "complex128"):
            values = 0.25 + np.cos(2 * np.pi * 8 * times)
            if dtype.startswith("complex"):
                values = values + 0.5j * np.sin(2 * np.pi * 8 * times)
            source = np.broadcast_to(values[:, None, None], (128, 2, 3)).astype(dtype)
            output_dtype = "float64" if dtype in ("float64", "complex128") else "float32"
            for frequency in (0.0, 8.0, 32.0):
                with self.subTest(dtype=dtype, frequency=frequency):
                    parameters = {
                        "fps": fps,
                        "window": np.hanning(32),
                        "window_size": 32,
                        "noverlap": 16,
                        "nfft": 32,
                        "target_freq": frequency,
                        "scale_range": 0.0,
                    }
                    expected = stft_block_cpu(
                        source, parameters, compute_dtype=dtype, output_dtype=output_dtype
                    )
                    actual = stft_block_cuda(
                        source, parameters, compute_dtype=dtype, output_dtype=output_dtype
                    )
                    self.assertEqual(actual[0].dtype, np.dtype(output_dtype))
                    relative = 1e-11 if output_dtype == "float64" else 1e-5
                    absolute = 1e-12 if output_dtype == "float64" else 1e-7
                    for observed, reference in zip(actual, expected):
                        np.testing.assert_allclose(
                            observed, reference, rtol=relative, atol=absolute
                        )

    def test_cwt_all_wavelets_and_real_complex_precisions_match_cpu(self):
        times = np.arange(128) / 128
        for dtype in ("float32", "float64", "complex64", "complex128"):
            values = np.sin(2 * np.pi * 12 * times)
            if dtype.startswith("complex"):
                values = values + 0.5j * np.cos(2 * np.pi * 12 * times)
            source = np.broadcast_to(values[:, None, None], (128, 2, 3)).astype(dtype)
            output_dtype = "float64" if dtype in ("float64", "complex128") else "float32"
            for wavelet in SUPPORTED_CWT_WAVELETS:
                with self.subTest(dtype=dtype, wavelet=wavelet):
                    parameters = {
                        "fps": 128,
                        "wavelet": wavelet,
                        "target_freq": 12.0,
                        "scale_range": 4.0,
                        "total_scales": 5,
                    }
                    expected = cwt_block_cpu(
                        source, parameters, compute_dtype=dtype, output_dtype=output_dtype
                    )
                    actual = cwt_block_cuda(
                        source, parameters, compute_dtype=dtype, output_dtype=output_dtype
                    )
                    tolerance = 1e-10 if output_dtype == "float64" else 5e-4
                    absolute = 1e-12 if output_dtype == "float64" else 5e-6
                    for observed, reference in zip(actual, expected):
                        np.testing.assert_allclose(
                            observed, reference, rtol=tolerance, atol=absolute
                        )

    def test_lifetime_heterogeneous_batch_and_invalid_status_match_cpu(self):
        times = np.linspace(0.0, 30.0, 200)
        fit_parameters = {
            "from_start_cal": True,
            "r_squared_min": 0.4,
            "peak_range": (0, 199),
            "tau_range": (1e-3, 100.0),
        }
        for model in ("single", "double"):
            with self.subTest(model=model):
                source = np.empty((200, 2, 3), dtype=np.float64)
                for index in range(6):
                    trace = (30 + 5 * index) * np.exp(-times / (2.5 + index / 5)) + index
                    if model == "double":
                        trace = trace + (15 + index) * np.exp(-times / (12 + index))
                    source[:, index // 3, index % 3] = trace
                source[:, 1, 1] = 4.0
                source[10, 1, 2] = np.nan
                expected = _fit_lifetime_block((source, None, times, model, fit_parameters))
                actual = lifetime_fit_block_cuda(
                    source, times, None, fit_parameters, model_type=model
                )
                np.testing.assert_array_equal(actual["fit_status"], expected["fit_status"])
                valid = expected["fit_status"] == 1
                self.assertEqual(int(valid.sum()), 4)
                for field in expected:
                    if field != "fit_status":
                        np.testing.assert_allclose(
                            actual[field][valid], expected[field][valid],
                            rtol=5e-4, atol=1e-5,
                        )

    def test_cancel_queued_gpu_request_reaps_worker(self):
        worker = CudaWorkerClient(0)
        process = worker._process
        token = CancellationToken()
        token.cancel()
        try:
            with self.assertRaises(TaskCancelled):
                worker.request("self_test", token=token, timeout=45)
            self.assertTrue(worker._closed)
            self.assertFalse(process.is_alive())
        finally:
            worker.close()

    def test_gpu_pipeline_splits_injected_oom_and_commits_disk_output(self):
        class LimitedWorker(CudaWorkerClient):
            rejected_blocks = 0

            def execute(self, algorithm_id, **payload):
                block = payload["block"]
                if block.shape[1] * block.shape[2] > 1:
                    type(self).rejected_blocks += 1
                    raise CudaBackendError("injected GPU OOM", out_of_memory=True)
                return super().execute(algorithm_id, **payload)

        times = np.arange(128) / 64
        source = np.broadcast_to(
            np.sin(2 * np.pi * 8 * times)[:, None, None], (128, 2, 2)
        ).astype(np.float32)
        parameters = {
            "fps": 64, "window": np.hanning(32), "window_size": 32,
            "noverlap": 16, "nfft": 32, "target_freq": 8.0, "scale_range": 0.0,
        }
        expected = stft_block_cpu(
            source, parameters, compute_dtype="float32", output_dtype="float32"
        )[0]
        request = ComputeRequest(
            task_id="target-oom", attempt_id=0, algorithm="stft", data_id="synthetic",
            shape=source.shape, dtype=source.dtype, axes="THW", source=source,
            parameters={"output_shape": expected.shape, "max_spatial_items": 4},
            backend=BackendPreference.GPU, precision=PrecisionPolicy.COMPATIBILITY,
        )
        device = DeviceCapability(
            "nvidia:0", "gpu", "target NVIDIA", status=CapabilityStatus.AVAILABLE,
            supported_algorithms=("stft",),
        )
        plan = plan_compute(
            request,
            ResourceBudget(host_limit_bytes=8 * 1024**2, device_limit_bytes=8 * 1024**2),
            capabilities=(device,), allow_cpu_fallback=False, disk_output_threshold_bytes=1,
        )
        with tempfile.TemporaryDirectory() as directory:
            with patch("compute.algorithms.stft_pipeline.CudaWorkerClient", LimitedWorker):
                result = run_stft_pipeline(
                    plan, parameters, cache_dir=directory, allow_cpu_fallback=False
                )
            self.assertEqual(result.plan.actual_backend, "gpu")
            self.assertFalse(result.fallback_reason)
            self.assertGreaterEqual(LimitedWorker.rejected_blocks, 3)
            np.testing.assert_allclose(
                np.load(result.output.path), expected, rtol=1e-5, atol=1e-7
            )


if __name__ == "__main__":
    unittest.main()
