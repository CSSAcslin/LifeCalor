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
from compute.algorithms import cwt_frequency_trace
from compute.algorithms.cwt_pipeline import run_cwt_pipeline
from compute.algorithms.preprocess_pipeline import run_preprocess_pipeline
from compute.model import (
    BackendPreference,
    CapabilityStatus,
    ComputeRequest,
    DeviceCapability,
    PrecisionPolicy,
    ResourceBudget,
)
from compute.planner import plan_compute
from compute.backends.cpu import cwt_block_cpu
from compute.worker import CudaBackendError


def make_plan(
    algorithm, source, parameters, *, threshold=10**9,
    backend=BackendPreference.CPU, capability=None,
):
    request = ComputeRequest(
        task_id=f"{algorithm}-test",
        attempt_id=0,
        algorithm=algorithm,
        data_id="source",
        shape=source.shape,
        dtype=source.dtype,
        axes="THW",
        source=source,
        parameters=parameters,
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
        capabilities=(capability,) if capability is not None else (),
        disk_output_threshold_bytes=threshold,
    )


class _CpuEquivalentCwtWorker:
    def __init__(self, _device_index):
        pass

    def execute(
        self, algorithm_id, *, block, params, compute_dtype, output_dtype,
        token, task_id, attempt_id, block_id,
    ):
        if algorithm_id != "cwt":
            raise AssertionError(algorithm_id)
        return cwt_block_cpu(
            block,
            params,
            compute_dtype=compute_dtype,
            output_dtype=output_dtype,
        )

    def close(self):
        pass


class _FailingCwtWorker:
    def __init__(self, _device_index):
        pass

    def execute(self, *args, **kwargs):
        raise CudaBackendError("simulated CWT backend failure")

    def close(self):
        pass


class StageDPipelineTests(unittest.TestCase):
    def _gpu_capability(self):
        return DeviceCapability(
            "nvidia:0",
            "gpu",
            "Test GPU",
            status=CapabilityStatus.AVAILABLE,
            supported_algorithms=("stft", "cwt"),
        )

    def test_cwt_tiles_match_trace_reference(self):
        fps = 64
        time = np.arange(64) / fps
        source = np.empty((64, 2, 3), dtype=np.float32)
        for y in range(2):
            for x in range(3):
                source[:, y, x] = np.sin(2 * np.pi * 8 * time) * (1 + y + x)
        params = {
            "target_freq": 8.0,
            "scale_range": 2.0,
            "total_scales": 3,
            "wavelet": "morl",
            "fps": fps,
        }
        workspace = 3 * source.shape[0] * np.dtype("complex64").itemsize
        plan = make_plan(
            "cwt",
            source,
            {"output_shape": source.shape, "workspace_bytes_per_spatial_item": workspace},
        )
        with tempfile.TemporaryDirectory() as directory:
            result = run_cwt_pipeline(plan, params, cache_dir=directory)

        expected = np.empty_like(source)
        for y in range(2):
            for x in range(3):
                expected[:, y, x], _, _ = cwt_frequency_trace(
                    source[:, y, x], **params
                )
        np.testing.assert_allclose(result.output, expected, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(result.whole_mean, expected.mean((1, 2)), rtol=1e-6)

    def test_cwt_large_output_commits_array_ref(self):
        source = np.ones((16, 2, 2), dtype=np.float32)
        params = {
            "target_freq": 4.0,
            "scale_range": 0.0,
            "total_scales": 1,
            "wavelet": "morl",
            "fps": 32,
        }
        plan = make_plan(
            "cwt",
            source,
            {"output_shape": source.shape, "workspace_bytes_per_spatial_item": 1024},
            threshold=1,
        )
        with tempfile.TemporaryDirectory() as directory:
            result = run_cwt_pipeline(plan, params, cache_dir=directory)
            self.assertIsInstance(result.output, ArrayRef)
            self.assertEqual(np.load(result.output.path).shape, source.shape)

    def test_cwt_gpu_worker_path_matches_cpu_contract(self):
        source = np.arange(64 * 2 * 2, dtype=np.float32).reshape(64, 2, 2)
        params = {
            "target_freq": 8.0,
            "scale_range": 2.0,
            "total_scales": 3,
            "wavelet": "morl",
            "fps": 64,
        }
        plan = make_plan(
            "cwt",
            source,
            {"output_shape": source.shape, "workspace_bytes_per_spatial_item": 2048},
            backend=BackendPreference.GPU,
            capability=self._gpu_capability(),
        )
        with tempfile.TemporaryDirectory() as directory, patch(
            "compute.algorithms.cwt_pipeline.CudaWorkerClient",
            _CpuEquivalentCwtWorker,
        ):
            result = run_cwt_pipeline(plan, params, cache_dir=directory)
        expected, _, _ = cwt_block_cpu(
            source, params, compute_dtype="float32", output_dtype="float32"
        )
        self.assertEqual(result.plan.actual_backend, "gpu")
        np.testing.assert_allclose(result.output, expected, rtol=1e-6, atol=1e-6)

    def test_cwt_gpu_failure_restarts_from_clean_cpu_output(self):
        source = np.ones((32, 2, 2), dtype=np.float32)
        params = {
            "target_freq": 4.0,
            "scale_range": 0.0,
            "total_scales": 1,
            "wavelet": "morl",
            "fps": 32,
        }
        plan = make_plan(
            "cwt",
            source,
            {"output_shape": source.shape, "workspace_bytes_per_spatial_item": 1024},
            backend=BackendPreference.GPU,
            capability=self._gpu_capability(),
        )
        with tempfile.TemporaryDirectory() as directory, patch(
            "compute.algorithms.cwt_pipeline.CudaWorkerClient",
            _FailingCwtWorker,
        ):
            result = run_cwt_pipeline(
                plan, params, cache_dir=directory, allow_cpu_fallback=True
            )
        self.assertEqual(result.plan.actual_backend, "cpu")
        self.assertIn("回退 CPU", result.fallback_reason)

    def test_preprocess_matches_exact_reference_and_can_stream_to_disk(self):
        source = np.arange(6 * 3 * 4, dtype=np.uint16).reshape(6, 3, 4) + 1
        plan = make_plan(
            "em_preprocess",
            source,
            {"output_shape": source.shape, "workspace_bytes_per_spatial_item": 128},
            threshold=1,
        )
        with tempfile.TemporaryDirectory() as directory:
            result = run_preprocess_pipeline(
                plan, background_frames=3, cache_dir=directory
            )
            self.assertIsInstance(result.output, ArrayRef)
            actual = np.load(result.output.path)
        background = np.median(source[:3].astype(np.float32), axis=0)
        expected = (source.astype(np.float32) - background) / background
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(result.background, background)

    def test_preprocess_progress_is_monotonic_across_chunks(self):
        source = np.arange(6 * 64 * 64, dtype=np.uint16).reshape(6, 64, 64) + 1
        plan = make_plan(
            "em_preprocess",
            source,
            {
                "output_shape": source.shape,
                "workspace_bytes_per_spatial_item": 4096,
            },
        )
        self.assertGreater(plan.chunk_count, 1)
        updates = []
        with tempfile.TemporaryDirectory() as directory:
            run_preprocess_pipeline(
                plan,
                background_frames=3,
                cache_dir=directory,
                progress=lambda current, total, message: updates.append(
                    (current, total, message)
                ),
            )

        currents = [current for current, _, _ in updates]
        totals = {total for _, total, _ in updates}
        self.assertEqual(currents, sorted(currents))
        self.assertEqual(len(totals), 1)
        self.assertEqual(currents[-1], updates[-1][1])


if __name__ == "__main__":
    unittest.main()
