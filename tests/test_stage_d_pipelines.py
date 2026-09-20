import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from ArrayCache import ArrayRef
from compute.algorithms import cwt_frequency_trace
from compute.algorithms.cwt_pipeline import run_cwt_pipeline
from compute.algorithms.preprocess_pipeline import run_preprocess_pipeline
from compute.model import BackendPreference, ComputeRequest, PrecisionPolicy, ResourceBudget
from compute.planner import plan_compute


def make_plan(algorithm, source, parameters, *, threshold=10**9):
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
        backend=BackendPreference.CPU,
        precision=PrecisionPolicy.COMPATIBILITY,
    )
    return plan_compute(
        request,
        ResourceBudget(host_limit_bytes=1024 * 1024, disk_free_bytes=1024 ** 3),
        disk_output_threshold_bytes=threshold,
    )


class StageDPipelineTests(unittest.TestCase):
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
