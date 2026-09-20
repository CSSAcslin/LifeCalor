import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from compute.algorithms.lifetime_pipeline import (
    _prepare_block, _tiles, run_lifetime_pipeline, spatial_kernel,
)
from compute.model import BackendPreference, ComputeRequest, PrecisionPolicy, ResourceBudget
from compute.planner import plan_compute


class LifetimePipelineTests(unittest.TestCase):
    def test_single_exponential_tiles_return_lifetime_and_r_squared(self):
        times = np.linspace(0, 4, 80)
        trace = 5.0 * np.exp(-times / 1.25) + 0.2
        source = np.broadcast_to(trace[:, None, None], (80, 2, 3)).copy()
        request = ComputeRequest(
            "lifetime-test", 0, "lifetime_single", "source",
            source.shape, source.dtype, "THW", source,
            parameters={
                "output_shape": source.shape[1:],
                "workspace_bytes_per_spatial_item": source.shape[0] * 16,
                "max_spatial_items": 2,
            },
            backend=BackendPreference.CPU,
            precision=PrecisionPolicy.COMPATIBILITY,
        )
        plan = plan_compute(
            request, ResourceBudget(host_limit_bytes=1024 * 1024)
        )
        result = run_lifetime_pipeline(
            plan,
            data_type="central positive",
            time_points=times,
            fit_params={
                "from_start_cal": True,
                "r_squared_min": 0.4,
                "peak_range": (0, 50),
                "tau_range": (1e-3, 1e2),
            },
        )
        np.testing.assert_allclose(result.lifetime_map, 1.25, rtol=1e-3)
        self.assertTrue(np.all(result.r_squared_map > 0.999))

    def test_spawned_workers_match_bounded_serial_contract(self):
        times = np.linspace(0, 3, 48)
        trace = 4.0 * np.exp(-times / 0.8) + 0.1
        source = np.broadcast_to(trace[:, None, None], (48, 1, 2)).copy()
        request = ComputeRequest(
            "lifetime-mp-test", 0, "lifetime_single", "source",
            source.shape, source.dtype, "THW", source,
            parameters={
                "output_shape": source.shape[1:],
                "workspace_bytes_per_spatial_item": source.shape[0] * 16,
                "max_spatial_items": 1,
            },
            backend=BackendPreference.CPU,
            precision=PrecisionPolicy.COMPATIBILITY,
        )
        plan = plan_compute(request, ResourceBudget(host_limit_bytes=1024 * 1024))
        result = run_lifetime_pipeline(
            plan,
            data_type="central positive",
            time_points=times,
            fit_params={
                "from_start_cal": True,
                "r_squared_min": 0.4,
                "peak_range": (0, 50),
                "tau_range": (1e-3, 1e2),
            },
            cpu_workers=2,
        )
        np.testing.assert_allclose(result.lifetime_map, 0.8, rtol=1e-3)
        self.assertTrue(np.all(result.r_squared_map > 0.999))
    def test_preconvolution_tiles_match_full_frame_at_internal_seams(self):
        source = np.arange(3 * 6 * 7, dtype=np.float64).reshape(3, 6, 7)
        request = ComputeRequest(
            "halo-test", 0, "lifetime_single", "source",
            source.shape, source.dtype, "THW", source,
            parameters={
                "output_shape": source.shape[1:],
                "workspace_bytes_per_spatial_item": source.shape[0] * 16,
                "max_spatial_items": 4,
            },
            backend=BackendPreference.CPU,
            precision=PrecisionPolicy.COMPATIBILITY,
        )
        plan = plan_compute(request, ResourceBudget(host_limit_bytes=1024 * 1024))
        kernel = spatial_kernel("smooth", 2)
        reconstructed = np.empty_like(source)
        for tile in _tiles(plan):
            row_start, row_stop, column_start, column_stop = tile
            reconstructed[:, row_start:row_stop, column_start:column_stop] = _prepare_block(
                source, tile, kernel
            )
        from scipy.ndimage import convolve
        expected = np.stack([convolve(frame, kernel, mode="mirror") for frame in source])
        np.testing.assert_allclose(reconstructed, expected)
    def test_spatial_kernel_matches_expected_three_by_three_smooth(self):
        np.testing.assert_allclose(
            spatial_kernel("smooth", 2),
            np.array([[0.1, 0.1, 0.1], [0.1, 0.2, 0.1], [0.1, 0.1, 0.1]]),
        )


if __name__ == "__main__":
    unittest.main()
