import os
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from compute.algorithms.cwt import SUPPORTED_CWT_WAVELETS, cwt_quality_spectrum
from compute.algorithms.lifetime import lifetime_model_and_jacobian
from compute.backends.cpu import cwt_block_cpu
from compute.backends.cuda import (
    cuda_self_test,
    cwt_block_cuda_prototype,
    cwt_quality_trace_cuda,
    lifetime_model_jacobian_cuda_prototype,
)
from compute.backends.lifetime_cuda import lifetime_fit_block_cuda


RUN_CUDA = os.environ.get("LIFECALOR_RUN_CUDA_TESTS") == "1"


@unittest.skipUnless(
    RUN_CUDA,
    "set LIFECALOR_RUN_CUDA_TESTS=1 on the target NVIDIA machine",
)
class Compute111CudaPrototypeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = cuda_self_test(0)

    def test_cwt_cuda_prototype_matches_cpu_for_all_supported_wavelets(self):
        fps = 128
        time = np.arange(128, dtype=np.float32) / fps
        trace = np.sin(2.0 * np.pi * 12.0 * time).astype(np.float32)
        block = np.broadcast_to(trace[:, None, None], (128, 2, 2)).copy()

        for wavelet in SUPPORTED_CWT_WAVELETS:
            with self.subTest(wavelet=wavelet):
                params = {
                    "target_freq": 12.0,
                    "scale_range": 4.0,
                    "total_scales": 5,
                    "wavelet": wavelet,
                    "fps": fps,
                }
                expected, expected_scales, expected_frequencies = cwt_block_cpu(
                    block,
                    params,
                    compute_dtype="float32",
                    output_dtype="float32",
                )
                actual, scales, frequencies = cwt_block_cuda_prototype(
                    block,
                    params,
                    compute_dtype="float32",
                    output_dtype="float32",
                    device_index=0,
                )
                scale = max(float(np.max(np.abs(expected))), 1.0)
                np.testing.assert_allclose(
                    actual, expected, rtol=5e-4, atol=5e-6 * scale
                )
                np.testing.assert_allclose(scales, expected_scales)
                np.testing.assert_allclose(frequencies, expected_frequencies)

    def test_cwt_quality_cuda_preserves_the_scale_spectrum(self):
        fps = 64
        values = np.sin(2 * np.pi * 8 * np.arange(64) / fps).astype(np.float32)
        scales = np.asarray([1.5, 2.5, 4.0])
        expected, expected_frequencies = cwt_quality_spectrum(
            values, scales, "morl", fps=fps
        )
        actual, frequencies = cwt_quality_trace_cuda(
            values,
            scales,
            "morl",
            fps=fps,
            compute_dtype="float32",
            output_dtype="float32",
            device_index=0,
        )
        scale = max(float(np.max(expected)), 1.0)
        np.testing.assert_allclose(
            actual, expected, rtol=5e-4, atol=5e-6 * scale
        )
        np.testing.assert_allclose(frequencies, expected_frequencies)

    def test_lifetime_cuda_model_and_jacobian_match_cpu_reference(self):
        time = np.linspace(0.0, 12.0, 160)
        cases = {
            "single": np.asarray([8.0, 2.5, 0.4]),
            "double": np.asarray([8.0, 2.0, 15.0, 12.0, 0.5]),
        }
        for model_type, parameters in cases.items():
            with self.subTest(model_type=model_type):
                expected_model, expected_jacobian = lifetime_model_and_jacobian(
                    time, parameters, model_type=model_type
                )
                model, jacobian = lifetime_model_jacobian_cuda_prototype(
                    time,
                    parameters,
                    model_type=model_type,
                    device_index=0,
                )
                np.testing.assert_allclose(model, expected_model, rtol=1e-12, atol=1e-12)
                np.testing.assert_allclose(
                    jacobian, expected_jacobian, rtol=1e-12, atol=1e-12
                )

    def test_bounded_lifetime_cuda_solver_recovers_single_and_double_models(self):
        times = np.linspace(0.0, 30.0, 200)
        fit_params = {
            "from_start_cal": True,
            "r_squared_min": 0.4,
            "peak_range": (0, 200),
            "tau_range": (1e-3, 100.0),
        }
        single = (8 * np.exp(-times / 2.5) + 0.5).reshape(-1, 1, 1)
        single_result = lifetime_fit_block_cuda(
            single,
            times,
            None,
            fit_params,
            model_type="single",
            device_index=0,
        )
        self.assertAlmostEqual(
            float(single_result["lifetime_map"][0, 0]), 2.5, places=4
        )
        double = (
            50 * np.exp(-times / 3.0)
            + 20 * np.exp(-times / 15.0)
            + 2
        ).reshape(-1, 1, 1)
        double_result = lifetime_fit_block_cuda(
            double,
            times,
            None,
            fit_params,
            model_type="double",
            device_index=0,
        )
        np.testing.assert_allclose(
            sorted((
                double_result["tau1_map"][0, 0],
                double_result["tau2_map"][0, 0],
            )),
            (3.0, 15.0),
            rtol=1e-4,
        )


if __name__ == "__main__":
    unittest.main()
