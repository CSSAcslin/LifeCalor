import sys
import unittest
from pathlib import Path

import numpy as np
import pywt

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from compute.algorithms.cwt import (
    SUPPORTED_CWT_WAVELETS,
    build_cwt_kernel_bank,
    cwt_coefficients_from_kernels,
    cwt_quality_spectrum,
    normalize_cwt_wavelet,
)
from compute.algorithms.lifetime import (
    LIFETIME_SOLVER_CONTRACT,
    LifetimeFitStatus,
    double_exponential,
    fit_lifetime,
    lifetime_model_and_jacobian,
)
from compute.algorithms.lifetime_pipeline import run_lifetime_pipeline
from compute.model import BackendPreference, ComputeRequest, PrecisionPolicy, ResourceBudget
from compute.planner import plan_compute


FIT_PARAMS = {
    "from_start_cal": True,
    "r_squared_min": 0.4,
    "peak_range": (0, 1000),
    "tau_range": (1e-3, 100.0),
}


class Compute111StageAContractTests(unittest.TestCase):
    def test_quality_spectrum_preserves_scale_axis_and_matches_pywavelets(self):
        fps = 64
        values = np.sin(2 * np.pi * 8 * np.arange(64) / fps).astype(np.float32)
        scales = np.asarray([1.5, 2.5, 4.0])
        expected, frequencies = pywt.cwt(
            values, scales, "morl", sampling_period=1.0 / fps
        )

        actual, actual_frequencies = cwt_quality_spectrum(
            values,
            scales,
            "morl",
            fps=fps,
            compute_dtype="float32",
            output_dtype="float32",
        )

        self.assertEqual(actual.shape, (3, 64))
        np.testing.assert_allclose(actual, np.abs(expected), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(actual_frequencies, frequencies)

    def test_supported_cwt_wavelets_match_pywavelets_reference_kernels(self):
        time = np.arange(96, dtype=np.float64) / 96.0
        values = np.sin(2.0 * np.pi * 9.0 * time)
        scales = np.asarray([2.5, 5.0, 9.0])

        for wavelet in SUPPORTED_CWT_WAVELETS:
            with self.subTest(wavelet=wavelet):
                bank = build_cwt_kernel_bank(
                    scales, wavelet, values.dtype, precision=10
                )
                actual = cwt_coefficients_from_kernels(values, bank)
                expected, frequencies = pywt.cwt(values, scales, wavelet)
                np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-12)
                np.testing.assert_allclose(bank.frequencies, frequencies)
                self.assertEqual(bank.pywavelets_version, pywt.__version__)
                self.assertTrue(all(not kernel.flags.writeable for kernel in bank.kernels))

    def test_cwt_wavelet_name_is_trimmed_but_unknown_names_are_rejected(self):
        self.assertEqual(normalize_cwt_wavelet("cmor8-3 "), "cmor8-3")
        with self.assertRaisesRegex(ValueError, "未验证"):
            normalize_cwt_wavelet("cmor2-1")

    def test_cwt_kernel_reference_freezes_real_and_complex_dtypes(self):
        base = np.sin(np.linspace(0.0, 4.0 * np.pi, 64))
        for dtype in (np.float32, np.float64, np.complex64, np.complex128):
            with self.subTest(dtype=np.dtype(dtype).name):
                values = base.astype(dtype)
                if np.dtype(dtype).kind == "c":
                    values = values + 0.25j * np.cos(
                        np.linspace(0.0, 4.0 * np.pi, 64)
                    ).astype(dtype)
                bank = build_cwt_kernel_bank([3.0, 6.0], "morl", dtype)
                actual = cwt_coefficients_from_kernels(values, bank)
                expected, _ = pywt.cwt(values, [3.0, 6.0], "morl")
                self.assertEqual(actual.dtype, expected.dtype)
                np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-7)

    def test_cwt_kernel_reference_preserves_pywavelets_nonfinite_behavior(self):
        values = np.sin(np.linspace(0.0, 2.0 * np.pi, 64))
        values[20] = np.nan
        bank = build_cwt_kernel_bank([4.0], "morl", values.dtype)
        actual = cwt_coefficients_from_kernels(values, bank)
        expected, _ = pywt.cwt(values, [4.0], "morl")
        np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
        np.testing.assert_allclose(
            actual[np.isfinite(expected)],
            expected[np.isfinite(expected)],
            rtol=1e-12,
            atol=1e-12,
        )

    def test_lifetime_analytic_jacobian_matches_finite_difference(self):
        time = np.linspace(0.0, 8.0, 41)
        cases = {
            "single": np.asarray([7.0, 2.5, 0.4]),
            "double": np.asarray([8.0, 1.8, 15.0, 11.0, 0.3]),
        }
        for model_type, parameters in cases.items():
            with self.subTest(model_type=model_type):
                model, jacobian = lifetime_model_and_jacobian(
                    time, parameters, model_type=model_type
                )
                numerical = np.empty_like(jacobian)
                for index in range(parameters.size):
                    step = 1e-6 * max(1.0, abs(float(parameters[index])))
                    upper = parameters.copy()
                    lower = parameters.copy()
                    upper[index] += step
                    lower[index] -= step
                    upper_model, _ = lifetime_model_and_jacobian(
                        time, upper, model_type=model_type
                    )
                    lower_model, _ = lifetime_model_and_jacobian(
                        time, lower, model_type=model_type
                    )
                    numerical[:, index] = (upper_model - lower_model) / (2.0 * step)
                self.assertEqual(model.shape, time.shape)
                np.testing.assert_allclose(jacobian, numerical, rtol=2e-5, atol=1e-7)

    def test_double_fit_moves_initial_values_inside_existing_bounds(self):
        times = np.linspace(0.0, 20.0, 240)
        values = double_exponential(times, 8.0, 2.0, 15.0, 12.0, 0.5)

        result = fit_lifetime(
            "sif", values, times, FIT_PARAMS, model_type="double"
        )

        self.assertEqual(result.status, LifetimeFitStatus.SUCCESS)
        np.testing.assert_allclose(result.parameters, [8.0, 2.0, 15.0, 12.0, 0.5], rtol=1e-3)
        self.assertEqual(LIFETIME_SOLVER_CONTRACT["method"], "trf")
        self.assertEqual(LIFETIME_SOLVER_CONTRACT["compute_dtype"], "float64")

    def test_double_lifetime_pipeline_returns_named_maps_and_status(self):
        times = np.linspace(0.0, 20.0, 240)
        trace = double_exponential(times, 8.0, 2.0, 15.0, 12.0, 0.5)
        source = np.broadcast_to(trace[:, None, None], (times.size, 1, 2)).copy()
        request = ComputeRequest(
            "double-map-test",
            0,
            "lifetime_double",
            "source",
            source.shape,
            source.dtype,
            "THW",
            source,
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
            data_type="sif",
            time_points=times,
            fit_params=FIT_PARAMS,
            model_type="double",
        )

        self.assertEqual(
            set(result.named_outputs),
            {
                "tau1_map",
                "tau2_map",
                "amplitude1_map",
                "amplitude2_map",
                "baseline_map",
                "r_squared_map",
                "fit_status",
            },
        )
        np.testing.assert_allclose(result.named_outputs["tau1_map"], 2.0, rtol=1e-3)
        np.testing.assert_allclose(result.named_outputs["tau2_map"], 12.0, rtol=1e-3)
        np.testing.assert_array_equal(
            result.named_outputs["fit_status"], int(LifetimeFitStatus.SUCCESS)
        )
        self.assertIs(result.lifetime_map, result.named_outputs["tau1_map"])


if __name__ == "__main__":
    unittest.main()
