import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from compute import ALGORITHM_CONTRACTS, compatibility_metadata
from compute.algorithms import (
    cwt_frequency_trace,
    fit_lifetime,
    has_correlated_window,
    reduce_cwt_coefficients,
    stft_frequency_trace,
)
from LifetimeCalculator import CalculationThread, LifetimeCalculator


FIT_PARAMS = {
    "from_start_cal": True,
    "r_squared_min": 0.4,
    "peak_range": (0, 1000),
    "tau_range": (1e-3, 100.0),
}


class ComputeContractTests(unittest.TestCase):
    def test_compatibility_metadata_is_serializable_and_explicit(self):
        metadata = compatibility_metadata("stft", np.float32, backend="cpu")

        self.assertEqual(metadata["precision_policy"], "compatibility")
        self.assertEqual(metadata["input_dtype"], "float32")
        self.assertEqual(metadata["output_dtype"], "float32")
        self.assertEqual(metadata["backend"], "cpu")
        self.assertTrue(ALGORITHM_CONTRACTS["stft"].accepts_complex)
        self.assertFalse(ALGORITHM_CONTRACTS["lifetime"].accepts_complex)

    def test_stft_dc_is_not_doubled(self):
        values = np.full(128, 3.0, dtype=np.float32)
        frequencies, _, magnitude, selected = stft_frequency_trace(
            values,
            fs=64,
            window=np.ones(64),
            nperseg=64,
            noverlap=0,
            nfft=64,
            target_freq=0.0,
        )

        self.assertEqual(frequencies[selected[0]], 0.0)
        self.assertAlmostEqual(float(np.max(magnitude)), 3.0, places=5)
        self.assertEqual(magnitude.dtype, np.float32)

    def test_stft_batch_and_serial_use_the_same_contract(self):
        times = np.arange(128) / 128.0
        first = np.sin(2 * np.pi * 16 * times).astype(np.float32)
        second = (0.5 * first).astype(np.float32)
        kwargs = dict(
            fs=128,
            window=np.hanning(64),
            nperseg=64,
            noverlap=32,
            nfft=64,
            target_freq=16.0,
        )

        _, serial_times, serial, serial_indices = stft_frequency_trace(first, **kwargs)
        _, batch_times, batch, batch_indices = stft_frequency_trace(
            np.stack((first, second)), **kwargs
        )

        np.testing.assert_array_equal(serial_times, batch_times)
        np.testing.assert_array_equal(serial_indices, batch_indices)
        np.testing.assert_allclose(batch[0], serial, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(batch[1], serial * 0.5, rtol=1e-6, atol=1e-6)

    def test_stft_complex_signal_can_select_negative_frequency(self):
        times = np.arange(128) / 64.0
        values = np.exp(-2j * np.pi * 8 * times).astype(np.complex64)
        frequencies, _, magnitude, selected = stft_frequency_trace(
            values,
            fs=64,
            window=np.ones(64),
            nperseg=64,
            noverlap=0,
            nfft=64,
            target_freq=-8.0,
        )

        self.assertAlmostEqual(float(frequencies[selected[0]]), -8.0)
        self.assertGreater(float(np.max(magnitude)), 0.99)

    def test_cwt_scale_reduction_normalizes_before_mean(self):
        coefficients = np.asarray([[1.0, 2.0], [3.0, 4.0]])
        scales = np.asarray([1.0, 4.0])
        expected = np.mean(
            2.0 * np.abs(coefficients) / np.sqrt(scales[:, None]), axis=0
        )

        result = reduce_cwt_coefficients(coefficients, scales)

        np.testing.assert_allclose(result, expected.astype(np.float32))
        self.assertEqual(result.shape, (2,))

    def test_cwt_multiple_scales_returns_one_time_trace(self):
        values = np.sin(np.linspace(0, 8 * np.pi, 128)).astype(np.float32)
        result, scales, frequencies = cwt_frequency_trace(
            values,
            target_freq=10.0,
            scale_range=4.0,
            total_scales=5,
            wavelet="morl",
            fps=128,
        )

        self.assertEqual(result.shape, values.shape)
        self.assertEqual(result.dtype, np.float32)
        self.assertEqual(scales.shape, (5,))
        self.assertEqual(frequencies.shape, (5,))

    def test_single_exponential_reference_recovers_tau_and_r_squared(self):
        times = np.linspace(0.0, 12.0, 160)
        values = 8.0 * np.exp(-times / 2.5) + 1.25

        result = fit_lifetime("sif", values, times, FIT_PARAMS, model_type="single")

        self.assertAlmostEqual(float(result.lifetime), 2.5, places=3)
        self.assertGreater(result.r_squared, 0.999999)

    def test_double_exponential_reference_recovers_both_taus(self):
        times = np.linspace(0.0, 60.0, 400)
        values = 50.0 * np.exp(-times / 3.0) + 20.0 * np.exp(-times / 15.0) + 2.0

        result = fit_lifetime("sif", values, times, FIT_PARAMS, model_type="double")

        np.testing.assert_allclose(sorted(result.lifetime), [3.0, 15.0], rtol=1e-3)
        self.assertGreater(result.r_squared, 0.999999)

    def test_lifetime_rejects_complex_observations(self):
        times = np.arange(8, dtype=np.float64)
        with self.assertRaisesRegex(ValueError, "real-valued"):
            fit_lifetime(
                "sif",
                np.ones(8, dtype=np.complex64),
                times,
                FIT_PARAMS,
            )

    def test_correlated_window_accepts_a_valid_decay(self):
        times = np.arange(12, dtype=np.float64)
        values = np.exp(-times / 3.0)
        self.assertTrue(has_correlated_window(values, times))

    def test_lifetime_map_fits_once_and_stores_real_r_squared(self):
        worker = CalculationThread()
        worker._is_calculating = True
        times = np.arange(12, dtype=np.float64)
        values = np.exp(-times / 3.0).reshape(12, 1, 1)
        expected = (np.zeros(3), 2.5, 0.91, values[:, 0, 0])

        with patch.object(
            LifetimeCalculator,
            "calculate_lifetime",
            return_value=expected,
        ) as calculate:
            lifetime_map, r_squared_map = worker.lifetime_map_cal(
                values, "sif", times, "single"
            )

        self.assertEqual(calculate.call_count, 1)
        self.assertEqual(lifetime_map[0, 0], 2.5)
        self.assertEqual(r_squared_map[0, 0], 0.91)
        self.assertEqual(calculate.call_args.kwargs["model_type"], "single")

    def test_legacy_single_map_helper_rejects_ambiguous_double_output(self):
        worker = CalculationThread()
        worker._is_calculating = True
        result = worker.lifetime_map_cal(
            np.ones((8, 1, 1)), "sif", np.arange(8), "double"
        )
        self.assertIsInstance(result, ValueError)
        self.assertIn("具名多结果", str(result))


if __name__ == "__main__":
    unittest.main()
