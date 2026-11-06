import sys
import unittest
from pathlib import Path

import numpy as np

CORE = Path(__file__).resolve().parents[1] / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from compute.algorithms.stft import stft_frequency_trace
from compute.backends.cpu import stft_block_cpu


class StftPrecisionTests(unittest.TestCase):
    def test_legacy_trace_output_retains_float32_default(self):
        times = np.arange(128) / 64
        result = stft_frequency_trace(
            np.cos(2 * np.pi * 8 * times), fs=64, window=np.ones(32),
            nperseg=32, noverlap=0, nfft=32, target_freq=8.0,
        )
        self.assertEqual(result[2].dtype, np.dtype("float32"))

    def test_double_backend_retains_sub_float32_amplitude_differences(self):
        times = np.arange(128) / 64
        amplitudes = 1.0 + np.arange(6).reshape(2, 3) * 1e-9 + 1.2345e-10
        for dtype in ("float64", "complex128"):
            with self.subTest(dtype=dtype):
                values = (
                    np.exp(2j * np.pi * 8 * times)
                    if dtype == "complex128"
                    else np.cos(2 * np.pi * 8 * times)
                )
                source = (values[:, None, None] * amplitudes).astype(dtype)
                output, _frequencies, _times, _selected = stft_block_cpu(
                    source,
                    {"fps": 64, "window": np.ones(32), "window_size": 32,
                     "noverlap": 0, "nfft": 32, "target_freq": 8.0},
                    compute_dtype=dtype,
                    output_dtype="float64",
                )
                self.assertEqual(output.dtype, np.dtype("float64"))
                np.testing.assert_allclose(
                    output[1:-1], np.broadcast_to(amplitudes, output[1:-1].shape),
                    rtol=1e-13, atol=1e-14,
                )


if __name__ == "__main__":
    unittest.main()
