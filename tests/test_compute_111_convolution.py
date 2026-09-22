import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from compute.algorithms.convolution_pipeline import (
    convolution_halo,
    run_spatiotemporal_convolution,
)


class ConvolutionPipelineTests(unittest.TestCase):
    def test_2d_asymmetric_kernel_matches_scipy_reference_across_tiles(self):
        source = np.arange(63, dtype=np.float32).reshape(7, 9)
        kernel = np.asarray([[1, 0, -1], [2, 0, -2]], dtype=np.float32)
        origin = (-1, 0)

        with tempfile.TemporaryDirectory() as directory:
            result = run_spatiotemporal_convolution(
                source, kernel, core_shape=(3, 4), boundary="reflect",
                origin=origin, cache_dir=directory,
            )

        expected = ndimage.convolve(source, kernel, mode="reflect", origin=origin)
        np.testing.assert_allclose(result, expected, rtol=0, atol=1e-6)

    def test_thw_mean_kernel_matches_reference_and_preserves_shape(self):
        source = np.arange(6 * 7 * 8, dtype=np.float32).reshape(6, 7, 8)
        kernel = np.ones((3, 2, 3), dtype=np.float32) / 18.0
        origin = (0, -1, 0)

        with tempfile.TemporaryDirectory() as directory:
            result = run_spatiotemporal_convolution(
                source, kernel, core_shape=(2, 3, 4), boundary="constant",
                cval=-2.5, origin=origin, cache_dir=directory,
            )

        expected = ndimage.convolve(
            source, kernel, mode="constant", cval=-2.5, origin=origin
        )
        self.assertEqual(result.shape, source.shape)
        np.testing.assert_allclose(result, expected, rtol=0, atol=1e-5)

    def test_halo_uses_kernel_shape_and_origin(self):
        before, after = convolution_halo((2, 4), origin=(-1, 1))
        self.assertEqual(before, (1, 0))
        self.assertEqual(after, (0, 3))

    def test_invalid_kernel_dimensionality_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "维数"):
            run_spatiotemporal_convolution(
                np.ones((4, 4)), np.ones((3, 3, 3)), core_shape=(2, 2)
            )


if __name__ == "__main__":
    unittest.main()
