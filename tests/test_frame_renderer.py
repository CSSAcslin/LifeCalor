import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from FrameRenderer import FrameRenderParams, FrameRenderer


class FrameRendererTests(unittest.TestCase):
    def test_render_grayscale_float_frame_to_uint8(self):
        frame = np.array([[0.0, 0.5], [1.0, 2.0]], dtype=np.float32)
        rendered = FrameRenderer.render(frame, FrameRenderParams(use_colormap=False, min_value=0.0, max_value=2.0))

        self.assertEqual(rendered.mode, "L")
        self.assertEqual(rendered.image.dtype, np.uint8)
        np.testing.assert_array_equal(rendered.image, np.array([[0, 63], [127, 255]], dtype=np.uint8))

    def test_render_complex_frame_uses_magnitude(self):
        frame = np.array([[3 + 4j, 0 + 0j]], dtype=np.complex64)
        rendered = FrameRenderer.render(frame, FrameRenderParams(use_colormap=False, min_value=0.0, max_value=5.0))

        np.testing.assert_array_equal(rendered.image, np.array([[255, 0]], dtype=np.uint8))

    def test_render_colormap_outputs_rgba(self):
        frame = np.array([[0.0, 1.0]], dtype=np.float32)
        rendered = FrameRenderer.render(frame, FrameRenderParams(use_colormap=True, colormap="gray", min_value=0.0, max_value=1.0))

        self.assertEqual(rendered.mode, "RGBA")
        self.assertEqual(rendered.image.shape, (1, 2, 4))
        self.assertEqual(rendered.image.dtype, np.uint8)

    def test_nan_and_inf_do_not_break_render(self):
        frame = np.array([[np.nan, np.inf, -np.inf, 1.0]], dtype=np.float32)
        rendered = FrameRenderer.render(frame, FrameRenderParams(use_colormap=False, min_value=0.0, max_value=1.0))

        self.assertEqual(rendered.image.dtype, np.uint8)
        self.assertEqual(rendered.image.shape, (1, 4))


    def test_render_custom_rainbow_colormap_outputs_rgba(self):
        frame = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
        rendered = FrameRenderer.render(frame, FrameRenderParams(use_colormap=True, colormap="Rainbow*", min_value=0.0, max_value=1.0))

        self.assertEqual(rendered.mode, "RGBA")
        self.assertEqual(rendered.image.shape, (1, 3, 4))
        self.assertFalse(np.array_equal(rendered.image[..., 0], rendered.image[..., 1]))

    def test_unknown_colormap_falls_back_to_jet_rgba(self):
        frame = np.array([[0.0, 1.0]], dtype=np.float32)
        rendered = FrameRenderer.render(frame, FrameRenderParams(use_colormap=True, colormap="missing-cmap", min_value=0.0, max_value=1.0))

        self.assertEqual(rendered.mode, "RGBA")
        self.assertEqual(rendered.image.shape, (1, 2, 4))

if __name__ == "__main__":
    unittest.main()
