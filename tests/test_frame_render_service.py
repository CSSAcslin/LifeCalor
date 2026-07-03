import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from display.source import DisplaySource
from display.service import FrameRenderService
from display.renderer import FrameRenderParams


class FrameRenderServiceTests(unittest.TestCase):
    def test_render_source_renders_requested_frame_and_caches_result(self):
        array = np.arange(12, dtype=np.float32).reshape(3, 2, 2)
        source = DisplaySource(
            source_id="source-a",
            source_type="Data",
            source_name="demo",
            source_format="image_import",
            array=array,
        )
        service = FrameRenderService(cache_capacity=4)
        params = FrameRenderParams(use_colormap=False, auto_range=True)

        first = service.render_source(source, 2, params)
        second = service.render_source(source, 2, params)

        self.assertIs(first, second)
        self.assertEqual(first.mode, "L")
        self.assertEqual(first.image.shape, (2, 2))
        np.testing.assert_array_equal(first.image, np.array([[0, 85], [170, 255]], dtype=np.uint8))
        self.assertEqual(service.cache_size, 1)

    def test_cache_key_includes_frame_index_and_render_params(self):
        array = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
        source = DisplaySource(
            source_id="source-b",
            source_type="Data",
            source_name="demo",
            source_format="image_import",
            array=array,
        )
        service = FrameRenderService(cache_capacity=4)

        frame0 = service.render_source(source, 0, FrameRenderParams(use_colormap=False, auto_range=True))
        frame1 = service.render_source(source, 1, FrameRenderParams(use_colormap=False, auto_range=True))
        fixed = service.render_source(source, 0, FrameRenderParams(use_colormap=False, auto_range=False, min_value=0, max_value=7))

        self.assertIsNot(frame0, frame1)
        self.assertIsNot(frame0, fixed)
        self.assertEqual(service.cache_size, 3)


if __name__ == "__main__":
    unittest.main()
