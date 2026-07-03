import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from display.source import DisplaySource
from display.renderer import FrameRenderParams
from display.worker import FrameRenderWorker


class FrameRenderWorkerTests(unittest.TestCase):
    def test_worker_emits_rendered_frame_with_request_id(self):
        source = DisplaySource(
            source_id="worker-source",
            source_type="Data",
            source_name="demo",
            source_format="image_import",
            array=np.arange(8, dtype=np.float32).reshape(2, 2, 2),
        )
        worker = FrameRenderWorker(cache_capacity=2)
        results = []
        worker.rendered.connect(lambda request_id, rendered: results.append((request_id, rendered)))

        worker.render(42, source, 1, FrameRenderParams(use_colormap=False, auto_range=True))

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], 42)
        self.assertEqual(results[0][1].mode, "L")
        np.testing.assert_array_equal(results[0][1].image, np.array([[0, 85], [170, 255]], dtype=np.uint8))

    def test_worker_emits_failure_for_bad_frame_request(self):
        source = DisplaySource(
            source_id="worker-source-bad",
            source_type="Data",
            source_name="demo",
            source_format="image_import",
            array=np.zeros((1, 2, 2), dtype=np.float32),
        )
        worker = FrameRenderWorker(cache_capacity=2)
        failures = []
        worker.failed.connect(lambda request_id, message: failures.append((request_id, message)))

        worker.render(7, source, 5, FrameRenderParams(use_colormap=False, auto_range=True))

        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0][0], 7)
        self.assertIn("frame index", failures[0][1])


if __name__ == "__main__":
    unittest.main()
