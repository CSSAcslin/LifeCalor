import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

CORE = Path(__file__).resolve().parents[1] / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from DataManager import DataManager


class StyledAviExportTests(unittest.TestCase):
    def test_colormap_avi_is_written_frame_by_frame_as_color(self):
        source = np.arange(3 * 16 * 16, dtype=np.float32).reshape(3, 16, 16)
        params = {
            "use_colormap": True,
            "colormap": "viridis",
            "auto_range": False,
            "min_value": float(source.min()),
            "max_value": float(source.max()),
        }
        with tempfile.TemporaryDirectory() as directory:
            files = DataManager().export_as_avi(source, directory, "styled", 1, params)
            capture = cv2.VideoCapture(files[0])
            try:
                self.assertTrue(capture.isOpened())
                self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 3)
                ok, frame = capture.read()
                self.assertTrue(ok)
                self.assertEqual(frame.shape, (16, 16, 3))
            finally:
                capture.release()


if __name__ == "__main__":
    unittest.main()
