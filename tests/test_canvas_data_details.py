import os
import sys
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
CORE = Path(__file__).resolve().parents[1] / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from PyQt5.QtWidgets import QApplication, QWidget
from display.data_details import CanvasDataDetailsDialog


class FakeCanvas(QWidget):
    def __init__(self, source):
        super().__init__()
        image = np.zeros((4, 8, 9), dtype=np.float32)
        self.data = SimpleNamespace(
            parent_data=weakref.ref(source), source_name="sample", source_format="unit",
            imageshape=image.shape, datatype=image.dtype, totalframes=4,
            fps=20, timestamp_inherited=1.0, ROI_applied=False,
        )
        self.id = 2
        self.current_time_idx = 1
        self.is_temporal = True
        self.colormap = None
        self.use_colormap = False
        self.min_value = 0.0
        self.max_value = 1.0
        self.is_sync_enabled = False
        self.render_status = "idle"

    def windowTitle(self):
        return "sample"


class CanvasDataDetailsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_dialog_is_metadata_only_and_has_four_views(self):
        class NoScan(np.ndarray):
            def min(self, *args, **kwargs):
                raise AssertionError("details scanned array")
            def max(self, *args, **kwargs):
                raise AssertionError("details scanned array")
        payload = np.ones((3, 4)).view(NoScan)
        class Source:
            pass
        source = Source()
        source.datashape = (4, 8, 9)
        source.datatype = np.dtype("float32")
        source.time_point = np.arange(4)
        source.parameters = {"fps": 20, "calibration": {"unit": "nm"}}
        source.out_processed = {"map": payload}
        canvas = FakeCanvas(source)
        dialog = CanvasDataDetailsDialog(canvas, canvas)
        self.addCleanup(dialog.close)
        self.addCleanup(canvas.close)
        self.assertEqual(dialog.tabs.count(), 4)
        self.assertIn("视频", dialog.summary.text())
        self.assertIn("shape=(4, 8, 9)", dialog.summary.text())


if __name__ == "__main__":
    unittest.main()