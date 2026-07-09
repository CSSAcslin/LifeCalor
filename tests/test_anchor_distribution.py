import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from DataProcessor import DataProcessor


class AnchorDistributionTests(unittest.TestCase):
    def test_value_distribution_uses_current_frame_values_inside_mask(self):
        frame = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        mask = np.array([[True, False, True], [False, True, False]])

        plot_data, metadata = DataProcessor.value_distribution_from_frame(frame, mask, bins=2)

        self.assertEqual(plot_data.shape, (2, 2))
        self.assertEqual(int(plot_data[:, 1].sum()), 3)
        self.assertEqual(metadata["pixel_count"], 3)
        self.assertEqual(metadata["min"], 1.0)
        self.assertEqual(metadata["max"], 5.0)
        self.assertEqual(metadata["value_mode"], "raw")

    def test_value_distribution_uses_complex_magnitude(self):
        frame = np.array([[3 + 4j, 1 + 0j], [0 + 2j, 8 + 0j]], dtype=np.complex64)
        mask = np.array([[True, False], [True, False]])

        _, metadata = DataProcessor.value_distribution_from_frame(frame, mask, bins=2)

        self.assertEqual(metadata["value_mode"], "abs_complex")
        self.assertEqual(metadata["min"], 2.0)
        self.assertEqual(metadata["max"], 5.0)

    def test_value_distribution_rejects_empty_roi(self):
        frame = np.ones((2, 2), dtype=np.float32)
        mask = np.zeros((2, 2), dtype=bool)

        with self.assertRaisesRegex(ValueError, "ROI 内没有可统计的像素"):
            DataProcessor.value_distribution_from_frame(frame, mask)

    def test_value_distribution_accepts_manual_value_range(self):
        frame = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.float32)
        mask = np.ones((2, 3), dtype=bool)

        plot_data, metadata = DataProcessor.value_distribution_from_frame(frame, mask, bins=5, value_range=(0, 10))

        self.assertEqual(plot_data.shape, (5, 2))
        self.assertEqual(metadata["range"], (0.0, 10.0))
        self.assertEqual(int(plot_data[:, 1].sum()), 6)

    def test_distribution_frame_supports_plain_two_dimensional_data_objects(self):
        class PlainData:
            data_origin = np.array([[1, 2], [3, 4]], dtype=np.float32)

        frame = DataProcessor._distribution_frame(PlainData(), 99)

        np.testing.assert_array_equal(frame, PlainData.data_origin)

    def test_anchor_distribution_ui_and_signal_are_wired(self):
        image_source = (CORE / "ImageDisplayWindow.py").read_text(encoding="utf-8")
        binder_source = (CORE / "display" / "canvas_signals.py").read_text(encoding="utf-8")
        plot_source = (CORE / "PlotGraphWidget.py").read_text(encoding="utf-8")

        self.assertIn("'value_distribution'", image_source)
        self.assertIn("值分布统计", image_source)
        self.assertIn("get_value_distribution = pyqtSignal", image_source)
        self.assertIn("if self.anchor_active and self.args_dict['anchor_select']", image_source)
        self.assertIn("-frame{frame_index}-值分布", image_source)
        self.assertIn("(canvas.get_value_distribution, proc_thread.get_value_distribution)", binder_source)
        self.assertIn("hist_precomputed", plot_source)
        self.assertIn("precomputed_histogram_items", plot_source)

    def test_other_methods_expose_roi_distribution_entry(self):
        main_source = (CORE / "MainWindow.py").read_text(encoding="utf-8")
        extra_source = (CORE / "ExtraDialog.py").read_text(encoding="utf-8")

        self.assertIn('self.roi_distribution_btn = QPushButton("选区分布统计")', main_source)
        self.assertIn("self.roi_distribution_btn.clicked.connect(self.process_roi_value_distribution)", main_source)
        self.assertIn("roi_value_distribution_signal", main_source)
        self.assertIn("get_roi_value_distribution", main_source)
        self.assertIn("class ValueDistributionDialog", extra_source)
        self.assertIn("统计范围必须是数字，支持科学计数法", extra_source)


if __name__ == "__main__":
    unittest.main()
