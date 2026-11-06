import os
import sys
import tempfile
import unittest
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy import signal

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
CORE = Path(__file__).resolve().parents[1] / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from PyQt5.QtCore import QCoreApplication, QEvent
from PyQt5.QtWidgets import QApplication, QMenu

from PlotGraphWidget import PlotGraphWidget


class CurveAnalysisActionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.widget = PlotGraphWidget()

    def tearDown(self):
        self.widget.close()
        self.widget.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)

    def add_curve(self, x_values, y_values, **options):
        self.widget.plot_data(np.column_stack((x_values, y_values)), **options)
        return self.widget.data_items[-1]

    @contextmanager
    def physical_picosecond_source(self):
        import DataManager
        from ArrayCache import ArrayCacheConfig

        seconds = np.arange(128) / 128
        samples = np.sin(2 * np.pi * 16 * seconds)
        values = np.broadcast_to(samples[:, None, None], (128, 2, 3)).copy()
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(DataManager, "_ARRAY_CACHE_CONFIG", ArrayCacheConfig(Path(directory))),
                patch.object(DataManager.Data, "history", deque(maxlen=3)),
                patch.object(DataManager.ProcessedData, "history", deque(maxlen=3)),
                patch.object(DataManager.Data, "_counter", 0),
            ):
                yield DataManager.Data(
                    values, seconds * 1e12, "npy", values,
                    parameters={"time_unit": "ps", "source_axes": "THW"},
                    name="physical-picosecond-signal",
                )

    def test_roi_statistics_preserves_physical_picosecond_units_through_psd(self):
        from DataProcessor import DataProcessor

        processor = DataProcessor()
        failures = []
        processor.processing_error_signal.connect(failures.append)
        processor.plot_singal.connect(self.widget.handle_plot_signal)
        with self.physical_picosecond_source() as source:
            processor.get_fast_selection(source, np.ones((2, 3), dtype=bool), "mean", "ROI trace")
            self.assertEqual(failures, [])
            self.assertEqual(len(self.widget.data_items), 1)
            self.assertTrue(self.widget.action_psd.isEnabled())
            self.widget.action_psd.trigger()
            item = self.widget.data_items[0]
            peak_hertz = item.xData[np.argmax(item.yData)]
            self.assertAlmostEqual(float(peak_hertz), 16.0, places=10)

    def test_spatial_average_router_preserves_physical_picosecond_units_through_psd(self):
        from LifetimeCalculator import CalculationThread
        from processing.result_router import ResultRouter

        plotted_series = []
        result_display = SimpleNamespace(
            plot_time_series=lambda axis, values: plotted_series.append((axis, values))
        )
        window = SimpleNamespace(result_display=result_display, graph_plot=self.widget)
        router = ResultRouter(window, None)
        worker = CalculationThread()
        failures = []
        worker.processing_error_signal.connect(failures.append)
        worker.processed_result.connect(router._signal_average)
        with self.physical_picosecond_source() as source:
            self.assertTrue(worker.easy_process(source, "mean"))
            self.assertEqual(failures, [])
            self.assertEqual(len(plotted_series), 1)
            self.assertEqual(len(self.widget.data_items), 1)
            self.assertTrue(self.widget.action_psd.isEnabled())
            self.widget.action_psd.trigger()
            item = self.widget.data_items[0]
            peak_hertz = item.xData[np.argmax(item.yData)]
            self.assertAlmostEqual(float(peak_hertz), 16.0, places=10)

    def test_shared_menu_actions_restore_the_original_data_and_peak_state(self):
        x_values = np.arange(64) / 64
        y_values = np.sin(2 * np.pi * 8 * x_values)
        item = self.add_curve(x_values, y_values)
        menu = QMenu(self.widget)
        for action in (self.widget.action_time, self.widget.action_psd,
                       self.widget.action_hist, self.widget.action_peak):
            menu.addAction(action)
        menu.aboutToShow.connect(self.widget.refresh_analysis_actions)

        menu.actions()[3].trigger()
        self.assertTrue(self.widget.action_peak.isChecked())
        self.assertTrue(self.widget.peak_region.isVisible())
        self.widget.action_peak.setChecked(False)
        self.assertFalse(self.widget.peak_region.isVisible())
        self.widget.action_peak.setChecked(True)
        self.assertTrue(self.widget.peak_region.isVisible())
        menu.actions()[1].trigger()
        self.assertTrue(self.widget.action_psd.isChecked())
        self.assertFalse(self.widget.action_time.isChecked())
        self.assertFalse(self.widget.action_peak.isChecked())
        self.assertFalse(self.widget.peak_region.isVisible())
        self.assertFalse(self.widget.peak_marker.isVisible())

        expected_x, expected_y = signal.welch(y_values, 64, nperseg=64)
        np.testing.assert_allclose(item.xData, expected_x)
        np.testing.assert_allclose(item.yData, expected_y)
        menu.actions()[0].trigger()
        np.testing.assert_array_equal(item.xData, x_values)
        np.testing.assert_array_equal(item.yData, y_values)
        self.assertTrue(self.widget.action_time.isChecked())

    def test_psd_converts_all_supported_time_units_to_hertz(self):
        seconds = np.arange(128) / 128
        amplitude = np.sin(2 * np.pi * 16 * seconds)
        expected_x, expected_y = signal.welch(amplitude, 128, nperseg=128)
        for unit, power in self.widget.unit_powers.items():
            with self.subTest(unit=unit):
                self.widget.clear_all()
                item = self.add_curve(seconds / 10.0 ** power, amplitude, time_unit=unit)
                self.assertTrue(self.widget.action_psd.isEnabled())
                self.widget.action_psd.trigger()
                np.testing.assert_allclose(item.xData, expected_x)
                np.testing.assert_allclose(item.yData, expected_y, atol=1e-15)

    def test_curves_with_different_units_share_one_time_basis(self):
        seconds = np.arange(32) / 32
        values = np.sin(2 * np.pi * seconds)
        first = self.add_curve(seconds * 1000, values, time_unit="ms")
        second = self.add_curve(seconds, values, time_unit="s")
        np.testing.assert_allclose(self.widget.data_cache[first][0],
                                   self.widget.data_cache[second][0])
        self.widget.action_psd.trigger()
        np.testing.assert_allclose(first.xData, second.xData)
        np.testing.assert_allclose(first.yData, second.yData)

    def test_psd_rejects_invalid_sampling_without_changing_the_plot(self):
        cases = (
            ([0, 0, 1], [1, 2, 3], None, "严格递增"),
            ([2, 1, 0], [1, 2, 3], None, "严格递增"),
            ([0, 1, 3], [1, 2, 3], None, "均匀采样"),
            ([0, np.nan, 2], [1, 2, 3], None, "非有限"),
            ([0, 1, 2], [1, np.inf, 3], None, "非有限"),
            ([0], [1], None, "至少需要"),
            ([0, 1, 2], [1, 2, 3], "frames", "时间单位"),
        )
        for x_values, y_values, unit, reason in cases:
            with self.subTest(reason=reason, x_values=x_values):
                self.widget.clear_all()
                item = self.add_curve(x_values, y_values, time_unit=unit)
                original_x, original_y = item.xData.copy(), item.yData.copy()
                self.assertFalse(self.widget.action_psd.isEnabled())
                self.assertIn(reason, self.widget.action_psd.toolTip())
                self.assertFalse(self.widget.update_analysis_mode(self.widget.action_psd))
                np.testing.assert_array_equal(item.xData, original_x)
                np.testing.assert_array_equal(item.yData, original_y)
                self.assertTrue(self.widget.action_time.isChecked())

    def test_plot_wide_transform_prepares_every_curve_before_commit(self):
        x_values = np.arange(16) / 16
        first = self.add_curve(x_values, np.sin(2 * np.pi * x_values), name="first")
        second = self.add_curve(x_values, np.cos(2 * np.pi * x_values), name="second")
        originals = [(item.xData.copy(), item.yData.copy()) for item in (first, second)]
        prepared = signal.welch(originals[0][1], 16, nperseg=16)
        with patch("PlotGraphWidget.signal.welch", side_effect=[prepared, ValueError("second failed")]):
            self.widget.action_psd.trigger()
        for item, (x_original, y_original) in zip((first, second), originals):
            np.testing.assert_array_equal(item.xData, x_original)
            np.testing.assert_array_equal(item.yData, y_original)
        self.assertTrue(self.widget.action_time.isChecked())
        self.assertFalse(self.widget.action_psd.isChecked())
        self.assertIn("second failed", self.widget.action_psd.toolTip())

    def test_precomputed_histogram_keeps_counts_and_blocks_time_analysis(self):
        x_values = np.arange(4)
        counts = np.array([1, 5, 2, 1])
        item = self.add_curve(x_values, counts, analysis_mode="hist_precomputed")
        self.assertTrue(self.widget.action_hist.isChecked())
        np.testing.assert_array_equal(item.xData, x_values)
        np.testing.assert_array_equal(item.yData, counts)
        self.assertFalse(self.widget.action_psd.isEnabled())
        self.assertFalse(self.widget.action_peak.isEnabled())
        self.assertIn("预计算", self.widget.action_psd.toolTip())
        self.assertFalse(self.widget.update_analysis_mode(self.widget.action_psd))
        self.assertTrue(self.widget.action_hist.isChecked())
        self.widget.action_time.trigger()
        np.testing.assert_array_equal(item.yData, counts)

    def test_peak_uses_modern_numpy_area_and_handles_flat_or_empty_regions(self):
        self.add_curve(np.arange(5), [0, 1, 2, 1, 0], name="triangle")
        self.widget.action_peak.trigger()
        self.widget.peak_region.setRegion([0, 4])
        self.widget._update_peak_analysis()
        self.assertTrue(self.widget.peak_marker.isVisible())
        self.assertIn("Area: 4", self.widget.peak_info_label.text())
        self.assertIn("FWHM: 2", self.widget.peak_info_label.text())
        self.widget.peak_region.setRegion([10, 12])
        self.widget._update_peak_analysis()
        self.assertFalse(self.widget.peak_marker.isVisible())
        self.assertIn("区间数据过少", self.widget.peak_info_label.text())
        self.widget.clear_all()
        self.add_curve(np.arange(5), np.zeros(5))
        self.widget.action_peak.trigger()
        self.widget.peak_region.setRegion([0, 4])
        self.widget._update_peak_analysis()
        self.assertFalse(self.widget.peak_marker.isVisible())
        self.assertIn("没有可量化", self.widget.peak_info_label.text())

    def test_peak_target_and_shared_state_follow_curve_visibility(self):
        first = self.add_curve(np.arange(5), [0, 1, 2, 1, 0], name="first")
        second = self.add_curve(np.arange(5), [0, 2, 4, 2, 0], name="second")
        self.widget.action_peak.trigger()
        self.widget.peak_region.setRegion([0, 4])
        first.setVisible(False)
        self.assertIn("second", self.widget.peak_info_label.text())
        second.setVisible(False)
        self.assertFalse(self.widget.action_peak.isEnabled())
        self.assertFalse(self.widget.action_peak.isChecked())
        self.assertFalse(self.widget.peak_region.isVisible())
        first.setVisible(True)
        self.assertTrue(self.widget.action_peak.isEnabled())
        self.assertFalse(self.widget.action_peak.isChecked())

    def test_clear_resets_actions_histogram_identity_and_overlays(self):
        self.add_curve(np.arange(5), [0, 1, 2, 1, 0])
        self.widget.action_peak.trigger()
        self.widget.clear_all()
        self.assertTrue(self.widget.action_time.isChecked())
        for action in (self.widget.action_time, self.widget.action_psd,
                       self.widget.action_hist, self.widget.action_peak):
            self.assertFalse(action.isEnabled())
        self.assertFalse(self.widget.action_peak.isChecked())
        self.assertFalse(self.widget.peak_region.isVisible())
        self.assertFalse(self.widget.precomputed_histogram_items)

    def test_repeated_menu_refresh_reuses_validation_and_does_not_integrate_peaks(self):
        item = self.add_curve(np.arange(5), [0, 1, 2, 1, 0])
        self.widget.action_peak.trigger()
        self.widget.peak_region.setRegion([0, 4])
        with (
            patch.object(self.widget, "_curve_arrays", wraps=self.widget._curve_arrays) as arrays,
            patch.object(self.widget, "_psd_inputs", wraps=self.widget._psd_inputs) as psd,
            patch.object(self.widget, "_update_peak_analysis", wraps=self.widget._update_peak_analysis) as peaks,
        ):
            for _ in range(5):
                self.widget.refresh_analysis_actions()
            arrays.assert_not_called()
            psd.assert_not_called()
            peaks.assert_not_called()

            x_values, y_values = self.widget.data_cache[item]
            self.widget.data_cache[item] = (x_values.copy(), y_values.copy())
            self.widget.refresh_analysis_actions()
            self.assertGreater(arrays.call_count, 0)
            self.assertGreater(psd.call_count, 0)
            peaks.assert_not_called()
            scan_counts = (arrays.call_count, psd.call_count)
            for _ in range(5):
                self.widget.refresh_analysis_actions()
            self.assertEqual((arrays.call_count, psd.call_count), scan_counts)
            peaks.assert_not_called()

            self.widget.data_cache[item][0][1] = x_values[0]
            self.assertFalse(self.widget.update_analysis_mode(self.widget.action_psd))
            self.assertGreater(psd.call_count, scan_counts[1])
            self.assertFalse(self.widget.action_psd.isEnabled())
            np.testing.assert_array_equal(item.xData, x_values)
            np.testing.assert_array_equal(item.yData, y_values)
        self.widget.clear_all()
        self.assertFalse(self.widget._analysis_validation_cache)


if __name__ == "__main__":
    unittest.main()
