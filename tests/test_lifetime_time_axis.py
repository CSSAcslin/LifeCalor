import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
CORE = Path(__file__).resolve().parents[1] / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from DataManager import Data, ProcessedData
from LifetimeCalculator import CalculationThread, LifetimeCalculator
from processing.time_axis import scientific_time_axis


class LifetimeTimeAxisTests(unittest.TestCase):
    def setUp(self):
        Data.history.clear()
        ProcessedData.history.clear()
        self.fit_settings = patch.object(LifetimeCalculator, "_cal_params", {
            "from_start_cal": True, "r_squared_min": 0.4,
            "peak_range": (0, 50), "tau_range": (0.001, 100),
        })
        self.fit_settings.start()
        self.addCleanup(self.fit_settings.stop)
        self.cache = tempfile.TemporaryDirectory()
        self.addCleanup(self.cache.cleanup)

    def tearDown(self):
        Data.history.clear()
        ProcessedData.history.clear()

    def source(self, source_format="tiff", *, physical=True):
        physical_times = np.linspace(0, 20, 41)
        values = (10 * np.exp(-physical_times / 5) + 1)[:, None, None]
        metadata = {"source_axes": "THW", "time_unit": "ps", "time_step": 0.5}
        if physical:
            metadata["time_axis_kind"] = "physical"
        axis = physical_times if physical else np.arange(len(values), dtype=float)
        return Data(values, axis, source_format, values.copy(), metadata, "known-tau5")

    def region_result(self, source):
        worker = CalculationThread()
        results, errors = [], []
        worker.processed_result.connect(results.append)
        worker.processing_error_signal.connect(errors.append)
        worker.region_analyze(source, 0.5, np.ones((1, 1), dtype=bool), "single")
        self.assertEqual(errors, [])
        self.assertEqual(len(results), 1)
        return results[0]

    def map_result(self, source, method):
        worker = CalculationThread()
        results, errors = [], []
        worker.processed_result.connect(results.append)
        worker.processing_error_signal.connect(errors.append)
        getattr(worker, method)(source, 0.5, "single", compute_options={
            "backend": "cpu", "cache_directory": self.cache.name,
        })
        self.assertEqual(errors, [])
        self.assertEqual(len(results), 1)
        return results[0]

    def test_physical_region_lifetime_is_not_scaled_by_ui_step(self):
        source = self.source()
        result = self.region_result(source)
        self.assertAlmostEqual(result.out_processed["lifetime"], 5, places=6)
        np.testing.assert_allclose(result.time_point, source.time_point)
        self.assertEqual(result.out_processed["time_axis_kind"], "physical")
        self.assertEqual(result.out_processed["time_axis_unit"], "ps")

    def test_registered_importers_without_explicit_physical_marker_keep_their_axis(self):
        for source_format in ("npy", "tiff", "video", "hdf5", "sif"):
            with self.subTest(source_format=source_format):
                source = self.source(source_format)
                source.parameters.pop("time_axis_kind")
                self.assertAlmostEqual(self.region_result(source).out_processed["lifetime"], 5, places=6)

    def test_physical_distribution_lifetime_is_not_scaled_by_ui_step(self):
        result = self.map_result(self.source(), "distribution_analyze")
        np.testing.assert_allclose(result.data_processed, [[5]], rtol=1e-6)
        self.assertEqual(result.out_processed["time_unit"], "ps")

    def test_physical_heat_transfer_uses_unscaled_lifetime(self):
        result = self.map_result(self.source(), "heat_transfer_calculation")
        np.testing.assert_allclose(result.data_processed, [[42.72 / 5]], rtol=1e-6)

    def test_existing_lifetime_map_is_not_refitted_or_rescaled_for_heat(self):
        result = self.map_result(self.source(), "distribution_analyze")
        heat = self.map_result(result, "heat_transfer_calculation")
        np.testing.assert_allclose(heat.data_processed, [[42.72 / 5]], rtol=1e-6)

    def test_legacy_frames_keep_the_ui_step_and_result_is_marked_physical(self):
        for source_format in ("tif_series", "sif", "unknown-old-format"):
            with self.subTest(source_format=source_format):
                source = self.source(source_format, physical=False)
                source.parameters["time_step"] = 99  # Historic fitting uses the explicit UI step.
                if source_format == "sif":
                    source.parameters["data_type"] = "sif_folder"
                result = self.region_result(source)
                self.assertAlmostEqual(result.out_processed["lifetime"], 5, places=6)
                np.testing.assert_allclose(result.time_point, np.linspace(0, 20, 41))
                np.testing.assert_allclose(scientific_time_axis(result, 41)[0], result.time_point)

    def test_explicit_frame_axis_overrides_registered_format_in_map_and_heat(self):
        source = self.source("npy", physical=False)
        source.parameters.update(time_axis_kind="frames", time_step=99)
        result = self.map_result(source, "distribution_analyze")
        np.testing.assert_allclose(result.data_processed, [[5]], rtol=1e-6)
        heat = self.map_result(source, "heat_transfer_calculation")
        np.testing.assert_allclose(heat.data_processed, [[42.72 / 5]], rtol=1e-6)

    def test_processed_legacy_source_preserves_inherited_frame_semantics(self):
        raw = self.source("tif_series", physical=False)
        source = ProcessedData(raw.timestamp, "legacy-processed", "Roi_applied", raw.time_point, raw.data_origin,
                               {**raw.parameters, "source_format_import": "tif_series"})
        self.assertAlmostEqual(self.region_result(source).out_processed["lifetime"], 5, places=6)

    def test_invalid_time_axis_does_not_publish_a_result(self):
        for axis in (np.arange(40), np.zeros(41), np.full(41, np.nan)):
            with self.subTest(axis=axis):
                source = self.source()
                source.time_point = axis
                worker = CalculationThread()
                results, errors = [], []
                worker.processed_result.connect(results.append)
                worker.processing_error_signal.connect(errors.append)
                worker.region_analyze(source, 0.5, np.ones((1, 1), dtype=bool), "single")
                self.assertEqual(results, [])
                self.assertEqual(len(errors), 1)


if __name__ == "__main__":
    unittest.main()
