import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import weakref

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
CORE = Path(__file__).resolve().parents[1] / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from DataManager import Data, ProcessedData
from DataProcessor import DataProcessor, MassDataProcessor
from LifetimeCalculator import CalculationThread, LifetimeCalculator
from processing.time_axis import scientific_time_axis


class AnalysisWorkerNumericsTests(unittest.TestCase):
    def setUp(self):
        Data.history.clear()
        ProcessedData.history.clear()

    def tearDown(self):
        Data.history.clear()
        ProcessedData.history.clear()

    def make_data(self, values):
        values = np.asarray(values)
        frames = len(values) if values.ndim == 3 else 1
        return Data(
            values, np.arange(frames, dtype=float), "test", values.copy(),
            {"fps": 20, "space_step": 0.2, "source_axes": "THW" if values.ndim == 3 else "HW"},
            "scientific",
        )

    def capture(self, worker):
        results, errors, cancelled = [], [], []
        worker.processed_result.connect(results.append)
        worker.processing_error_signal.connect(errors.append)
        worker.processing_cancelled_signal.connect(lambda: cancelled.append(True))
        return results, errors, cancelled

    def test_integer_anchor_statistics_and_three_supported_source_types(self):
        values = np.array([[[1, 2], [9, 9]], [[2, 5], [9, 9]]], dtype=np.uint16)
        raw = self.make_data(values)
        processed = ProcessedData(raw.timestamp, "roi", "Roi_applied", raw.time_point, values)
        canvas = SimpleNamespace(image_backup=values, time_point=raw.time_point)
        mask = np.array([[True, True], [False, False]])
        for source in (raw, processed, canvas):
            for method, expected in (("mean", [1.5, 3.5]), ("std", [0.5, 1.5]), ("quantile_075", [1.75, 4.25])):
                with self.subTest(source=type(source).__name__, method=method):
                    result = DataProcessor().get_fast_selection(source, mask, method, "roi")
                    np.testing.assert_allclose(result[:, 1], expected)

    def test_roi_mean_divides_by_roi_pixel_count(self):
        raw = self.make_data([[[1, 3], [100, 100]], [[2, 5], [100, 100]]])
        worker = CalculationThread()
        results, errors, _ = self.capture(worker)
        self.assertTrue(worker.easy_process(raw, "avg", [[True, True], [False, False]]))
        np.testing.assert_allclose(results[0].data_processed[:, 1], [2, 3.5])
        self.assertEqual(errors, [])

    def test_roi_series_carries_ps_and_scales_legacy_tiff_frame_numbers(self):
        raw = self.make_data(np.ones((3, 2, 2)))
        raw.format_import = "tif_series"
        raw.parameters.update(time_step=50, time_unit="ps")
        worker = DataProcessor()
        plots, legacy = [], []
        worker.plot_singal.connect(lambda array, kwargs: plots.append((array, kwargs)))
        worker.plot_series_signal.connect(lambda array, name: legacy.append((array, name)))
        worker.get_fast_selection(raw, np.ones((2, 2), dtype=bool), "mean", "picoseconds")
        np.testing.assert_allclose(plots[0][0][:, 0], [0, 50, 100])
        self.assertEqual(plots[0][1]["time_unit"], "ps")
        self.assertEqual(len(plots), 1)
        self.assertEqual(len(legacy), 1)

    def test_canvas_series_inherits_weak_parent_time_metadata(self):
        raw = self.make_data(np.ones((3, 2, 2)))
        raw.format_import = "tif_series"
        raw.parameters.update(time_step=0.5, time_unit="ps")
        canvas = SimpleNamespace(
            image_backup=raw.data_origin, time_point=raw.time_point,
            parent_data=weakref.ref(raw), source_format="tif_series",
        )
        plots = []
        worker = DataProcessor()
        worker.plot_singal.connect(lambda array, kwargs: plots.append((array, kwargs)))
        worker.get_fast_selection(canvas, np.ones((2, 2), dtype=bool), "mean", "anchor")
        np.testing.assert_allclose(plots[0][0][:, 0], [0, 0.5, 1])
        self.assertEqual(plots[0][1]["time_unit"], "ps")

    def test_derived_roi_retains_legacy_source_format_scaling(self):
        roi = ProcessedData(
            1, "roi", "Roi_applied", np.array([0, 1, 2]), np.ones((3, 2, 2)),
            {"source_format_import": "tif_series", "time_step": 20, "time_unit": "fs"},
        )
        axis, unit = scientific_time_axis(roi, 3)
        np.testing.assert_allclose(axis, [0, 20, 40])
        self.assertEqual(unit, "fs")

    def test_sif_folder_frame_coordinates_use_time_step(self):
        raw = self.make_data(np.ones((3, 2, 2)))
        raw.format_import = "sif"
        raw.time_point = np.array([0, 2, 4])
        raw.parameters.update(data_type="sif_folder", time_step=0.5, time_unit="ps")
        axis, unit = scientific_time_axis(raw, 3)
        np.testing.assert_allclose(axis, [0, 1, 2])
        self.assertEqual(unit, "ps")

    def test_generic_import_physical_axis_is_preserved(self):
        raw = self.make_data(np.ones((3, 2, 2)))
        raw.format_import = "tiff"
        raw.time_point = np.array([0, 50, 100])
        raw.parameters.update(time_step=50, time_unit="ps")
        axis, unit = scientific_time_axis(raw, 3)
        np.testing.assert_allclose(axis, raw.time_point)
        self.assertEqual(unit, "ps")

    def test_fps_based_generic_import_unit_is_seconds(self):
        raw = self.make_data(np.ones((3, 2, 2)))
        raw.format_import = "npy"
        raw.time_point = np.array([0, 0.05, 0.1])
        raw.parameters.update(time_basis="fps", time_unit="ps")
        axis, unit = scientific_time_axis(raw, 3)
        np.testing.assert_allclose(axis, raw.time_point)
        self.assertEqual(unit, "s")

    def test_mean_result_records_physical_axis_without_rescaling_twice(self):
        raw = self.make_data(np.ones((3, 2, 2)))
        raw.format_import = "tif_series"
        raw.parameters.update(time_step=25, time_unit="ps")
        worker = CalculationThread()
        results, _, _ = self.capture(worker)
        self.assertTrue(worker.easy_process(raw, "mean"))
        result = results[0]
        np.testing.assert_allclose(result.time_point, [0, 25, 50])
        np.testing.assert_allclose(result.data_processed[:, 0], result.time_point)
        self.assertEqual(result.out_processed["time_unit"], "ps")
        self.assertEqual(result.out_processed["time_axis_kind"], "physical")
        np.testing.assert_allclose(scientific_time_axis(result, 3)[0], result.time_point)

    def test_unknown_units_and_histogram_semantics_are_explicit(self):
        raw = self.make_data(np.ones((3, 2, 2)))
        self.assertEqual(scientific_time_axis(raw, 3)[1], "unknown")
        worker = DataProcessor()
        plots = []
        worker.plot_singal.connect(lambda array, kwargs: plots.append(kwargs))
        worker.get_roi_value_distribution(raw, np.ones((2, 2), dtype=bool), 0, 2, None, "histogram")
        self.assertEqual(plots[0]["analysis_mode"], "hist_precomputed")
        self.assertNotIn("time_unit", plots[0])

    def test_empty_roi_does_not_publish_a_result(self):
        worker = CalculationThread()
        results, errors, _ = self.capture(worker)
        self.assertFalse(worker.easy_process(self.make_data(np.ones((2, 2, 2))), "mean", np.zeros((2, 2))))
        self.assertEqual(results, [])
        self.assertEqual(len(errors), 1)

    def test_temporal_mean_and_single_image_rejection(self):
        worker = MassDataProcessor()
        results, errors, _ = self.capture(worker)
        values = np.arange(12).reshape(3, 2, 2)
        self.assertTrue(worker.accumulate_amplitude(self.make_data(values)))
        np.testing.assert_allclose(results[0].data_processed, values.mean(axis=0))
        self.assertEqual(results[0].out_processed["scientific_axes"], "HW")
        self.assertFalse(worker.accumulate_amplitude(self.make_data(values[0])))
        self.assertEqual(len(results), 1)
        self.assertEqual(len(errors), 1)

    def test_fourier_round_trip_preserves_hw_singleton_and_sequence_shapes(self):
        for shape in ((3, 4), (1, 3, 4), (2, 3, 4), (2, 1, 4)):
            with self.subTest(shape=shape):
                values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) - 5
                raw = self.make_data(values)
                worker = MassDataProcessor()
                results, errors, _ = self.capture(worker)
                self.assertTrue(worker.twoD_fourier_transform(raw))
                transformed = results[0]
                self.assertEqual(transformed.data_processed.shape, shape)
                self.assertTrue(np.iscomplexobj(transformed.out_processed["twoD_FFT"]))
                self.assertTrue(worker.twoD_inverse_fourier_transform(transformed))
                self.assertEqual(results[1].data_processed.shape, shape)
                np.testing.assert_allclose(results[1].data_processed, values, atol=1e-6)
                self.assertEqual(errors, [])

    def test_inverse_transform_rejects_spectrum_without_phase(self):
        worker = MassDataProcessor()
        results, errors, _ = self.capture(worker)
        self.assertFalse(worker.twoD_inverse_fourier_transform(self.make_data(np.ones((2, 3, 4)))))
        self.assertEqual(results, [])
        self.assertEqual(len(errors), 1)

    def test_fourier_round_trip_preserves_physical_ps_axis(self):
        raw = self.make_data(np.ones((3, 2, 2)))
        raw.format_import = "tif_series"
        raw.parameters.update(time_step=50, time_unit="ps", source_format_import="tif_series")
        worker = MassDataProcessor()
        results, errors, _ = self.capture(worker)
        self.assertTrue(worker.twoD_fourier_transform(raw))
        self.assertTrue(worker.twoD_inverse_fourier_transform(results[0]))
        np.testing.assert_allclose(results[0].time_point, [0, 50, 100])
        np.testing.assert_allclose(results[1].time_point, [0, 50, 100])
        self.assertEqual(results[1].out_processed["time_unit"], "ps")
        self.assertEqual(errors, [])

    def test_fourier_cancel_does_not_register_partial_output(self):
        raw = self.make_data(np.ones((3, 3, 4)))
        worker = MassDataProcessor()
        results, _, cancelled = self.capture(worker)
        worker.processing_progress_signal.connect(lambda done, total: worker.cancellation_token.cancel() if done == 1 else None)
        self.assertFalse(worker.twoD_fourier_transform(raw))
        self.assertEqual(results, [])
        self.assertEqual(len(ProcessedData.history), 0)
        self.assertEqual(cancelled, [True])

    def test_diffusion_fits_variances_and_coefficient_in_worker(self):
        positions = np.linspace(-5, 5, 41)
        profiles = {
            0: np.column_stack((positions, np.exp(-positions ** 2 / 2))),
            2: np.column_stack((positions, np.exp(-positions ** 2 / 6))),
        }
        worker = CalculationThread()
        results, errors, _ = self.capture(worker)
        self.assertTrue(worker.diffusion_calculation(profiles, 0.5, 1.0, 1.0, "diffusion"))
        np.testing.assert_allclose(results[0].out_processed["sigma"][1], [1, 3], rtol=1e-6)
        self.assertAlmostEqual(results[0].out_processed["diffusion_coefficient"], 1, places=6)
        self.assertEqual(errors, [])

    def test_insufficient_successful_diffusion_fits_are_atomic(self):
        positions = np.arange(5, dtype=float)
        profile = np.column_stack((positions, np.ones(5)))
        worker = CalculationThread()
        results, errors, _ = self.capture(worker)
        with patch.object(LifetimeCalculator, "gaussian_fit", side_effect=[RuntimeError("fit failed"), (np.array([1, 1, 1]), None)]):
            self.assertFalse(worker.diffusion_calculation({0: profile, 1: profile}, 1, 1, 1, "invalid"))
        self.assertEqual(results, [])
        self.assertEqual(len(ProcessedData.history), 0)
        self.assertEqual(len(errors), 1)

    def test_diffusion_rejected_frames_leave_aligned_successful_results(self):
        positions = np.arange(5, dtype=float)
        profile = np.column_stack((positions, np.ones(5)))
        worker = CalculationThread()
        results, errors, _ = self.capture(worker)
        fits = [(np.array([1, 1, 1]), None), RuntimeError("fit failed"), (np.array([1, 1, 2]), None)]
        with patch.object(LifetimeCalculator, "gaussian_fit", side_effect=fits):
            self.assertTrue(worker.diffusion_calculation({0: profile, 1: profile, 2: profile}, 1, 1, 1, "aligned"))
        output = results[0].out_processed
        np.testing.assert_array_equal(output["time_series"], [0, 2])
        self.assertEqual(output["signal"].shape[0], 2)
        self.assertEqual(output["fitting"].shape[0], 2)
        self.assertEqual(output["rejected_frames"][0]["frame"], 1)
        self.assertEqual(errors, [])

    def test_bad_frame_detection_handles_zero_mad_and_emits_results(self):
        worker = DataProcessor()
        detected = []
        worker.bad_frames_detected.connect(detected.append)
        self.assertEqual(worker.detect_bad_frames_auto(np.zeros((9, 2, 2))), [])
        frames = np.zeros((9, 2, 2))
        frames[4] = 10
        self.assertEqual(worker.detect_bad_frames_auto(frames), [3, 4, 5])
        self.assertEqual(detected, [[], [3, 4, 5]])

    def test_bad_frame_repair_publishes_new_data_and_preserves_source(self):
        values = np.stack([np.full((2, 2), value) for value in (1, 99, 2)])
        raw = self.make_data(values)
        worker = DataProcessor()
        fixed, failures = [], []
        worker.bad_frames_fixed.connect(fixed.append)
        worker.bad_frame_failed.connect(failures.append)
        result = worker.fix_bad_frames(raw, [1], 1)
        self.assertIsNot(result, raw)
        np.testing.assert_allclose(result.data_origin[1], 1.5)
        np.testing.assert_array_equal(raw.data_origin, values)
        self.assertEqual(result.parameters["bad_frame_source_timestamp"], raw.timestamp)
        self.assertEqual(result.parameters["bad_frame_indices"], [1])
        self.assertEqual(fixed, [result])
        self.assertEqual(failures, [])

    def test_bad_frame_repair_failure_preserves_original_history(self):
        raw = self.make_data(np.ones((2, 2, 2)))
        worker = DataProcessor()
        failures = []
        worker.bad_frame_failed.connect(failures.append)
        count = len(Data.history)
        self.assertIsNone(worker.fix_bad_frames(raw, [0, 1], 1))
        self.assertEqual(len(Data.history), count)
        self.assertEqual(len(failures), 1)

    def test_optical_flow_accepts_float_grayscale_and_converts_physical_speed(self):
        raw = self.make_data(np.arange(3 * 8 * 8, dtype=np.float32).reshape(3, 8, 8))
        worker = MassDataProcessor()
        results, errors, _ = self.capture(worker)

        def flow(previous, current, *args, **kwargs):
            self.assertEqual(previous.dtype, np.uint8)
            self.assertEqual(current.dtype, np.uint8)
            vectors = np.zeros((*previous.shape, 2), dtype=np.float32)
            vectors[..., 0] = 1
            return vectors

        with patch("DataProcessor.cv2.calcOpticalFlowFarneback", side_effect=flow):
            self.assertTrue(worker.heartbeat_movement(raw, 2, -1, [0, 2]))
        np.testing.assert_allclose(results[0].out_processed["mean_speed"], [2])
        np.testing.assert_allclose(results[0].out_processed["time_seconds"], [0.1])
        self.assertEqual(errors, [])

    def test_optical_flow_checks_budget_before_computing(self):
        worker = MassDataProcessor()
        results, errors, _ = self.capture(worker)
        raw = self.make_data(np.ones((2, 8, 8)))
        with patch("DataProcessor._ensure_working_memory", side_effect=MemoryError("budget")), patch("DataProcessor.cv2.calcOpticalFlowFarneback") as compute:
            self.assertFalse(worker.heartbeat_movement(raw, 2, -1, [0, 1]))
        compute.assert_not_called()
        self.assertEqual(results, [])
        self.assertEqual(len(errors), 1)

    def test_optical_flow_cancellation_does_not_publish_partial_history(self):
        raw = self.make_data(np.ones((3, 8, 8)))
        worker = MassDataProcessor()
        results, _, cancelled = self.capture(worker)

        def cancel_flow(previous, current, *args, **kwargs):
            worker.cancellation_token.cancel()
            return np.zeros((*previous.shape, 2), dtype=np.float32)

        with patch("DataProcessor.cv2.calcOpticalFlowFarneback", side_effect=cancel_flow):
            self.assertFalse(worker.heartbeat_movement(raw, 2, -1, [0, 1, 2]))
        self.assertEqual(results, [])
        self.assertEqual(len(ProcessedData.history), 0)
        self.assertEqual(cancelled, [True])

    def test_optical_flow_rejects_missing_fps(self):
        raw = self.make_data(np.ones((2, 8, 8)))
        raw.parameters["fps"] = 0
        worker = MassDataProcessor()
        results, errors, _ = self.capture(worker)
        self.assertFalse(worker.heartbeat_movement(raw, 2, -1, [0, 1]))
        self.assertEqual(results, [])
        self.assertEqual(len(errors), 1)

    def test_real_opencv_flow_accepts_uint16_and_constant_float_sequences(self):
        for dtype in (np.uint16, np.float64):
            with self.subTest(dtype=dtype):
                worker = MassDataProcessor()
                results, errors, _ = self.capture(worker)
                raw = self.make_data(np.zeros((2, 8, 8), dtype=dtype))
                self.assertTrue(worker.heartbeat_movement(raw, 2, -1, [0, 1]))
                np.testing.assert_allclose(results[0].out_processed["mean_speed"], [0])
                self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
