import os
import sys
import tempfile
import threading
import time
import unittest
try:
    from .qt_test_support import QtTestCase
except ImportError:
    from qt_test_support import QtTestCase
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
CORE = Path(__file__).resolve().parents[1] / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

import numpy as np
from scipy import ndimage
from PyQt5 import sip
from PyQt5.QtCore import QEventLoop, QThread, pyqtSignal
from PyQt5.QtWidgets import QApplication, QMainWindow, QTableWidgetItem

from ArrayCache import ArrayCacheConfig, ArrayRef
from DataManager import Data, ProcessedData, ImagingData
from DataProcessor import MassDataProcessor
from convolution.engine import (
    ConvolutionPlan, effective_kernel, execute_convolution,
    preview_convolution, validate_convolution,
)
from convolution.dialog import ConvolutionDialog
from compute.algorithms.convolution_pipeline import run_spatiotemporal_convolution
from history.manifest import build_manifest_item, restore_history_item
from processing.result_router import ResultRouter
from tasks.model import CancellationToken, TaskCancelled, TaskRecord
from tasks import TaskCoordinator, TaskStatus
from processing.controller import ProcessingController
from MainWindow import MainWindow
from startup.shutdown import WindowShutdownController


class _ConvolutionWindow(QMainWindow):
    """Exercise the real entry slots, queued worker and task coordinator."""
    convolution_signal = pyqtSignal(object)
    process_convolution = MainWindow.process_convolution
    _submit_convolution_plan = MainWindow._submit_convolution_plan
    _convolution_closed = MainWindow._convolution_closed
    _cancel_convolution = MainWindow._cancel_convolution

    def __init__(self, source, cache_dir):
        super().__init__()
        self.source = source
        self.outputs = []
        self.task_coordinator = TaskCoordinator(self)
        self.processing_controller = ProcessingController(self, self.task_coordinator)
        self.result_router = ResultRouter(self, self.processing_controller)
        self.avi_thread = QThread(self)
        self.mass_data_processor = MassDataProcessor()
        self.mass_data_processor.moveToThread(self.avi_thread)
        self.avi_thread.finished.connect(self.mass_data_processor.deleteLater)
        self.convolution_signal.connect(self.mass_data_processor.convolve)
        self.mass_data_processor.processed_result.connect(self._result)
        self.mass_data_processor.processing_progress_signal.connect(self._progress)
        self.mass_data_processor.processing_cancelled_signal.connect(self._cancelled)
        self.mass_data_processor.processing_error_signal.connect(self._failed)
        self.tool_params = {"memory_budget_mb": 256, "cache_threshold_mb": 512,
                            "cache_directory": cache_dir}

    def calculator_sources(self):
        return [self.source]

    def get_compute_options(self, algorithm):
        return {"host_memory_limit_mb": 256}

    def add_new_canvas(self, value):
        self.outputs.append(value)

    def _result(self, value):
        self.result_router.route(value)

    def _progress(self, current, total):
        self.processing_controller.progress("em_processing", current, total)

    def _cancelled(self):
        self.processing_controller.cancelled("em_processing")

    def _failed(self, error):
        self.processing_controller.fail(error, categories=("em_processing",))

    def shutdown(self):
        if sip.isdeleted(self):
            return
        dialog = getattr(self, "convolution_dialog", None)
        if dialog is not None and not sip.isdeleted(dialog):
            dialog.close()
        if not sip.isdeleted(self.avi_thread):
            self.avi_thread.quit()
            self.avi_thread.wait(3000)
        self.close()


class ConvolutionWorkspaceTests(QtTestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        config = ArrayCacheConfig(Path(self.directory.name), threshold_bytes=1024 ** 3)
        self.cache_patch = patch("DataManager._ARRAY_CACHE_CONFIG", config)
        self.cache_patch.start()
        self.addCleanup(self.cache_patch.stop)
        Data.history.clear()
        ProcessedData.history.clear()

    @staticmethod
    def source(values):
        values = np.asarray(values)
        axes = "THW" if values.ndim == 3 else "HW"
        points = np.arange(values.shape[0]) * 0.2 if values.ndim == 3 else None
        return Data(values, points, "test", values,
                    {"scientific_axes": axes, "fps": 5, "space_step": 0.8,
                     "space_unit": "um", "unrelated": np.ones((3, 4))}, "source")

    def test_global_boundaries_and_long_asymmetric_kernels_match_scipy(self):
        source = np.arange(6, dtype=np.float64).reshape(2, 3)
        weights = np.arange(24, dtype=np.float64).reshape(4, 6) - 3
        for boundary in ("reflect", "mirror", "nearest", "wrap", "constant"):
            with self.subTest(boundary=boundary):
                expected = ndimage.convolve(source, weights, mode=boundary, cval=2.5, origin=(-1, 1))
                actual = run_spatiotemporal_convolution(
                    source, weights, core_shape=(1, 2), boundary=boundary,
                    cval=2.5, origin=(-1, 1),
                )
                np.testing.assert_allclose(actual, expected)

    def test_preview_uses_neighbors_outside_spatial_roi_and_temporal_frame(self):
        source = self.source(np.arange(5 * 6 * 7, dtype=np.float64).reshape(5, 6, 7))
        weights = np.arange(36, dtype=np.float64).reshape(3, 4, 3) - 5
        plan = ConvolutionPlan(source, weights, "THW", boundary="wrap", origin=(0, -1, 1))
        expected = ndimage.convolve(source.data_origin, weights, mode="wrap", origin=plan.origin)
        for frame, row, column in ((0, 0, 0), (2, 2, 3), (4, 5, 6)):
            with self.subTest(position=(frame, row, column)):
                original, actual = preview_convolution(plan, frame, row, column, size=2)
                np.testing.assert_array_equal(original, source.data_origin[frame, row:row + 2, column:column + 2])
                np.testing.assert_allclose(actual, expected[frame, row:row + 2, column:column + 2])

    def test_zero_sum_normalization_and_axis_reinterpretation_are_rejected(self):
        source = self.source(np.ones((3, 4)))
        plan = ConvolutionPlan(source, np.array([[-1, 1]]), "HW", normalize=True)
        with self.assertRaisesRegex(ValueError, "零和"):
            validate_convolution(plan)
        source.parameters["scientific_axes"] = "WH"
        with self.assertRaisesRegex(ValueError, "源数据轴"):
            validate_convolution(replace(plan, normalize=False))

    def test_dense_long_kernel_budget_includes_ndimage_offsets_before_compute(self):
        source = self.source(np.ones((16, 16, 16), dtype=np.float64))
        plan = ConvolutionPlan(source, np.ones((15, 15, 15)), "THW",
                               memory_limit_bytes=4 * 1024 * 1024,
                               cache_dir=self.directory.name)
        with patch("convolution.engine.run_spatiotemporal_convolution") as compute:
            with self.assertRaises(MemoryError):
                execute_convolution(plan)
            compute.assert_not_called()
        with patch("convolution.engine.convolve_region") as compute:
            with self.assertRaisesRegex(MemoryError, "64 MB"):
                preview_convolution(replace(plan, memory_limit_bytes=512 * 1024 * 1024), 0, 0, 0)
            compute.assert_not_called()

    def test_output_retains_complex_values_fractional_weights_and_metadata(self):
        values = np.arange(24).reshape(2, 3, 4) + 1j
        source = self.source(values)
        plan = ConvolutionPlan(source, np.ones((1, 3, 3)), "THW", normalize=True)
        result = execute_convolution(plan)
        expected = ndimage.convolve(values, effective_kernel(plan), mode="reflect")
        np.testing.assert_allclose(result.data_processed, expected)
        self.assertTrue(np.iscomplexobj(result.data_processed))
        np.testing.assert_array_equal(result.time_point, source.time_point)
        self.assertEqual(result.out_processed["space_step"], 0.8)
        self.assertNotIn("unrelated", result.out_processed)
        self.assertEqual(result.out_processed["convolution"]["source_timestamp"], source.timestamp)
        self.assertEqual(result.out_processed["compute"]["actual_backend"], "cpu")

    def test_disk_result_restores_and_enters_canvas_and_export_data_path(self):
        source = self.source(np.arange(60, dtype=np.float32).reshape(3, 4, 5))
        plan = ConvolutionPlan(source, np.ones((1, 1, 1)), "THW",
                               disk_threshold_bytes=1, cache_dir=self.directory.name)
        result = execute_convolution(plan)
        storage = result.__dict__["_data_processed_storage"]
        self.assertIsInstance(storage, ArrayRef)
        restored = restore_history_item(build_manifest_item(result), self.directory.name)
        np.testing.assert_allclose(restored.data_processed, source.data_origin)
        np.testing.assert_array_equal(restored.time_point, source.time_point)
        canvas_data = ImagingData.create_image(restored)
        self.assertEqual(canvas_data.totalframes, 3)
        routed = []
        completed = []
        window = SimpleNamespace(add_new_canvas=lambda value: routed.append(value))
        router = ResultRouter(window, SimpleNamespace(complete_for_result=completed.append))
        self.assertTrue(router.route(restored))
        self.assertEqual(routed, [restored])
        self.assertEqual(completed, ["spatiotemporal_convolution"])
        export_path = Path(self.directory.name) / "export.npy"
        np.save(export_path, restored.data_processed)
        np.testing.assert_allclose(np.load(export_path), source.data_origin)
        # Close mappings before Windows removes the temporary directory.
        for value in (result, restored):
            value.data_processed._mmap.close()
        canvas_data.display_source.array._mmap.close()

    def test_mid_task_cancel_aborts_disk_output_without_registering_history(self):
        source = self.source(np.arange(10000, dtype=np.float32).reshape(100, 100))
        plan = ConvolutionPlan(source, np.ones((3, 3)), "HW",
                               disk_threshold_bytes=1, cache_dir=self.directory.name)
        token = CancellationToken()

        def progress(current, total, message):
            if current:
                token.cancel()

        with self.assertRaises(TaskCancelled):
            execute_convolution(plan, token=token, progress=progress)
        self.assertEqual(len(ProcessedData.history), 0)
        self.assertEqual(list(Path(self.directory.name).rglob("*.npy")), [])

    def test_queued_cancel_does_not_cancel_next_worker_operation(self):
        source = self.source(np.ones((3, 4)))
        plan = ConvolutionPlan(source, np.ones((1, 1)), "HW")
        worker = MassDataProcessor()
        cancelled = []
        results = []
        worker.processing_cancelled_signal.connect(lambda: cancelled.append(True))
        worker.processed_result.connect(results.append)
        first = CancellationToken()
        first.cancel()
        worker.enqueue_task_context("first", first)
        worker.enqueue_task_context("second", CancellationToken())
        self.assertFalse(worker.convolve(plan))
        self.assertTrue(worker.convolve(plan))
        self.assertEqual(len(cancelled), 1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].out_processed["compute"]["task_id"], "second")

    def test_kernel_layer_edits_and_dialog_task_state_are_preserved(self):
        source = self.source(np.ones((4, 5, 6)))
        dialog = ConvolutionDialog(lambda: [source])
        self.addCleanup(dialog.close)
        self.assertTrue(dialog.set_source(source))
        dialog.sizes[0].setValue(3)
        dialog.preset.setCurrentText("自定义")
        dialog._generate_kernel()
        dialog.weights_table.setItem(0, 0, QTableWidgetItem("2.5"))
        dialog.layer_selector.setValue(1)
        dialog.weights_table.setItem(1, 1, QTableWidgetItem("-1"))
        plan = dialog.build_plan()
        self.assertEqual(plan.kernel[0, 0, 0], 2.5)
        self.assertEqual(plan.kernel[1, 1, 1], -1)
        task = TaskRecord("时空卷积", "em_processing")
        task.start()
        dialog.bind_task(task)
        self.assertFalse(dialog.run_button.isEnabled())
        task.cancel()
        dialog.task_finished(task)
        self.assertTrue(dialog.run_button.isEnabled())
        self.assertIn("取消", dialog.status.text())

    def wait_until(self, predicate):
        deadline = time.monotonic() + 4
        while not predicate() and time.monotonic() < deadline:
            self.app.processEvents(QEventLoop.AllEvents, 20)
            time.sleep(0.002)
        self.assertTrue(predicate(), "后台卷积未在预期时间内完成")

    def test_entry_reuses_window_then_reopens_and_runs_through_task_route(self):
        source = self.source(np.arange(60, dtype=np.float32).reshape(3, 4, 5))
        window = _ConvolutionWindow(source, self.directory.name)
        self.addCleanup(window.shutdown)
        window.process_convolution()
        first = window.convolution_dialog
        window.process_convolution()
        self.assertIs(window.convolution_dialog, first)
        with patch("sys.excepthook") as unhandled:
            first.close()
            first.close()
        unhandled.assert_not_called()
        self.assertIsNone(window.convolution_dialog)
        self.app.processEvents()
        window.process_convolution()
        second = window.convolution_dialog
        self.assertIsNot(second, first)
        second.set_source(source)
        second.run_button.click()
        self.wait_until(lambda: len(window.outputs) == 1)
        np.testing.assert_array_equal(window.outputs[0].data_processed, source.data_origin)
        self.assertIsNone(second.task_id)
        self.assertTrue(second.run_button.isEnabled())

    def test_gui_cancel_button_cancels_running_block_and_reenables_execution(self):
        source = self.source(np.ones((100, 100), dtype=np.float32))
        window = _ConvolutionWindow(source, self.directory.name)
        self.addCleanup(window.shutdown)
        window.process_convolution()
        dialog = window.convolution_dialog
        dialog.set_source(source)
        entered = threading.Event()
        resume = threading.Event()
        self.addCleanup(resume.set)

        def paused_pipeline(*args, progress, **kwargs):
            def notify(current, total, message):
                progress(current, total, message)
                if current == 1:
                    entered.set()
                    resume.wait(3)
            return run_spatiotemporal_convolution(*args, progress=notify, **kwargs)

        with patch("convolution.engine.run_spatiotemporal_convolution", side_effect=paused_pipeline):
            dialog.run_button.click()
            self.wait_until(entered.is_set)
            task = window.task_coordinator.registry.get(dialog.task_id)
            dialog.cancel_button.click()
            self.assertEqual(task.status, TaskStatus.CANCELLING)
            resume.set()
            self.wait_until(lambda: task.status == TaskStatus.CANCELLED)
        self.assertTrue(dialog.run_button.isEnabled())
        self.assertEqual(window.outputs, [])
        self.assertEqual(len(ProcessedData.history), 0)

    def test_closed_dialog_preview_is_still_owned_until_main_window_shutdown(self):
        source = self.source(np.ones((5, 6), dtype=np.float32))
        window = _ConvolutionWindow(source, self.directory.name)
        self.addCleanup(window.shutdown)
        window.shutdown_controller = WindowShutdownController(window)
        window.show()
        window.process_convolution()
        dialog = window.convolution_dialog
        dialog.set_source(source)
        entered = threading.Event()
        resume = threading.Event()
        self.addCleanup(resume.set)

        def paused_preview(*args):
            entered.set()
            resume.wait(3)
            return np.ones((2, 2)), np.ones((2, 2))

        with patch("convolution.dialog.preview_convolution", side_effect=paused_preview):
            dialog._request_preview()
            self.wait_until(entered.is_set)
            pool = window._convolution_preview_pool
            started = time.monotonic()
            dialog.close()
            self.assertLess(time.monotonic() - started, 0.2)
            self.assertIsNone(window.convolution_dialog)
            self.app.processEvents()
            self.assertTrue(pool.shutdown_pending())
            window.close()
            self.assertTrue(window.isVisible())
            self.assertFalse(window.shutdown_controller._finished)
            # The GUI remains active while the native preview is finishing.
            self.app.processEvents()
            resume.set()
            self.wait_until(lambda: not pool.shutdown_pending())
            self.wait_until(lambda: not window.isVisible())
        self.assertEqual(pool.tasks, {})


if __name__ == "__main__":
    unittest.main()
