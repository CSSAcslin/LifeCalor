import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from PyQt5.QtCore import QEventLoop, QThreadPool
from PyQt5.QtWidgets import QApplication

from DataManager import Data, ProcessedData
from DataProcessor import MassDataProcessor
from ExtraDialog import DataTreeViewDialog
from calculator import CalculationEngine, CalculationPlan, OperandSpec
from calculator.dialog import DataCalculatorDialog
from calculator.metadata import CalculationMetadataPolicy
from calculator.preview import CalculatorPreviewTask
from widget.DataTreeWidget import DataHistoryTreeWidget
from history.manifest import restore_history_item


class CalculatorWorkspaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        for module_name in list(sys.modules):
            if module_name == "matplotlib" or module_name.startswith("matplotlib."):
                sys.modules.pop(module_name, None)
        Data.history.clear()
        ProcessedData.history.clear()

    def close_dialog(self, dialog):
        dialog.close()
        self.app.processEvents(QEventLoop.AllEvents, 50)

    @staticmethod
    def data(value, name, parameters=None):
        value = np.asarray(value)
        points = np.arange(value.shape[0]) if value.ndim == 3 else np.array([0.0])
        return Data(value, points, "test", value, name=name, parameters=parameters or {})

    def test_invalid_broadcast_is_a_visible_failed_step(self):
        a = self.data(np.ones((4, 3, 2)), "A")
        b = self.data(np.ones((5, 4)), "B")
        result = CalculationEngine.validate(
            CalculationPlan("A + B", [OperandSpec("A", a), OperandSpec("B", b)])
        )

        self.assertFalse(result.valid)
        self.assertEqual(result.steps[-1].status, "invalid")
        self.assertEqual(result.steps[-1].expression, "A + B")
        self.assertEqual(result.steps[-1].input_shapes, ((4, 3, 2), (5, 4)))

    def test_dialog_highlights_the_failed_validation_row(self):
        a = self.data(np.ones((4, 3, 2)), "a")
        b = self.data(np.ones((5, 4)), "b")
        dialog = DataCalculatorDialog([a, b])
        self.addCleanup(self.close_dialog, dialog)
        dialog.expression.setPlainText("A + B")
        dialog._validate()

        last = dialog.trace_table.rowCount() - 1
        self.assertEqual(dialog.trace_table.item(last, 4).text(), "失败")
        self.assertEqual(dialog.trace_table.currentRow(), last)
        self.assertIn("无法广播", dialog.trace_table.item(last, 0).toolTip())

    def test_metadata_inherits_scalars_but_not_large_arrays(self):
        source = self.data(
            np.ones((4, 3, 2)),
            "source",
            {
                "fps": 20,
                "time_unit": "s",
                "space_unit": "um",
                "value_unit": "count",
                "custom_scalar": 7,
                "large_extra": np.ones((4, 3, 2)),
            },
        )
        plan = CalculationPlan("A * 2", [OperandSpec("A", source, axes="THW")])
        value, validation = CalculationEngine.execute(plan)
        metadata = CalculationMetadataPolicy.build(plan, validation, value)

        self.assertEqual(metadata.out_processed["fps"], 20)
        self.assertEqual(metadata.parameters["custom_scalar"], 7)
        self.assertEqual(metadata.parameters["scientific_axes"], "THW")
        self.assertNotIn("large_extra", metadata.out_processed)
        np.testing.assert_array_equal(metadata.time_point, source.time_point)

    def test_dialog_uses_stable_on_demand_slots_and_frame_input(self):
        a_values = np.arange(24, dtype=np.float32).reshape(4, 3, 2)
        a = self.data(a_values, "a")
        b = self.data(np.ones((4, 3, 2)), "b")
        c = self.data(np.ones((4, 3, 2)) * 2, "c")
        dialog = DataCalculatorDialog([a], source_provider=lambda: [a, b, c])
        self.addCleanup(self.close_dialog, dialog)

        self.assertEqual(dialog.slot_table.rowCount(), 1)
        self.assertEqual(dialog.slots[0].axes, "THW")
        dialog.add_source(b)
        dialog.add_source(c)
        dialog.slot_table.selectRow(1)
        dialog._remove_selected_slot()
        self.assertEqual([slot.alias for slot in dialog.slots], ["A", "C"])

        dialog.slot_table.selectRow(0)
        dialog.frame_input.setValue(2)
        self.assertIsNone(dialog._preview_array)
        dialog._request_preview()
        QThreadPool.globalInstance().waitForDone(2000)
        self.app.processEvents(QEventLoop.AllEvents, 50)
        np.testing.assert_array_equal(dialog.preview.image, a_values[2])
        self.assertIn("帧 2", dialog.preview_label.text())
        self.assertTrue(dialog._preview_array.flags.c_contiguous)
        self.assertTrue(dialog._preview_array.flags.owndata)
        self.assertEqual(dialog.preview.imageItem.axisOrder, "row-major")

    def test_dialog_starts_empty_and_selection_does_not_read_arrays(self):
        source = self.data(np.ones((3, 4, 5)), "source")
        dialog = DataCalculatorDialog([], source_provider=lambda: [source])
        self.addCleanup(self.close_dialog, dialog)
        self.assertEqual(dialog.slot_table.rowCount(), 0)
        self.assertIsNone(dialog._preview_array)
        dialog.add_source(source)
        self.app.processEvents(QEventLoop.AllEvents, 20)
        self.assertIsNone(dialog._preview_array)
        self.assertIn("读取帧", dialog.preview_label.text())

    def test_background_preview_handles_strided_frames(self):
        values = np.arange(192, dtype=np.float32).reshape(3, 8, 8)[:, :, ::-1]
        source = self.data(values, "strided")
        dialog = DataCalculatorDialog([source])
        self.addCleanup(self.close_dialog, dialog)
        dialog.frame_input.setValue(1)
        dialog._request_preview()
        QThreadPool.globalInstance().waitForDone(2000)
        self.app.processEvents(QEventLoop.AllEvents, 50)

        np.testing.assert_array_equal(dialog.preview.image, values[1])
        self.assertTrue(dialog._preview_array.flags.c_contiguous)
        self.assertTrue(dialog._preview_array.flags.owndata)
        self.assertEqual(dialog.preview.imageItem.axisOrder, "row-major")
        dialog.close()

    def test_completed_calculation_previews_result_not_input(self):
        source = self.data(np.zeros((3, 4, 5), dtype=np.float32), "input")
        result_values = np.full((3, 4, 5), 7, dtype=np.float32)
        result = ProcessedData(
            source.timestamp,
            "result",
            "Multi_data_math",
            time_point=np.arange(3),
            data_processed=result_values,
            out_processed={"scientific_axes": "THW"},
        )
        dialog = DataCalculatorDialog([source])
        self.addCleanup(self.close_dialog, dialog)

        dialog.set_execution_finished(result)
        QThreadPool.globalInstance().waitForDone(2000)
        self.app.processEvents(QEventLoop.AllEvents, 50)

        np.testing.assert_array_equal(dialog.preview.image, result_values[0])
        self.assertEqual(dialog._preview_mode, "result")
        self.assertIn("运算结果", dialog.preview_label.text())
        self.assertIn(result.name, dialog.preview_label.text())

        dialog._slot_selection_changed()
        self.assertEqual(dialog._preview_mode, "input")
        self.assertIsNone(dialog._preview_array)
        self.assertIn("输入数据", dialog.preview_label.text())

    def test_restored_cache_is_not_touched_until_preview_is_requested(self):
        values = np.arange(60, dtype=np.float32).reshape(3, 4, 5)
        with tempfile.TemporaryDirectory() as directory:
            cache_file = Path(directory) / "restored.npy"
            np.save(cache_file, values)
            restored = restore_history_item({
                "kind": "ProcessedData",
                "name": "restored",
                "type_processed": "math",
                "timestamp": 22.0,
                "timestamp_inherited": 12.0,
                "serial_number": 8,
                "shape": [3, 4, 5],
                "dtype": "float32",
                "ndim": 3,
                "datamin": 0.0,
                "datamax": 59.0,
                "metadata": {"parameters": {"scientific_axes": "THW"}},
                "arrays": {
                    "data_processed": {
                        "path": str(cache_file),
                        "shape": [3, 4, 5],
                        "dtype": "float32",
                        "nbytes": int(values.nbytes),
                        "created_at": 1.0,
                        "field_name": "data_processed",
                    }
                },
            })
            dialog = DataCalculatorDialog([], source_provider=lambda: [restored])
            self.addCleanup(self.close_dialog, dialog)

            self.assertEqual(dialog.slot_table.rowCount(), 0)
            self.assertIsNone(dialog._preview_array)
            dialog.add_source(restored)
            self.app.processEvents(QEventLoop.AllEvents, 20)
            self.assertIsNone(dialog._preview_array)

            dialog.frame_input.setValue(2)
            dialog._request_preview()
            QThreadPool.globalInstance().waitForDone(2000)
            self.app.processEvents(QEventLoop.AllEvents, 50)
            np.testing.assert_array_equal(dialog.preview.image, values[2])

    def test_worker_initializes_parameters_and_history_metadata(self):
        source = self.data(
            np.ones((3, 2, 2)), "source",
            {"fps": 5, "time_unit": "s", "space_unit": "um", "large": np.ones((3, 2, 2))},
        )
        plan = CalculationPlan(
            "A + 1", [OperandSpec("A", source, axes="THW")],
            metadata_overrides={"value_unit": "count"},
        )
        results = []
        worker = MassDataProcessor()
        worker.processed_result.connect(results.append)

        self.assertTrue(worker.calculation_operation(plan))
        result = results[0]
        self.assertEqual(result.parameters["fps"], 5)
        self.assertEqual(result.parameters["value_unit"], "count")
        self.assertNotIn("large", result.out_processed)
        self.assertEqual(ProcessedData.history[-1].parameters["space_unit"], "um")

    def test_calculation_failure_uses_dedicated_error_signal(self):
        source = self.data(np.ones((2, 3, 4)), "source")
        worker = MassDataProcessor()
        errors = []
        results = []
        worker.calculator_failed.connect(errors.append)
        worker.processed_result.connect(results.append)
        plan = CalculationPlan("A + missing", [OperandSpec("A", source, axes="THW")])

        self.assertFalse(worker.calculation_operation(plan))
        self.assertEqual(results, [])
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0].stage, "多数据运算")
        self.assertIn("expression", errors[0].details)

    def test_shared_tree_does_not_scan_parameter_arrays(self):
        class NoScanArray(np.ndarray):
            def min(self, *args, **kwargs):
                raise AssertionError("tree scanned array minimum")

            def max(self, *args, **kwargs):
                raise AssertionError("tree scanned array maximum")

        source = self.data(np.ones((2, 3, 4)), "source")
        source.parameters["large"] = np.ones((4, 4)).view(NoScanArray)
        dialog = DataTreeViewDialog()
        self.addCleanup(self.close_dialog, dialog)
        dialog.refresh_data()
        self.assertGreater(dialog.tree.topLevelItemCount(), 0)

    def test_shared_tree_preserves_legacy_dialog_shell(self):
        self.data(np.ones((2, 3, 4)), "source")
        dialog = DataTreeViewDialog()
        self.addCleanup(self.close_dialog, dialog)
        dialog.refresh_data()

        self.assertIsInstance(dialog.tree, DataHistoryTreeWidget)
        self.assertGreater(dialog.tree.topLevelItemCount(), 0)
        first = dialog.tree.topLevelItem(0)
        self.assertTrue(first.toolTip(0))

    def test_running_execution_cannot_be_reenabled_by_validation_timer(self):
        source = self.data(np.ones((2, 3, 4)), "source")
        dialog = DataCalculatorDialog([source])
        self.addCleanup(self.close_dialog, dialog)
        dialog.expression.setPlainText("A + 1")
        dialog._execution_running = True

        dialog._validate()

        self.assertTrue(dialog.validation.valid)
        self.assertFalse(dialog.execute_button.isEnabled())

    def test_large_metadata_sequences_are_not_copied(self):
        class FailOnIteration(list):
            def __iter__(self):
                raise AssertionError("large metadata sequence was scanned")

        value = FailOnIteration(range(CalculationMetadataPolicy.MAX_SEQUENCE_ITEMS + 1))
        self.assertIsNone(CalculationMetadataPolicy.safe_value(value))

    def test_tree_summarizes_large_containers_without_stringifying_them(self):
        class FailOnString(list):
            def __str__(self):
                raise AssertionError("large container was stringified")

        value = FailOnString(range(5000))
        self.assertEqual(DataHistoryTreeWidget._format_value(value), "FailOnString，5000 项")

    def test_large_preview_is_downsampled_before_gui_transfer(self):
        source = self.data(np.arange(64, dtype=np.float32).reshape(8, 8), "large-2d")
        dialog = DataCalculatorDialog([source])
        self.addCleanup(self.close_dialog, dialog)
        original_limit = CalculatorPreviewTask.MAX_PREVIEW_PIXELS
        self.addCleanup(setattr, CalculatorPreviewTask, "MAX_PREVIEW_PIXELS", original_limit)
        CalculatorPreviewTask.MAX_PREVIEW_PIXELS = 16

        dialog._request_preview()
        QThreadPool.globalInstance().waitForDone(2000)
        self.app.processEvents(QEventLoop.AllEvents, 50)

        self.assertEqual(dialog.preview.image.shape, (4, 4))
        self.assertIn("2x", dialog.preview_label.text())

    def test_tree_parent_lookup_tolerates_missing_or_invalid_timestamps(self):
        tree = DataHistoryTreeWidget()
        tree.node_map = {1.0: object(), "invalid": object()}
        self.assertIsNone(tree._find_parent(None))
        self.assertIsNone(tree._find_parent("also-invalid"))
        self.assertIsNotNone(tree._find_parent(1.0))


if __name__ == "__main__":
    unittest.main()
