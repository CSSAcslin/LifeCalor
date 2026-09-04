import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))


class MainWindowArchitectureTests(unittest.TestCase):
    def test_mainwindow_uses_extracted_parameter_store(self):
        source = (CORE / "MainWindow.py").read_text(encoding="utf-8")

        self.assertIn("from settings import load_param_group", source)
        self.assertIn("return load_param_group(", source)

    def test_mainwindow_initializes_named_task_states(self):
        source = (CORE / "MainWindow.py").read_text(encoding="utf-8")

        self.assertIn("from TaskState import TaskState", source)
        self.assertIn("self.task_states", source)
        self.assertIn("TaskState(\"import\")", source)
        self.assertIn("TaskState(\"calculation\")", source)
        self.assertIn("TaskState(\"em_processing\")", source)
        self.assertIn("TaskState(\"export\")", source)

    def test_mainwindow_delegates_thread_and_export_policy(self):
        source = (CORE / "MainWindow.py").read_text(encoding="utf-8")

        self.assertIn("from ThreadController import is_thread_active as thread_is_active", source)
        self.assertIn("return thread_is_active(", source)
        self.assertIn("stop_qthread(", source)
        self.assertIn("from exporting import ExportController", source)
        self.assertIn("self.export_controller.export_data()", source)
        self.assertIn("self.export_controller.export_em_data(result)", source)


    def test_mainwindow_delegates_selection_export_workflow_and_task_start(self):
        source = (CORE / "MainWindow.py").read_text(encoding="utf-8")

        self.assertIn("from selection import SelectionController", source)
        self.assertIn("from exporting import ExportController", source)
        self.assertIn("from history import HistoryController", source)
        self.assertIn("from TaskController import ensure_thread_running", source)
        self.assertIn("self.selection_controller.select_data(", source)
        self.assertIn("self.selection_controller.select_roi(", source)
        self.assertIn("ensure_thread_running(", source)
        export_source = (CORE / "exporting" / "controller.py").read_text(encoding="utf-8")
        self.assertIn("save_dataframe(", export_source)

    def test_calculator_is_modeless_and_starts_without_implicit_source(self):
        source = (CORE / "MainWindow.py").read_text(encoding="utf-8")
        block = source[source.index("    def process_math"):source.index("    def data_crop")]

        self.assertIn("DataCalculatorDialog(", block)
        self.assertIn("            [],", block)
        self.assertIn("dialog.show()", block)
        self.assertIn("dialog.execute_requested.connect", block)
        self.assertNotIn("dialog.exec_()", block)
        self.assertIn("self.task_coordinator.complete", block)
        self.assertIn("TaskStatus.CANCELLING", block)
        self.assertIn("return False", block)

    def test_selection_modules_live_in_selection_package(self):
        self.assertTrue((CORE / "selection" / "__init__.py").exists())
        self.assertTrue((CORE / "selection" / "policy.py").exists())
        self.assertTrue((CORE / "selection" / "roi.py").exists())
        self.assertTrue((CORE / "selection" / "controller.py").exists())

    def test_mainwindow_delegates_export_and_history_controllers(self):
        source = (CORE / "MainWindow.py").read_text(encoding="utf-8")
        self.assertIn("from exporting import ExportController", source)
        self.assertIn("from history import HistoryController", source)
        self.assertIn("self.export_controller = ExportController(self)", source)
        self.assertIn("self.history_controller = HistoryController(self)", source)
        self.assertIn("self.selection_controller = SelectionController(self)", source)
        export_block = source[source.index("def export_image"):source.index("def data_history_view")]
        self.assertIn("self.export_controller.export_image()", export_block)
        self.assertIn("self.export_controller.export_data()", export_block)
        self.assertIn("self.export_controller.export_em_data(result)", export_block)
        self.assertNotIn("QFileDialog.getSaveFileName", export_block)
        history_block = source[source.index("def data_history_view"):source.index("def data_history_clear")]
        self.assertIn("self.history_controller.data_history_view()", history_block)
        self.assertIn("self.history_controller.process_history_view()", history_block)
        self.assertIn("self.history_controller.load_cached_history_async(target, attr_name)", history_block)
        self.assertNotIn("ArrayLoadWorker(target)", history_block)

    def test_exporting_history_settings_progress_packages_exist(self):
        self.assertTrue((CORE / "exporting" / "__init__.py").exists())
        self.assertTrue((CORE / "exporting" / "controller.py").exists())
        self.assertTrue((CORE / "exporting" / "policy.py").exists())
        self.assertTrue((CORE / "exporting" / "workflow.py").exists())
        self.assertTrue((CORE / "history" / "__init__.py").exists())
        self.assertTrue((CORE / "history" / "controller.py").exists())
        self.assertTrue((CORE / "settings" / "parameter_store.py").exists())
        self.assertTrue((CORE / "progress" / "policy.py").exists())


if __name__ == "__main__":
    unittest.main()
