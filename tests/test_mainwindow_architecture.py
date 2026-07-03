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

        self.assertIn("from ParameterStore import load_param_group", source)
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
        self.assertIn("from ExportPolicy import can_export_em_data, prepare_dataframe_for_export", source)
        self.assertIn("prepare_dataframe_for_export(", source)
        self.assertIn("can_export_em_data(", source)


    def test_mainwindow_delegates_selection_export_workflow_and_task_start(self):
        source = (CORE / "MainWindow.py").read_text(encoding="utf-8")

        self.assertIn("from selection import SelectionController", source)
        self.assertIn("from ExportWorkflow import save_dataframe", source)
        self.assertIn("from TaskController import ensure_thread_running", source)
        self.assertIn("self.selection_controller.select_data(", source)
        self.assertIn("self.selection_controller.select_roi(", source)
        self.assertIn("save_dataframe(", source)
        self.assertIn("ensure_task_thread_running(", source)

    def test_selection_modules_live_in_selection_package(self):
        self.assertTrue((CORE / "selection" / "__init__.py").exists())
        self.assertTrue((CORE / "selection" / "policy.py").exists())
        self.assertTrue((CORE / "selection" / "roi.py").exists())
        self.assertTrue((CORE / "selection" / "controller.py").exists())

if __name__ == "__main__":
    unittest.main()
