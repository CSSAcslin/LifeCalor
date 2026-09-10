import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))


class MainWindowArchitectureTests(unittest.TestCase):
    def setUp(self):
        self.source = (CORE / "MainWindow.py").read_text(encoding="utf-8")

    def test_mainwindow_uses_extracted_parameter_store(self):
        self.assertIn("from settings import load_param_group", self.source)
        self.assertIn("return load_param_group(", self.source)

    def test_mainwindow_uses_unified_processing_and_task_components(self):
        self.assertIn("from processing import ProcessingController, ResultRouter", self.source)
        self.assertIn("from tasks.panel import TaskPanel", self.source)
        self.assertIn("self.processing_controller = ProcessingController", self.source)
        self.assertIn("self.result_router = ResultRouter", self.source)
        self.assertIn("self.task_panel = TaskPanel", self.source)
        self.assertNotIn("from TaskState import TaskState", self.source)
        self.assertNotIn("self.task_states", self.source)
        self.assertNotIn("from TaskController import ensure_thread_running", self.source)

    def test_mainwindow_delegates_thread_export_and_result_routing(self):
        self.assertIn("from ThreadController import is_thread_active as thread_is_active", self.source)
        self.assertIn("return thread_is_active(", self.source)
        self.assertIn("from exporting import ExportController", self.source)
        self.assertIn("self.export_controller.export_data()", self.source)
        self.assertIn("self.export_controller.export_em_data(result)", self.source)
        result_block = self.source[self.source.index("    def processed_result"):self.source.index("    def draw_result")]
        self.assertIn("self.result_router.route(data)", result_block)
        self.assertNotIn("match process_type", result_block)

    def test_mainwindow_delegates_selection_export_history_and_task_start(self):
        self.assertIn("from selection import SelectionController", self.source)
        self.assertIn("from exporting import ExportController", self.source)
        self.assertIn("from history import HistoryController", self.source)
        self.assertIn("self.selection_controller.select_data(", self.source)
        self.assertIn("self.selection_controller.select_roi(", self.source)
        self.assertIn("self.processing_controller.begin(thread_name, task_key)", self.source)
        export_source = (CORE / "exporting" / "controller.py").read_text(encoding="utf-8")
        self.assertIn("save_dataframe(", export_source)

    def test_calculator_is_modeless_and_has_one_independent_task(self):
        block = self.source[self.source.index("    def process_math"):self.source.index("    def data_crop")]
        self.assertIn("DataCalculatorDialog(", block)
        self.assertIn("            [],", block)
        self.assertIn("dialog.show()", block)
        self.assertIn("dialog.execute_requested.connect", block)
        self.assertNotIn("dialog.exec_()", block)
        self.assertIn('"多数据运算", "calculator"', block)
        self.assertIn("cancel_callback=self.mass_data_processor.stop", block)
        self.assertNotIn('ensure_task_thread_running("avi_thread", "em_processing")', block)

    def test_controller_packages_exist(self):
        expected = [
            "selection/controller.py", "exporting/controller.py", "history/controller.py",
            "settings/parameter_store.py", "progress/policy.py", "processing/controller.py",
            "processing/result_router.py", "tasks/panel.py", "memory/budget.py",
            "performance/monitor.py", "importing/registry.py",
        ]
        for relative in expected:
            self.assertTrue((CORE / relative).exists(), relative)

    def test_dead_cache_dialog_and_config_stubs_are_removed(self):
        extra = (CORE / "ExtraDialog.py").read_text(encoding="utf-8")
        self.assertNotIn("class CacheSettingsDialog", extra)
        self.assertNotIn("    def save_config(self):", self.source)
        self.assertNotIn("    def load_config(self, preset_name):", self.source)


if __name__ == "__main__":
    unittest.main()