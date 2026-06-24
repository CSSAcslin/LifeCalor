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


if __name__ == "__main__":
    unittest.main()
