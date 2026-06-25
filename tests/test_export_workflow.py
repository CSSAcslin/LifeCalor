import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

import ExportWorkflow as export_workflow
import TaskState as task_state_module


class FakeFrame:
    def __init__(self):
        self.saved = []

    def to_csv(self, path, **kwargs):
        self.saved.append((path, kwargs))


class ExportWorkflowTests(unittest.TestCase):
    def test_save_dataframe_marks_export_completed(self):
        state = task_state_module.TaskState("export")
        frame = FakeFrame()

        result = export_workflow.save_dataframe(frame, "out.csv", True, state)

        self.assertTrue(result)
        self.assertEqual(state.status.value, "completed")
        self.assertEqual(frame.saved, [("out.csv", {"index": False, "header": True})])

    def test_save_dataframe_marks_export_failed_on_error(self):
        class BrokenFrame:
            def to_csv(self, path, **kwargs):
                raise RuntimeError("disk full")

        state = task_state_module.TaskState("export")

        result = export_workflow.save_dataframe(BrokenFrame(), "out.txt", False, state)

        self.assertFalse(result)
        self.assertEqual(state.status.value, "failed")
        self.assertIn("disk full", state.error)


if __name__ == "__main__":
    unittest.main()
