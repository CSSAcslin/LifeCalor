import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))
import unittest


class TaskStateTests(unittest.TestCase):
    def test_task_state_transitions_record_progress_and_errors(self):
        from TaskState import TaskStatus, TaskState

        state = TaskState("stft")
        self.assertEqual(state.status, TaskStatus.IDLE)

        state.start(total=100)
        state.advance(25)
        self.assertEqual(state.status, TaskStatus.RUNNING)
        self.assertEqual(state.current, 25)
        self.assertEqual(state.total, 100)

        state.fail("boom")
        self.assertEqual(state.status, TaskStatus.FAILED)
        self.assertEqual(state.error, "boom")

    def test_task_state_cannot_advance_after_completion(self):
        from TaskState import TaskState

        state = TaskState("export")
        state.start(total=2)
        state.complete()

        with self.assertRaises(RuntimeError):
            state.advance(1)


if __name__ == "__main__":
    unittest.main()

