import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

import TaskController as task_controller
import TaskState as task_state_module


class FakeThread:
    def __init__(self, running=False):
        self.running = running
        self.started = 0

    def isRunning(self):
        return self.running

    def start(self):
        self.started += 1
        self.running = True


class TaskControllerTests(unittest.TestCase):
    def test_start_thread_marks_task_running_and_starts_inactive_thread(self):
        state = task_state_module.TaskState("calculation")
        thread = FakeThread(running=False)

        started = task_controller.ensure_thread_running(thread, state)

        self.assertTrue(started)
        self.assertTrue(thread.running)
        self.assertEqual(thread.started, 1)
        self.assertEqual(state.status.value, "running")

    def test_start_thread_marks_task_running_without_restarting_active_thread(self):
        state = task_state_module.TaskState("em_processing")
        thread = FakeThread(running=True)

        started = task_controller.ensure_thread_running(thread, state)

        self.assertFalse(started)
        self.assertEqual(thread.started, 0)
        self.assertEqual(state.status.value, "running")


if __name__ == "__main__":
    unittest.main()
