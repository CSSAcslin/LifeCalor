import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))


class FakeThread:
    def __init__(self, running=True):
        self.running = running
        self.calls = []

    def isRunning(self):
        return self.running

    def quit(self):
        self.calls.append("quit")
        self.running = False

    def wait(self):
        self.calls.append("wait")

    def deleteLater(self):
        self.calls.append("deleteLater")


class ThreadControllerTests(unittest.TestCase):
    def test_active_thread_uses_running_state_and_deleted_guard(self):
        from ThreadController import is_thread_active

        thread = FakeThread(running=True)

        self.assertTrue(is_thread_active(thread, expected_type=FakeThread, is_deleted=lambda _: False))
        self.assertFalse(is_thread_active(thread, expected_type=FakeThread, is_deleted=lambda _: True))
        self.assertFalse(is_thread_active(FakeThread(running=False), expected_type=FakeThread, is_deleted=lambda _: False))

    def test_stop_thread_quits_waits_and_deletes_running_thread(self):
        from ThreadController import stop_thread

        thread = FakeThread(running=True)
        stopped = stop_thread(thread, expected_type=FakeThread, is_deleted=lambda _: False)

        self.assertTrue(stopped)
        self.assertEqual(thread.calls, ["quit", "wait", "deleteLater"])

    def test_stop_thread_ignores_missing_or_inactive_thread(self):
        from ThreadController import stop_thread

        inactive = FakeThread(running=False)

        self.assertFalse(stop_thread(None, expected_type=FakeThread, is_deleted=lambda _: False))
        self.assertFalse(stop_thread(inactive, expected_type=FakeThread, is_deleted=lambda _: False))
        self.assertEqual(inactive.calls, [])


if __name__ == "__main__":
    unittest.main()
