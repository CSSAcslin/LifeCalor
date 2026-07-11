import sys
import unittest
from pathlib import Path

CORE = Path(__file__).resolve().parents[1] / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from tasks.model import TaskRegistry, TaskStatus


class TaskRegistryTests(unittest.TestCase):
    def test_registry_tracks_multiple_independent_tasks(self):
        registry = TaskRegistry()
        first = registry.create("读取缓存", "cache_read")
        second = registry.create("导出视频", "export", foreground=False)

        first.start(100)
        second.start(20)
        first.advance(25, 100)
        second.advance(4, 20)

        self.assertEqual(len(registry.active()), 2)
        self.assertEqual(first.current, 25)
        self.assertEqual(second.current, 4)
        self.assertIs(registry.foreground(), first)

    def test_cancel_foreground_only_sets_token_and_does_not_wait(self):
        registry = TaskRegistry()
        foreground = registry.create("计算", "processing")
        background = registry.create("渲染", "render", foreground=False)
        foreground.start()
        background.start()

        cancelled = registry.request_cancel_foreground()

        self.assertIs(cancelled, foreground)
        self.assertEqual(foreground.status, TaskStatus.CANCELLING)
        self.assertTrue(foreground.token.is_cancelled)
        self.assertEqual(background.status, TaskStatus.RUNNING)

    def test_cancel_callback_can_notify_worker(self):
        called = []
        registry = TaskRegistry()
        task = registry.create("写缓存", "cache_write")
        task.cancel_callback = lambda: called.append(True)
        task.start()

        self.assertTrue(registry.request_cancel(task.task_id))
        self.assertEqual(called, [True])


if __name__ == "__main__":
    unittest.main()
