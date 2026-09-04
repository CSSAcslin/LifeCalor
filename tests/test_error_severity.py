import sys
import unittest
from pathlib import Path

CORE = Path(__file__).resolve().parents[1] / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from diagnostics import AppError
from diagnostics.broker import ErrorBroker


class FakePresenter:
    def __init__(self):
        self.presented = []

    def enqueue(self, parent, error):
        self.presented.append((parent, error))

    def reset(self):
        self.presented.clear()


class ErrorSeverityTests(unittest.TestCase):
    def setUp(self):
        self.presenter = FakePresenter()
        self.broker = ErrorBroker(dedup_seconds=30, presenter=self.presenter)

    def test_warning_is_logged_without_popup(self):
        with self.assertLogs(level="WARNING"):
            self.broker.report(None, AppError("提醒", "可继续运行", severity="warning"))
        self.assertEqual(self.presenter.presented, [])

    def test_error_uses_single_presenter_request(self):
        self.broker.report(None, AppError("失败", "无法继续", severity="error"))
        self.assertEqual(len(self.presenter.presented), 1)

    def test_same_error_object_is_reported_once(self):
        error = AppError("失败", "无法继续", severity="error")
        self.broker.report(None, error)
        self.broker.report(None, error)
        self.assertEqual(len(self.presenter.presented), 1)

    def test_equivalent_error_burst_is_deduplicated(self):
        with self.assertLogs(level="WARNING"):
            for _ in range(20):
                self.broker.report(None, AppError("失败", "同一错误", stage="测试", severity="error"))
        self.assertEqual(len(self.presenter.presented), 1)


if __name__ == "__main__":
    unittest.main()