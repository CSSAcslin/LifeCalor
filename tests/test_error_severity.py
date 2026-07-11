import sys
import unittest
from pathlib import Path
from unittest.mock import patch

CORE = Path(__file__).resolve().parents[1] / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from diagnostics import AppError, show_app_error


class ErrorSeverityTests(unittest.TestCase):
    @patch("diagnostics.reporter.QMessageBox.critical")
    def test_warning_is_logged_without_popup(self, critical):
        with self.assertLogs(level="WARNING"):
            show_app_error(None, AppError("提醒", "可继续运行", severity="warning"))
        critical.assert_not_called()

    @patch("diagnostics.reporter.QMessageBox.critical")
    def test_error_uses_popup(self, critical):
        show_app_error(None, AppError("失败", "无法继续", severity="error"))
        critical.assert_called_once_with(None, "失败", "无法继续")


if __name__ == "__main__":
    unittest.main()
