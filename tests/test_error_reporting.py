import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from diagnostics import AppError, format_exception_details


class ErrorReportingTests(unittest.TestCase):
    def test_exception_details_include_stage_type_data_shape_and_dtype(self):
        data = SimpleNamespace(name="demo", imageshape=(3, 4, 5), datatype="float32")
        try:
            raise ValueError("bad frame")
        except ValueError as exc:
            details = format_exception_details(exc, stage="anchor distribution", data=data)

        self.assertIn("阶段: anchor distribution", details)
        self.assertIn("异常: ValueError: bad frame", details)
        self.assertIn("name: demo", details)
        self.assertIn("imageshape: (3, 4, 5)", details)
        self.assertIn("dtype: float32", details)
        self.assertIn("Traceback:", details)

    def test_app_error_defaults_to_critical(self):
        error = AppError("标题", "消息")

        self.assertEqual(error.severity, "critical")
        self.assertEqual(error.context, {})


if __name__ == "__main__":
    unittest.main()
