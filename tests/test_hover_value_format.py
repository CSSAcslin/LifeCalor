import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from display.hover import format_hover_value


class HoverValueFormatTests(unittest.TestCase):
    def test_small_values_use_scientific_not_zero(self):
        self.assertEqual(format_hover_value(1.23e-4), "1.23e-04")
        self.assertEqual(format_hover_value(-4.56e-5), "-4.56e-05")

    def test_large_values_do_not_force_decimal_noise(self):
        self.assertEqual(format_hover_value(65536), "65536")
        self.assertEqual(format_hover_value(12345.678), "12345.7")

    def test_none_and_nan_are_display_safe(self):
        self.assertEqual(format_hover_value(None), "-")
        self.assertEqual(format_hover_value(float("nan")), "nan")


if __name__ == "__main__":
    unittest.main()
