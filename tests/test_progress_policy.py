import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from progress.policy import normalize_progress


class ProgressPolicyTests(unittest.TestCase):
    def test_normal_progress_values_are_preserved(self):
        current, maximum, percent = normalize_progress(25, 100)
        self.assertEqual((current, maximum), (25, 100))
        self.assertEqual(percent, 25.0)

    def test_large_byte_progress_is_scaled_to_qprogressbar_safe_range(self):
        current, maximum, percent = normalize_progress(3 * 1024 ** 3, 6 * 1024 ** 3)
        self.assertEqual(maximum, 1000)
        self.assertEqual(current, 500)
        self.assertEqual(percent, 50.0)

    def test_reset_progress_is_preserved(self):
        self.assertEqual(normalize_progress(-1, None), (-1, None, 0.0))


if __name__ == "__main__":
    unittest.main()
