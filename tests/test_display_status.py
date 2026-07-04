import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from display.status import render_status_update


class DisplayStatusTests(unittest.TestCase):
    def test_only_failed_render_status_updates_main_status(self):
        self.assertEqual(render_status_update("failed", "bad frame"), ("bad frame", "failed"))
        self.assertIsNone(render_status_update("rendering", "rendering frame"))
        self.assertIsNone(render_status_update("completed", "done"))


if __name__ == "__main__":
    unittest.main()
