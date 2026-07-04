import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from display.render_controller import RenderRequestState


class RenderControllerTests(unittest.TestCase):
    def test_busy_request_is_coalesced_to_latest_frame(self):
        state = RenderRequestState()
        first = state.start_or_queue(1)
        second = state.start_or_queue(2)
        third = state.start_or_queue(3)
        self.assertEqual(first, 1)
        self.assertIsNone(second)
        self.assertIsNone(third)
        self.assertEqual(state.pending_frame_index, 3)

    def test_stale_result_is_ignored(self):
        state = RenderRequestState()
        request_id = state.next_request_id()
        newer = state.next_request_id()
        self.assertTrue(state.is_stale(request_id))
        self.assertFalse(state.is_stale(newer))


if __name__ == "__main__":
    unittest.main()
