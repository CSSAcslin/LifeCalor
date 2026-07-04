import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from display.canvas_signals import disconnect_canvas_signal


class FakeSignal:
    def __init__(self):
        self.disconnected = False

    def disconnect(self, slot):
        self.disconnected = True


class CanvasSignalTests(unittest.TestCase):
    def test_disconnect_ignores_unconnected_signal(self):
        class BrokenSignal:
            def disconnect(self, slot):
                raise TypeError("not connected")

        disconnect_canvas_signal(BrokenSignal(), object())

    def test_disconnect_calls_signal_disconnect(self):
        signal = FakeSignal()
        slot = object()
        disconnect_canvas_signal(signal, slot)
        self.assertTrue(signal.disconnected)


if __name__ == "__main__":
    unittest.main()
