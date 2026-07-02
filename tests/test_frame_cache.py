import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from FrameCache import FrameCache, frame_cache_key
from FrameRenderer import FrameRenderParams, RenderedFrame


class FrameCacheTests(unittest.TestCase):
    def test_cache_key_changes_when_render_params_change(self):
        a = frame_cache_key("source", 1, FrameRenderParams(use_colormap=False, min_value=0, max_value=1))
        b = frame_cache_key("source", 1, FrameRenderParams(use_colormap=True, colormap="jet", min_value=0, max_value=1))

        self.assertNotEqual(a, b)

    def test_lru_cache_evicts_oldest_frame(self):
        cache = FrameCache(capacity=2)
        frame = RenderedFrame(np.zeros((2, 2), dtype=np.uint8), mode="L")
        cache.put(("a", 0), frame)
        cache.put(("b", 0), frame)
        cache.get(("a", 0))
        cache.put(("c", 0), frame)

        self.assertIsNotNone(cache.get(("a", 0)))
        self.assertIsNone(cache.get(("b", 0)))
        self.assertIsNotNone(cache.get(("c", 0)))


if __name__ == "__main__":
    unittest.main()
