import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from DataManager import Data, ProcessedData
from display.source import DisplaySourceFactory


class DisplaySourceTests(unittest.TestCase):
    def setUp(self):
        Data.clear_history(remove_cache=False)
        ProcessedData.clear_history(remove_cache=False)

    def test_data_source_keeps_reference_and_returns_single_frame(self):
        raw = np.arange(24, dtype=np.float32).reshape(3, 2, 4)
        image = raw.mean(axis=0)
        data = Data(raw, np.arange(3), "test", image)

        source = DisplaySourceFactory.from_data(data)

        self.assertEqual(source.shape, raw.shape)
        self.assertEqual(source.frame_count, 3)
        self.assertTrue(np.shares_memory(source.get_frame(1), raw))
        np.testing.assert_array_equal(source.get_frame(1), raw[1])

    def test_processed_out_source_returns_requested_payload_frame(self):
        raw = np.arange(24, dtype=np.float32).reshape(3, 2, 4)
        processed = ProcessedData(1.0, "processed", "ROI_stft", np.arange(3), raw, out_processed={"extra": raw + 10})

        source = DisplaySourceFactory.from_data(processed, key="extra")

        self.assertEqual(source.source_type, "ProcessedData")
        self.assertEqual(source.frame_count, 3)
        np.testing.assert_array_equal(source.get_frame(2), raw[2] + 10)

    def test_two_dimensional_source_has_one_frame(self):
        raw = np.arange(12, dtype=np.float32).reshape(3, 4)
        data = Data(raw, np.arange(1), "test", raw)

        source = DisplaySourceFactory.from_data(data)

        self.assertEqual(source.frame_count, 1)
        np.testing.assert_array_equal(source.get_frame(0), raw)


if __name__ == "__main__":
    unittest.main()
