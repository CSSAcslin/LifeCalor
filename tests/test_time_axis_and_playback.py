import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from DataProcessor import DataProcessor
from DataManager import ProcessedData
from display.playback_policy import playback_interval_ms
from history.manifest import build_manifest_item, restore_history_item


class TimeAxisAndPlaybackTests(unittest.TestCase):
    def test_zero_or_missing_fps_uses_default_playback_duration(self):
        self.assertEqual(playback_interval_ms(0, 5), 3000)
        self.assertEqual(playback_interval_ms(None, 5), 3000)
        self.assertEqual(playback_interval_ms(20, 5), 50)

    def test_timeline_prefers_original_fps_clock_over_numeric_axis(self):
        from display.playback_policy import timeline_label

        self.assertEqual(timeline_label(25, 20, np.arange(30) / 20), "00:01:05")

    def test_fast_selection_axis_rebuilds_legacy_stft_metadata(self):
        data = SimpleNamespace(time_point=np.array([0.0]), out_processed={"fps": 100.0, "window_step": 25}, source_name="restored-stft")

        axis = DataProcessor.aligned_time_axis(data, 4)

        np.testing.assert_allclose(axis, [0.0, 0.25, 0.5, 0.75])

    def test_manifest_round_trip_preserves_processed_time_axis(self):
        source = np.ones((4, 2, 2), dtype=np.float32)
        processed = ProcessedData(1.0, "stft", "ROI_stft", np.arange(4) * 0.25, source)

        restored = restore_history_item(build_manifest_item(processed))

        np.testing.assert_allclose(restored.time_point, np.arange(4) * 0.25)
        self.assertEqual(restored.parameters, {})


    def test_legacy_manifest_rebuilds_stft_axis_from_metadata(self):
        item = {
            "kind": "ProcessedData",
            "name": "legacy-stft",
            "type_processed": "ROI_stft",
            "shape": [4, 2, 2],
            "dtype": "float32",
            "ndim": 3,
            "metadata": {
                "time_point": {"kind": "ndarray_summary", "shape": [1], "dtype": "float64"},
                "out_processed_metadata": {"fps": 100.0, "window_step": 25},
            },
            "arrays": {},
        }

        restored = restore_history_item(item)

        np.testing.assert_allclose(restored.time_point, [0.0, 0.25, 0.5, 0.75])


if __name__ == "__main__":
    unittest.main()
