import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np

CORE = Path(__file__).resolve().parents[1] / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from importing.model import ImportRequest
from importing.readers import AviImporter


class VideoImportCountTests(unittest.TestCase):
    def _read_video(self, reported_count, decoded_count):
        metadata = {
            cv2.CAP_PROP_FRAME_COUNT: reported_count,
            cv2.CAP_PROP_FRAME_HEIGHT: 2,
            cv2.CAP_PROP_FRAME_WIDTH: 3,
            cv2.CAP_PROP_FPS: 50,
        }
        probe, reader = Mock(), Mock()
        probe.isOpened.return_value = True
        probe.get.side_effect = lambda key: metadata[key]
        frames = [
            (True, np.full((2, 3, 3), index, dtype=np.uint8))
            for index in range(decoded_count)
        ]
        reader.read.side_effect = frames + [(False, None)]
        with patch("importing.readers.cv2.VideoCapture", side_effect=[probe, reader]):
            result = AviImporter().read(
                ImportRequest("avi", Path("sample.avi"), {"time_step": 0.02})
            )
        probe.release.assert_called_once()
        reader.release.assert_called_once()
        return result

    def test_short_decode_reports_counts_warning_and_actual_time_axis(self):
        result = self._read_video(reported_count=4, decoded_count=2)
        self.assertEqual(result.data.shape, (2, 2, 3))
        self.assertEqual(result.parameters["reported_frame_count"], 4)
        self.assertEqual(result.parameters["decoded_frame_count"], 2)
        self.assertIn("报告 4 帧", result.parameters["import_warning"])
        self.assertIn("实际解码 2 帧", result.parameters["import_warning"])
        np.testing.assert_allclose(result.time_point, [0.0, 0.02])

    def test_complete_decode_records_counts_without_short_decode_warning(self):
        result = self._read_video(reported_count=2, decoded_count=2)
        self.assertEqual(result.parameters["reported_frame_count"], 2)
        self.assertEqual(result.parameters["decoded_frame_count"], 2)
        self.assertNotIn("import_warning", result.parameters)


if __name__ == "__main__":
    unittest.main()
