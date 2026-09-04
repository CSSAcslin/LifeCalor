import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

CORE = Path(__file__).resolve().parents[1] / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from dataio.classification import DataCategory, describe_source, describe_value


class DataClassificationTests(unittest.TestCase):
    def test_scalar_vector_line_and_image(self):
        self.assertEqual(describe_value(3).category, DataCategory.SCALAR)
        self.assertEqual(describe_value(np.ones(4)).category, DataCategory.VECTOR)
        self.assertEqual(describe_value(np.ones(4), time_length=4).category, DataCategory.LINEAR)
        self.assertEqual(describe_value(np.ones((3, 4))).category, DataCategory.IMAGE)

    def test_thw_is_video_but_unlabelled_cube_is_matrix(self):
        video = describe_value(shape=(5, 3, 4), dtype="float32", axes=("T", "H", "W"))
        matrix = describe_value(shape=(5, 3, 4), dtype="float32")
        self.assertEqual(video.category, DataCategory.VIDEO)
        self.assertEqual(matrix.category, DataCategory.MATRIX_3D)

    def test_unique_time_length_infers_video(self):
        source = SimpleNamespace(
            datashape=(7, 3, 4), datatype=np.dtype("float32"),
            time_point=np.arange(7), parameters={}, out_processed={},
        )
        self.assertEqual(describe_source(source).category, DataCategory.VIDEO)

    def test_classifier_does_not_scan_array_values(self):
        class NoScan(np.ndarray):
            def min(self, *args, **kwargs):
                raise AssertionError("array values were scanned")
            def max(self, *args, **kwargs):
                raise AssertionError("array values were scanned")
        value = np.ones((2, 3, 4)).view(NoScan)
        self.assertEqual(describe_value(value).shape, (2, 3, 4))


if __name__ == "__main__":
    unittest.main()