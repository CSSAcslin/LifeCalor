import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile

CORE = Path(__file__).resolve().parents[1] / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from DataManager import DataManager, render_frame_for_export


class StyledExportTests(unittest.TestCase):
    def test_colormap_export_renders_each_frame_without_full_stack(self):
        source = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        params = {
            "use_colormap": True,
            "colormap": "viridis",
            "auto_range": False,
            "min_value": 0.0,
            "max_value": 23.0,
        }
        with tempfile.TemporaryDirectory() as directory:
            files = DataManager().export_as_tif(source, directory, "styled", True, params)
            self.assertEqual(len(files), 2)
            first = tifffile.imread(files[0])
            expected = render_frame_for_export(source[0], params)
            np.testing.assert_array_equal(first, expected)
            self.assertEqual(first.shape, (3, 4, 4))


if __name__ == "__main__":
    unittest.main()
