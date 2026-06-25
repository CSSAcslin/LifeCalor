import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

for module_name in list(sys.modules):
    if any(module_name == prefix or module_name.startswith(prefix + ".") for prefix in ("scipy", "pywt", "PyQt5", "matplotlib", "PIL", "cv2", "tifffile", "h5py")):
        sys.modules.pop(module_name, None)

from DataProcessor import get_unfolded_data


class UnfoldedDataPolicyTests(unittest.TestCase):
    def test_get_unfolded_data_uses_existing_cache_when_present(self):
        existing = np.arange(12, dtype=np.float32).reshape(3, 4)
        data = SimpleNamespace(out_processed={"unfolded_data": existing})

        self.assertIs(get_unfolded_data(data), existing)

    def test_get_unfolded_data_rebuilds_from_three_dimensional_data_processed(self):
        source = np.arange(12, dtype=np.float32).reshape(3, 2, 2)
        data = SimpleNamespace(out_processed={}, data_processed=source, data_origin=None)

        unfolded = get_unfolded_data(data)

        self.assertEqual(unfolded.shape, (4, 3))
        np.testing.assert_array_equal(unfolded, source.reshape((3, 4)).T)

    def test_get_unfolded_data_rebuilds_from_data_origin(self):
        source = np.arange(12, dtype=np.float32).reshape(3, 2, 2)
        data = SimpleNamespace(out_processed={}, data_origin=source)

        unfolded = get_unfolded_data(data)

        self.assertEqual(unfolded.shape, (4, 3))
        np.testing.assert_array_equal(unfolded, source.reshape((3, 4)).T)


if __name__ == "__main__":
    unittest.main()
