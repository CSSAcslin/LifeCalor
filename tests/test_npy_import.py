import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

CORE = Path(__file__).resolve().parents[1] / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from DataManager import Data
from ImportManager import ImportManager
from tasks.model import CancellationToken


class NpyImportTests(unittest.TestCase):
    def setUp(self):
        self.previous_history = Data.history.copy()
        Data.history.clear()

    def tearDown(self):
        Data.history.clear()
        Data.history.extend(self.previous_history)

    def test_npy_reader_builds_data_with_dtype_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            source = np.arange(60, dtype=np.complex64).reshape(5, 3, 4)
            path = Path(directory) / "sample.npy"
            np.save(path, source)
            manager = ImportManager()
            manager.set_cancellation_token(CancellationToken())
            results = []
            progress = []
            manager.import_finished.connect(results.append)
            manager.processing_progress_signal.connect(lambda current, total: progress.append((current, total)))

            manager.load_npy(str(path), time_step=0.25, time_unit="ms", space_step=2.0, space_unit="um")

            self.assertEqual(len(results), 1)
            restored = results[0]
            np.testing.assert_array_equal(restored.data_origin, source)
            self.assertEqual(restored.datatype, np.dtype(np.complex64))
            self.assertEqual(restored.parameters["source_shape"], (5, 3, 4))
            self.assertTrue(restored.parameters["external_npy"])
            np.testing.assert_allclose(restored.time_point, np.arange(5) * 0.25)
            self.assertGreaterEqual(len(progress), 2)


if __name__ == "__main__":
    unittest.main()
