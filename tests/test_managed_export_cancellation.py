import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

CORE = Path(__file__).resolve().parents[1] / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from DataManager import DataManager
from tasks.model import CancellationToken


class _CanvasData:
    def __init__(self, array):
        self.image_backup = array
        self.image_data = array[0]


class ManagedExportCancellationTests(unittest.TestCase):
    def test_cancelled_export_removes_staging_and_publishes_no_files(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = DataManager()
            token = CancellationToken()
            manager.set_cancellation_token(token)
            manager.data_progress_signal.connect(lambda current, total: token.cancel() if 0 < current < total else None)
            cancelled = []
            manager.export_cancelled.connect(lambda: cancelled.append(True))

            result = manager.export_data(
                _CanvasData(np.arange(5 * 8 * 8, dtype=np.float32).reshape(5, 8, 8)),
                directory,
                "cancelled",
                "tif",
                True,
                {},
            )

            self.assertIsNone(result)
            self.assertEqual(cancelled, [True])
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
