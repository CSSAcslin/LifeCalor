import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from DataManager import Data, DataManager
from DataProcessor import MassDataProcessor


class ProcessingCancellationTests(unittest.TestCase):
    def setUp(self):
        Data.history.clear()
        self.data = Data(
            np.ones((3, 2, 2), dtype=np.float32), np.arange(3), "test",
            np.ones((3, 2, 2), dtype=np.float32), {"fps": 10}, "cancel",
        )

    def tearDown(self):
        Data.history.clear()

    def test_mass_processor_emits_cancelled_instead_of_silent_return(self):
        worker = MassDataProcessor()
        emitted = []
        worker.processing_cancelled_signal.connect(lambda: emitted.append(True))
        worker.abortion = True

        result = worker.pre_process(self.data, bg_num=1, unfold=False)

        self.assertFalse(result)
        self.assertEqual(emitted, [True])

    def test_roi_manager_reports_cancelled_token(self):
        manager = DataManager()
        emitted = []
        manager.processing_cancelled_signal.connect(lambda: emitted.append(True))
        manager.cancellation_token.cancel()

        result = manager.ROI_processed(
            self.data, np.ones((2, 2), dtype=bool), 1.0, False, False, 1.0
        )

        self.assertFalse(result)
        self.assertEqual(emitted, [True])


if __name__ == "__main__":
    unittest.main()