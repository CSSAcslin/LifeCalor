import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from PyQt5.QtWidgets import QApplication, QComboBox
from ResultDisplayWidget import ResultDisplayWidget


class LifetimeResultFieldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_double_lifetime_heatmap_switches_named_field_without_recompute(self):
        tau1 = np.full((2, 3), 2.0)
        tau2 = np.full((2, 3), 12.0)
        data = SimpleNamespace(
            data_processed=tau1,
            out_processed={
                "tau1_map": tau1,
                "tau2_map": tau2,
                "r_squared_map": np.ones((2, 3)),
                "fit_status": np.ones((2, 3), dtype=np.uint8),
                "result_fields": (
                    "tau1_map",
                    "tau2_map",
                    "r_squared_map",
                    "fit_status",
                ),
                "active_result_field": "tau1_map",
            },
        )
        widget = ResultDisplayWidget()
        widget.display_distribution_map(data)
        selector = widget.currentWidget().findChild(
            QComboBox, "lifetimeResultField"
        )
        self.assertIsNotNone(selector)
        selector.setCurrentText("tau2_map")
        self.app.processEvents()

        self.assertEqual(
            data.out_processed["active_result_field"], "tau2_map"
        )
        np.testing.assert_array_equal(
            widget.current_dataframe.to_numpy(), tau2
        )
        widget.close()


if __name__ == "__main__":
    unittest.main()
