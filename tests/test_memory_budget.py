import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from memory import MemoryBudget, array_nbytes, estimate_resident_bytes


class MemoryBudgetTests(unittest.TestCase):
    def test_array_size_respects_dtype(self):
        self.assertEqual(array_nbytes((10, 20), np.uint8), 200)
        self.assertEqual(array_nbytes((10, 20), np.float32), 800)
        self.assertEqual(array_nbytes((10, 20), np.complex128), 3200)

    def test_budget_checks_resident_and_new_allocation_together(self):
        budget = MemoryBudget.from_megabytes(256)
        with self.assertRaisesRegex(MemoryError, "全局内存预算"):
            budget.ensure(200 * 1024**2, 100 * 1024**2, operation="测试")

    def test_resident_estimator_counts_shared_array_once(self):
        array = np.zeros((8, 8), dtype=np.float32)
        holder = type("Holder", (), {})()
        holder.__dict__["_data_origin_storage"] = array
        holder.__dict__["_image_import_storage"] = array
        holder.__dict__["out_processed"] = {"same": array}
        self.assertEqual(estimate_resident_bytes([holder]), array.nbytes)


if __name__ == "__main__":
    unittest.main()