import os
import pickle
import sys
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from compute.model import (
    BackendPreference,
    ComputeRequest,
    PrecisionPolicy,
    ResourceBudget,
)
from compute.planner import normalize_progress, plan_compute, resolve_precision
from memory.budget import array_nbytes


def request(**changes):
    values = dict(
        task_id="task-1",
        attempt_id=0,
        algorithm="stft",
        data_id="data-1",
        shape=(100, 20, 30),
        dtype="float32",
        axes="THW",
        source=object(),
        parameters={},
        backend=BackendPreference.AUTO,
        precision=PrecisionPolicy.COMPATIBILITY,
    )
    values.update(changes)
    return ComputeRequest(**values)


class ComputePlannerTests(unittest.TestCase):
    def test_request_freezes_shape_dtype_and_nested_parameters(self):
        params = {"scales": [1, 2, 3], "nested": {"enabled": True}}
        item = request(dtype=np.dtype("<f4"), parameters=params)
        params["scales"].append(4)

        self.assertEqual(item.dtype, "float32")
        self.assertEqual(item.parameters["scales"], (1, 2, 3))
        with self.assertRaises(TypeError):
            item.parameters["new"] = 1
        with self.assertRaises(FrozenInstanceError):
            item.axes = "HW"

    def test_request_is_pickle_safe_for_spawned_workers(self):
        item = request(parameters={"nested": {"values": [1, 2]}})
        restored = pickle.loads(pickle.dumps(item))
        self.assertEqual(restored.parameters["nested"]["values"], (1, 2))

    def test_huge_byte_estimate_uses_python_integer_without_allocation(self):
        shape = (100000, 1024, 1280)
        self.assertEqual(array_nbytes(shape, np.complex128), 2097152000000)
        self.assertGreater(array_nbytes(shape, np.float32), 2 ** 31)

    def test_auto_backend_is_conservatively_cpu(self):
        plan = plan_compute(
            request(),
            ResourceBudget(host_limit_bytes=256 * 1024 * 1024),
        )
        self.assertEqual(plan.actual_backend, "cpu")
        self.assertIn("保守", plan.backend_reason)
        self.assertLessEqual(plan.peak_host_bytes, 256 * 1024 * 1024)

    def test_gpu_request_falls_back_or_fails_according_to_policy(self):
        item = request(backend=BackendPreference.GPU)
        fallback = plan_compute(
            item,
            ResourceBudget(host_limit_bytes=256 * 1024 * 1024),
            allow_cpu_fallback=True,
        )
        self.assertEqual(fallback.actual_backend, "cpu")
        self.assertIn("回退", fallback.backend_reason)
        with self.assertRaisesRegex(RuntimeError, "严格 GPU"):
            plan_compute(
                item,
                ResourceBudget(host_limit_bytes=256 * 1024 * 1024),
                allow_cpu_fallback=False,
            )

    def test_lifetime_rejects_unverified_single_precision(self):
        with self.assertRaisesRegex(ValueError, "尚未验证单精度"):
            resolve_precision("lifetime_single", "float32", PrecisionPolicy.SINGLE)

    def test_complex_lifetime_requires_explicit_observable(self):
        with self.assertRaisesRegex(ValueError, "不接受复数"):
            resolve_precision("lifetime", "complex64", PrecisionPolicy.COMPATIBILITY)

    def test_progress_scaling_handles_counts_above_qt_integer_range(self):
        total = 2 ** 50
        self.assertEqual(normalize_progress(total // 2, total), (5000, 10000))
        self.assertEqual(normalize_progress(total * 2, total), (10000, 10000))

    def test_disk_budget_is_checked_before_large_result(self):
        item = request(
            shape=(100000, 1024, 1280),
            parameters={"output_shape": (100000, 1024, 1280)},
        )
        with self.assertRaisesRegex(OSError, "可用空间不足"):
            plan_compute(
                item,
                ResourceBudget(
                    host_limit_bytes=4 * 1024 ** 3,
                    disk_free_bytes=1024,
                ),
            )


if __name__ == "__main__":
    unittest.main()
