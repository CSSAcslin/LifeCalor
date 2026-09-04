import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from calculator import CalculationEngine, CalculationPlan, OperandSpec
from DataManager import Data, ProcessedData


class CalculatorEngineTests(unittest.TestCase):
    def setUp(self):
        Data.history.clear()
        ProcessedData.history.clear()

    @staticmethod
    def data(value, name):
        value = np.asarray(value)
        return Data(value, np.arange(value.shape[0]) if value.ndim == 3 else np.array([0.0]), "test", value, name=name)

    def test_broadcasts_a_2d_frame_across_a_3d_stack(self):
        a = self.data(np.ones((4, 3, 2), dtype=np.float32), "A")
        b = self.data(np.arange(24, dtype=np.float32).reshape(4, 3, 2), "B")
        plan = CalculationPlan("A - B", [OperandSpec("A", a), OperandSpec("B", b, slice_text="2, :, :")])

        validation = CalculationEngine.validate(plan)
        result, _ = CalculationEngine.execute(plan)

        self.assertTrue(validation.valid, validation.error)
        self.assertEqual(validation.output_shape, (4, 3, 2))
        np.testing.assert_array_equal(result, a.data_origin - b.data_origin[2])

    def test_reports_incompatible_shapes_before_execution(self):
        a = self.data(np.ones((4, 3, 2)), "A")
        b = self.data(np.ones((5, 4)), "B")

        validation = CalculationEngine.validate(CalculationPlan("A + B", [OperandSpec("A", a), OperandSpec("B", b)]))

        self.assertFalse(validation.valid)
        self.assertIn("无法广播", validation.error)

    def test_reduction_and_indexing_produce_shape_trace(self):
        a = self.data(np.ones((4, 3, 2)), "A")
        plan = CalculationPlan("mean(A[:, :, 0], axis=0)", [OperandSpec("A", a)])

        validation = CalculationEngine.validate(plan)

        self.assertTrue(validation.valid, validation.error)
        self.assertEqual(validation.output_shape, (3,))
        self.assertGreaterEqual(len(validation.steps), 3)

    def test_rejects_attribute_access_and_unknown_functions(self):
        a = self.data(np.ones((2, 2)), "A")

        attribute = CalculationEngine.validate(CalculationPlan("A.__class__", [OperandSpec("A", a)]))
        function = CalculationEngine.validate(CalculationPlan("open(A)", [OperandSpec("A", a)]))

        self.assertFalse(attribute.valid)
        self.assertFalse(function.valid)

    def test_comparison_can_drive_where(self):
        a = self.data(np.array([[-2.0, 1.0], [3.0, -4.0]]), "A")
        plan = CalculationPlan("where(A > 0, A, 0)", [OperandSpec("A", a)])
        validation = CalculationEngine.validate(plan)
        result, _ = CalculationEngine.execute(plan)
        self.assertTrue(validation.valid, validation.error)
        np.testing.assert_array_equal(result, np.array([[0.0, 1.0], [3.0, 0.0]]))

    def test_transpose_validates_and_executes_axis_order(self):
        a = self.data(np.arange(24).reshape(4, 3, 2), "A")
        plan = CalculationPlan("transpose(A, axes=(1, 2, 0))", [OperandSpec("A", a)])
        validation = CalculationEngine.validate(plan)
        result, _ = CalculationEngine.execute(plan)
        self.assertTrue(validation.valid, validation.error)
        self.assertEqual(validation.output_shape, (3, 2, 4))
        np.testing.assert_array_equal(result, np.transpose(a.data_origin, (1, 2, 0)))

    def test_validation_rejects_function_calls_execution_would_reject(self):
        a = self.data(np.ones((2, 3), dtype=np.float32), "A")
        operands = [OperandSpec("A", a)]

        for expression in (
            "abs(A, 1)",
            "mean(A, 0)",
            "mean(A, axis=(0, 0))",
            "clip(A, 0, 1, 2)",
        ):
            with self.subTest(expression=expression):
                validation = CalculationEngine.validate(CalculationPlan(expression, operands))
                self.assertFalse(validation.valid)

    def test_validation_dtype_matches_numpy_execution(self):
        float_data = self.data(np.ones((2, 3), dtype=np.float32), "float")
        int_data = self.data(np.ones((2, 3), dtype=np.int16), "int")
        cases = (
            ("A / A", float_data),
            ("sqrt(A)", float_data),
            ("sum(A, axis=0)", int_data),
        )

        for expression, source in cases:
            with self.subTest(expression=expression):
                plan = CalculationPlan(expression, [OperandSpec("A", source)])
                result, validation = CalculationEngine.execute(plan)
                self.assertEqual(validation.output_dtype, str(result.dtype))


if __name__ == "__main__":
    unittest.main()
