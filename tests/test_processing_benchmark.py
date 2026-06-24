import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))
import unittest


class ProcessingBenchmarkTests(unittest.TestCase):
    def test_benchmark_result_records_operation_shape_and_elapsed_time(self):
        from ProcessingBenchmark import measure_operation

        result = measure_operation("noop", lambda: [[1, 2], [3, 4]])

        self.assertEqual(result.operation, "noop")
        self.assertEqual(result.output_shape, (2, 2))
        self.assertGreaterEqual(result.elapsed_seconds, 0.0)


if __name__ == "__main__":
    unittest.main()

