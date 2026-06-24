import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))
import unittest


class AlgorithmRegistryTests(unittest.TestCase):
    def test_registry_registers_and_runs_algorithm(self):
        from AlgorithmRegistry import AlgorithmRegistry

        registry = AlgorithmRegistry()
        registry.register("double", lambda data, scale=2: data * scale, description="multiply input")

        self.assertEqual(registry.list_names(), ["double"])
        self.assertEqual(registry.describe("double"), "multiply input")
        self.assertEqual(registry.run("double", 4, scale=3), 12)

    def test_registry_rejects_duplicate_names(self):
        from AlgorithmRegistry import AlgorithmRegistry

        registry = AlgorithmRegistry()
        registry.register("same", lambda data: data)

        with self.assertRaises(ValueError):
            registry.register("same", lambda data: data)


if __name__ == "__main__":
    unittest.main()

