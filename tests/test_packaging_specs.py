import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"


class PackagingSpecTests(unittest.TestCase):
    def test_cpu_spec_excludes_optional_cuda_runtime(self):
        source = (CORE / "LifeCalor-CPU.spec").read_text(encoding="utf-8")
        tree = ast.parse(source)
        self.assertIn("'cupy'", source)
        self.assertIn("'cupyx'", source)
        self.assertIn("'cupy_backends'", source)
        self.assertIn("'compute.backends.cuda'", source)
        self.assertIn("name='LifeCalor-CPU'", source)
        self.assertIsInstance(tree, ast.Module)

    def test_full_and_cpu_specs_remain_separate(self):
        full = (CORE / "LifeCalor.spec").read_text(encoding="utf-8")
        cpu = (CORE / "LifeCalor-CPU.spec").read_text(encoding="utf-8")
        self.assertIn("name='LifeCalor'", full)
        self.assertIn("'compute.backends.cuda'", full)
        self.assertIn("for package in ('cupy', 'cupyx', 'cupy_backends'", full)
        self.assertIn("collect_runtime_package(package)", full)
        self.assertIn("'compute.backends.cuda'", cpu)
        self.assertNotEqual(full, cpu)


if __name__ == "__main__":
    unittest.main()
