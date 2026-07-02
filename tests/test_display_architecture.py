import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"


class DisplayArchitectureTests(unittest.TestCase):
    def test_imaging_create_image_does_not_copy_full_source_stack(self):
        source = (CORE / "DataManager.py").read_text(encoding="utf-8")
        create_image = source[source.index("def create_image"):source.index("    def apply_ROI", source.index("def create_image"))]
        self.assertNotIn("data_obj.data_origin.copy()", create_image)
        self.assertNotIn("data_obj.data_processed.copy()", create_image)
        self.assertIn("DisplaySourceFactory", source)


if __name__ == "__main__":
    unittest.main()
