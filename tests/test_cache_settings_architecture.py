import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"


class CacheSettingsArchitectureTests(unittest.TestCase):
    def test_mainwindow_exposes_cache_settings(self):
        source = (CORE / "MainWindow.py").read_text(encoding="utf-8")
        self.assertIn("cache_threshold_mb", source)
        self.assertIn("cache_directory", source)
        self.assertIn("configure_array_cache", source)
        self.assertIn("choose_cache_directory", source)
        self.assertIn("cache_progress_update", source)

    def test_data_manager_accepts_progress_callback(self):
        source = (CORE / "DataManager.py").read_text(encoding="utf-8")
        self.assertIn("set_array_cache_progress_callback", source)
        self.assertIn("progress_callback", source)


if __name__ == "__main__":
    unittest.main()
