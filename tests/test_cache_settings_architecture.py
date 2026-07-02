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
        self.assertIn("cache_settings_edit_dialog", source)
        self.assertIn("CacheSettingsDialog", source)
        self.assertIn("cache_settings_edit.triggered.connect(self.cache_settings_edit_dialog)", source)
        self.assertIn("clear_array_cache", source)
        self.assertIn("cache_cleanup_startup", source)
        self.assertIn("cache_progress_update", source)
        self.assertIn("cache_progress_signal = pyqtSignal(object, object, str)", source)
        self.assertIn("self.cache_progress_signal.connect(self.cache_progress_update)", source)
        self.assertIn("set_array_cache_progress_callback(self.cache_progress_signal.emit)", source)
        self.assertNotIn("set_array_cache_progress_callback(self.cache_progress_update)", source)
        self.assertNotIn("QInputDialog", source)

    def test_cache_settings_dialog_follows_existing_dialog_pattern(self):
        source = (CORE / "ExtraDialog.py").read_text(encoding="utf-8")
        self.assertIn("class CacheSettingsDialog(QDialog)", source)
        self.assertIn("cache_directory_edit", source)
        self.assertIn("cache_threshold_spin", source)
        self.assertIn("browse_cache_directory", source)
        self.assertIn("clear_cache_btn", source)
        self.assertIn("clear_cache_requested", source)
        self.assertIn("QFileDialog.getExistingDirectory", source)
        self.assertIn("def apply_settings(self):", source)

    def test_mainwindow_loads_cached_history_on_worker_thread(self):
        source = (CORE / "MainWindow.py").read_text(encoding="utf-8")
        self.assertIn("cache_progress_signal = pyqtSignal(object, object, str)", source)
        self.assertIn("load_cached_history_async", source)
        self.assertIn("ArrayLoadWorker", source)
        self.assertIn("cache_load_thread", source)
        self.assertIn("normalize_progress", source)
        self.assertIn("self.load_cached_history_async(selected_data, 'data')", source)
        self.assertIn("self.load_cached_history_async(selected_data, 'processed_data')", source)
        self.assertNotIn("self.data = self.data.find_history(selected_timestamp)", source)
        self.assertNotIn("self.processed_data = self.processed_data.find_history(selected_timestamp)", source)

    def test_data_manager_accepts_progress_callback(self):
        source = (CORE / "DataManager.py").read_text(encoding="utf-8")
        self.assertIn("set_array_cache_progress_callback", source)
        self.assertIn("progress_callback", source)
        self.assertIn("clear_array_cache", source)
        self.assertIn("collect_array_refs", source)


if __name__ == "__main__":
    unittest.main()
