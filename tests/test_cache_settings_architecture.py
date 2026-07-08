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
        main_source = (CORE / "MainWindow.py").read_text(encoding="utf-8")
        history_source = (CORE / "history" / "controller.py").read_text(encoding="utf-8")
        self.assertIn("cache_progress_signal = pyqtSignal(object, object, str)", main_source)
        self.assertIn("self.history_controller.load_cached_history_async(target, attr_name)", main_source)
        self.assertIn("ArrayLoadWorker", history_source)
        self.assertIn("cache_load_thread", history_source)
        self.assertIn("from progress import normalize_progress", main_source)
        self.assertIn("self.load_cached_history_async(selected_data, 'data')", history_source)
        self.assertIn("self.load_cached_history_async(selected_data, 'processed_data')", history_source)
        self.assertNotIn("self.data = self.data.find_history(selected_timestamp)", main_source)
        self.assertNotIn("self.processed_data = self.processed_data.find_history(selected_timestamp)", main_source)

    def test_data_manager_accepts_progress_callback(self):
        source = (CORE / "DataManager.py").read_text(encoding="utf-8")
        self.assertIn("set_array_cache_progress_callback", source)
        self.assertIn("progress_callback", source)
        self.assertIn("clear_array_cache", source)
        self.assertIn("collect_array_refs", source)

    def test_cache_restore_can_be_cancelled(self):
        history_source = (CORE / "history" / "controller.py").read_text(encoding="utf-8")
        data_source = (CORE / "DataManager.py").read_text(encoding="utf-8")
        dialog_source = (CORE / "history" / "dialog.py").read_text(encoding="utf-8")
        self.assertIn("cancel_cached_history_load", history_source)
        self.assertIn("cancelled_signal", data_source)
        self.assertIn("CacheLoadCancelled", data_source)
        self.assertIn("cancel_load_requested", dialog_source)

    def test_cache_directory_switch_is_explicitly_handled(self):
        history_source = (CORE / "history" / "controller.py").read_text(encoding="utf-8")
        dialog_source = (CORE / "history" / "dialog.py").read_text(encoding="utf-8")
        self.assertIn("handle_cache_directory_change", history_source)
        self.assertIn("旧缓存", history_source)
        self.assertIn("当前目录", dialog_source)


if __name__ == "__main__":
    unittest.main()
