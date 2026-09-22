import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from PyQt5.QtCore import QSettings, Qt
from PyQt5.QtWidgets import QApplication, QMainWindow

from compute.capabilities import HardwareSnapshot
from compute.model import PrecisionPolicy
from compute.settings import ComputeSettingsStore
from history.dialog import HistoryCacheManagerDialog
from preferences.controller import PreferencesController
from preferences.dialog import PreferencesDialog


class _ResultDisplay:
    def __init__(self):
        self.calls = []

    def update_plot_settings(self, params, update=False):
        self.calls.append((dict(params), bool(update)))


class _HistoryController:
    def __init__(self):
        self.directory_changes = []

    def handle_cache_directory_change(self, old, new):
        self.directory_changes.append((old, new))


class _Canvas:
    def __init__(self):
        self.calls = []

    def set_toolset(self, params, update_display_style=True):
        self.calls.append((dict(params), bool(update_display_style)))


class _ImageDisplay:
    def __init__(self, tool_params):
        self.tool_parameters = dict(tool_params)
        self.display_canvas = [_Canvas()]


class _Window(QMainWindow):
    def __init__(self, settings, root):
        super().__init__()
        self.settings = settings
        self.current_version = "test"
        self.root = Path(root)
        self.tool_params = {
            "pen_size": 2,
            "pen_color": "#008000",
            "fill_color": "#006400",
            "vector_color": "#FFFF00",
            "vector_width": 2,
            "auto_fill": False,
            "anchor_select": False,
            "anchor_shape": "square",
            "anchor_size": 5,
            "anchor_method": "mean",
            "cache_directory": str(self.root / "cache"),
            "cache_threshold_mb": 512,
            "memory_budget_mb": 4096,
            "cache_cleanup_startup": True,
        }
        self.plot_params = {
            "line_style": "--",
            "line_width": 2,
            "marker_style": "s",
            "marker_size": 6,
            "color": "#1f77b4",
            "show_grid": False,
            "heatmap_cmap": "jet",
            "contour_levels": 10,
            "set_axis": True,
            "_from_start_cal": False,
        }
        self.cal_set_params = {
            "from_start_cal": False,
            "r_squared_min": 0.4,
            "peak_min": 0,
            "peak_max": 50,
            "tau_min": 1e-3,
            "tau_max": 1e3,
        }
        self.result_display = _ResultDisplay()
        self.history_controller = _HistoryController()
        self.image_display = _ImageDisplay(self.tool_params)
        self.theme_changes = []
        self.cache_applies = 0

    def get_log_path(self):
        return self.root / "lifecalor.log"

    def default_cache_directory(self):
        return str(self.root / "default-cache")

    def _save_param_group(self, group, params):
        self.settings.beginGroup(group)
        for key, value in params.items():
            self.settings.setValue(key, value)
        self.settings.endGroup()

    def _set_interface_theme(self, theme):
        self.theme_changes.append(theme)
        self.settings.setValue("appearance/theme", theme)
        return True

    def apply_cache_settings(self):
        self.cache_applies += 1

    def _sync_lifetime_compute_controls(self):
        pass

    def save_params(self):
        pass


def _snapshot():
    return HardwareSnapshot(
        captured_at=0,
        cpu_name="Test CPU",
        physical_cores=4,
        logical_cores=8,
        cpu_percent=10.0,
        ram_total_bytes=16 * 1024 ** 3,
        ram_available_bytes=8 * 1024 ** 3,
        devices=(),
    )


class PreferencesDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def make_window(self, directory):
        settings = QSettings(
            str(Path(directory) / "settings.ini"), QSettings.IniFormat
        )
        settings.setValue("should_check", True)
        return _Window(settings, directory)

    def test_cancel_keeps_draft_out_of_settings(self):
        with tempfile.TemporaryDirectory() as directory, patch(
            "compute.dialog.probe_hardware", return_value=_snapshot()
        ):
            window = self.make_window(directory)
            dialog = PreferencesDialog(window)
            self.addCleanup(dialog.close)
            self.assertTrue(dialog.pages["计算与加速"]._probe_thread.wait(3000))
            dialog.pages["常规与更新"].auto_update.setChecked(False)
            dialog.pages["外观与画布"].pen_size.setValue(9)
            dialog.reject()
            self.assertTrue(window.settings.value("should_check", type=bool))
            self.assertEqual(window.tool_params["pen_size"], 2)
            self.assertEqual(window.theme_changes, [])

    def test_apply_writes_only_changed_page(self):
        with tempfile.TemporaryDirectory() as directory, patch(
            "compute.dialog.probe_hardware", return_value=_snapshot()
        ):
            window = self.make_window(directory)
            dialog = PreferencesDialog(window)
            self.addCleanup(dialog.close)
            self.assertTrue(dialog.pages["计算与加速"]._probe_thread.wait(3000))
            dialog.pages["常规与更新"].auto_update.setChecked(False)
            self.assertTrue(dialog.apply_changes())
            self.assertFalse(window.settings.value("should_check", type=bool))
            self.assertEqual(window.cache_applies, 0)
            self.assertEqual(window.result_display.calls, [])

    def test_controller_reuses_one_non_modal_window_and_switches_page(self):
        with tempfile.TemporaryDirectory() as directory, patch(
            "compute.dialog.probe_hardware", return_value=_snapshot()
        ):
            window = self.make_window(directory)
            controller = PreferencesController(window)
            window.preferences_controller = controller
            first = controller.show(PreferencesController.PAGE_GENERAL)
            second = controller.show(PreferencesController.PAGE_CACHE)
            self.assertIs(first, second)
            self.assertEqual(
                second.navigation.currentItem().data(Qt.UserRole),
                PreferencesController.PAGE_CACHE,
            )
            self.assertFalse(second.isModal())
            self.assertTrue(second.pages["计算与加速"]._probe_thread.wait(3000))
            second.close()

    def test_algorithm_precision_override_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = QSettings(
                str(Path(directory) / "settings.ini"), QSettings.IniFormat
            )
            store = ComputeSettingsStore(settings)
            original = store.load()
            updated = type(original)(
                **{
                    **original.__dict__,
                    "algorithm_precisions": (
                        ("cwt", PrecisionPolicy.DOUBLE),
                    ),
                }
            )
            store.save(updated)
            loaded = store.load()
            self.assertEqual(
                loaded.precision_for("cwt"), PrecisionPolicy.DOUBLE
            )
            self.assertEqual(
                loaded.precision_for("stft"), loaded.precision
            )

    def test_history_cache_page_is_summary_with_preferences_link(self):
        dialog = HistoryCacheManagerDialog(
            params={"cache_directory": tempfile.gettempdir()},
            current_items=[],
            manifest_items=[],
            cache_summary={},
        )
        self.addCleanup(dialog.close)
        self.assertFalse(hasattr(dialog, "cache_directory_edit"))
        self.assertTrue(hasattr(dialog, "open_cache_preferences_btn"))

    def test_appearance_apply_updates_existing_canvas_tools_only(self):
        with tempfile.TemporaryDirectory() as directory, patch(
            "compute.dialog.probe_hardware", return_value=_snapshot()
        ):
            window = self.make_window(directory)
            dialog = PreferencesDialog(window)
            self.addCleanup(dialog.close)
            self.assertTrue(dialog.pages["计算与加速"]._probe_thread.wait(3000))
            page = dialog.pages["外观与画布"]
            page.pen_size.setValue(7)
            self.assertTrue(dialog.apply_changes())
            canvas = window.image_display.display_canvas[0]
            self.assertEqual(window.image_display.tool_parameters["pen_size"], 7)
            self.assertEqual(canvas.calls[-1][0]["pen_size"], 7)
            self.assertFalse(canvas.calls[-1][1])


if __name__ == "__main__":
    unittest.main()
