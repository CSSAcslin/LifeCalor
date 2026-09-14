import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from PyQt5.QtCore import QSettings, QSize
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QDialog,
    QMainWindow,
    QTableWidget,
    QTableWidgetItem,
    QToolBar,
    QToolButton,
)

import resources_rc
from appearance.manager import SETTINGS_KEY, ThemeManager
from appearance.chrome import WindowChromeManager
from appearance.tokens import theme_tokens
from startup.splash import StartupSplash


def _luminance(color):
    values = [int(color[index:index + 2], 16) / 255.0 for index in (1, 3, 5)]
    values = [
        value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4
        for value in values
    ]
    return 0.2126 * values[0] + 0.7152 * values[1] + 0.0722 * values[2]


def _contrast(first, second):
    high, low = sorted((_luminance(first), _luminance(second)), reverse=True)
    return (high + 0.05) / (low + 0.05)


class AppearanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.settings = QSettings(
            str(Path(self.tempdir.name) / "theme.ini"),
            QSettings.IniFormat,
        )
        self.manager = ThemeManager(self.app, settings=self.settings)

    def tearDown(self):
        if getattr(self.app, "_lifecalor_theme_manager", None) is self.manager:
            del self.app._lifecalor_theme_manager
        self.app.setStyleSheet("")
        self.tempdir.cleanup()

    def test_light_theme_is_the_existing_white_green_stylesheet(self):
        self.assertTrue(self.manager.set_theme("light", persist=False, force=True))
        expected = (CORE / "style.qss").read_text(encoding="utf-8")
        self.assertTrue(self.app.styleSheet().startswith(expected))
        self.assertIn("Original white-green theme", self.app.styleSheet())
        self.assertEqual(
            self.app.palette().color(self.app.palette().Window),
            self.app.style().standardPalette().color(self.app.palette().Window),
        )

    def test_dark_theme_is_color_only_overlay_and_persists(self):
        self.assertTrue(self.manager.set_theme("dark"))
        self.assertEqual(self.settings.value(SETTINGS_KEY), "dark")
        self.assertIn("Color-only overlay", self.app.styleSheet())
        self.assertEqual(self.manager.current_theme, "dark")

    def test_unknown_preference_falls_back_to_light(self):
        self.settings.setValue(SETTINGS_KEY, "not-a-theme")
        self.assertTrue(self.manager.apply_preferred_theme())
        self.assertEqual(self.manager.current_theme, "light")

    def test_toolbar_geometry_does_not_change_between_themes(self):
        window = QMainWindow()
        toolbar = QToolBar(window)
        toolbar.setIconSize(QSize(36, 36))
        window.addToolBar(toolbar)
        for index in range(14):
            toolbar.addAction(QIcon(), f"tool-{index}")
        window.resize(1200, 400)
        window.show()

        self.manager.set_theme("light", persist=False, force=True)
        self.app.processEvents()
        light_bar = toolbar.sizeHint()
        light_buttons = [
            button.sizeHint()
            for button in toolbar.findChildren(QToolButton)
        ]

        self.manager.set_theme("dark", persist=False, force=True)
        self.app.processEvents()
        dark_bar = toolbar.sizeHint()
        dark_buttons = [
            button.sizeHint()
            for button in toolbar.findChildren(QToolButton)
        ]

        self.assertEqual(light_bar, dark_bar)
        self.assertEqual(light_buttons, dark_buttons)
        window.close()

    def test_splash_uses_dark_colors_when_dark_was_selected(self):
        self.manager.set_theme("dark", persist=False, force=True)
        self.app._lifecalor_theme_manager = self.manager
        splash = StartupSplash()
        self.assertIn(theme_tokens("dark")["panel_bg"], splash.styleSheet())
        splash.close()

    def test_key_text_contrast_meets_desktop_target(self):
        for theme_id in ("light", "dark"):
            tokens = theme_tokens(theme_id)
            self.assertGreaterEqual(_contrast(tokens["text"], tokens["panel_bg"]), 4.5)
            self.assertGreaterEqual(
                _contrast(tokens["secondary"], tokens["panel_bg"]),
                4.5,
            )
            self.assertGreaterEqual(
                _contrast(tokens["on_accent"], tokens["accent"]),
                4.5,
            )

    def test_launcher_applies_theme_before_splash(self):
        source = (CORE / "launcher.py").read_text(encoding="utf-8")
        source = source[source.index("def main()"):]
        self.assertLess(
            source.index("_configure_appearance(app)"),
            source.index("splash = StartupSplash(icon_path)"),
        )

    def test_custom_window_chrome_tracks_title_and_window_state(self):
        class ProjectDialog(QDialog):
            pass

        ProjectDialog.__module__ = "ExtraDialog"
        window = ProjectDialog()
        window.setWindowTitle("测试窗口")
        chrome = WindowChromeManager(self.app).decorate(window)

        self.assertIsNotNone(chrome)
        self.assertTrue(window.windowFlags() & Qt.FramelessWindowHint)
        self.assertEqual(chrome.title_bar.title_label.text(), "测试窗口")
        self.assertEqual(chrome.title_bar.close_button.toolTip(), "关闭")
        self.assertFalse(chrome.title_bar.close_button.icon().isNull())
        self.assertEqual(chrome.title_bar.close_button.iconSize(), QSize(18, 18))

        window.setWindowTitle("新标题")
        self.app.processEvents()
        self.assertEqual(chrome.title_bar.title_label.text(), "新标题")
        window.close()

    def test_embedded_canvas_window_is_not_decorated(self):
        EmbeddedCanvas = type(
            "ImageDisplayWindow",
            (QMainWindow,),
            {"__module__": "ImageDisplayWindow"},
        )
        canvas = EmbeddedCanvas()
        manager = WindowChromeManager(self.app)
        self.assertIsNone(manager.decorate(canvas))
        self.assertFalse(canvas.windowFlags() & Qt.FramelessWindowHint)
        canvas.close()

    def test_main_titlebar_theme_button_toggles_and_updates_hint(self):
        ProjectMainWindow = type(
            "MainWindow",
            (QMainWindow,),
            {"__module__": "MainWindow"},
        )
        self.app._lifecalor_theme_manager = self.manager
        self.manager.set_theme("light", persist=False, force=True)
        window = ProjectMainWindow()
        chrome = WindowChromeManager(self.app).decorate(window)
        button = chrome.title_bar.theme_button

        self.assertIsNotNone(button)
        self.assertFalse(button.icon().isNull())
        self.assertEqual(button.toolTip(), "切换到石墨深色")
        button.click()
        self.assertEqual(self.manager.current_theme, "dark")
        self.assertEqual(button.toolTip(), "切换到清爽浅色")
        window.close()

    def test_dark_table_header_uses_dark_surface(self):
        self.manager.set_theme("dark", persist=False, force=True)
        table = QTableWidget(1, 2)
        table.setHorizontalHeaderLabels(["名称", "类型"])
        table.setItem(0, 0, QTableWidgetItem("示例"))
        table.resize(360, 140)
        table.show()
        self.app.processEvents()

        header = table.horizontalHeader()
        image = header.grab().toImage()
        sample = image.pixelColor(80, max(1, header.height() - 5)).name().upper()
        self.assertEqual(sample, theme_tokens("dark")["raised_bg"])
        table.close()

    def test_plot_theme_adapter_avoids_global_configuration_or_replot(self):
        source = (CORE / "PlotGraphWidget.py").read_text(encoding="utf-8")
        self.assertNotIn("setConfigOption('background'", source)
        apply_theme = source[source.index("    def apply_theme"):]
        apply_theme = apply_theme[:apply_theme.index("    def _apply_legend_theme")]
        self.assertNotIn("plot_data(", apply_theme)
        self.assertNotIn("setData(", apply_theme)
        self.assertNotIn("autoRange(", apply_theme)


if __name__ == "__main__":
    unittest.main()
