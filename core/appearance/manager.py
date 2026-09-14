from __future__ import annotations

import logging
import sys
import weakref
from pathlib import Path
from string import Template

from PyQt5.QtCore import QFile, QIODevice, QObject, QSettings, pyqtSignal
from PyQt5.QtGui import QColor, QPalette
from PyQt5.QtWidgets import QApplication

from .tokens import THEME_DARK, THEME_LIGHT, normalize_theme_id, theme_tokens


SETTINGS_KEY = "appearance/theme"


def _resource_root() -> Path:
    frozen_root = getattr(sys, "_MEIPASS", None)
    return Path(frozen_root) if frozen_root else Path(__file__).resolve().parents[1]


def _read_resource(path: Path, resource_name: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        resource = QFile(resource_name)
        if not resource.open(QIODevice.ReadOnly | QIODevice.Text):
            raise
        try:
            return bytes(resource.readAll()).decode("utf-8")
        finally:
            resource.close()


def _palette(tokens):
    colors = tokens.colors
    palette = QPalette()
    roles = {
        QPalette.Window: colors["panel_bg"],
        QPalette.WindowText: colors["text"],
        QPalette.Base: colors["input_bg"],
        QPalette.AlternateBase: colors["raised_bg"],
        QPalette.ToolTipBase: colors["raised_bg"],
        QPalette.ToolTipText: colors["text"],
        QPalette.Text: colors["text"],
        QPalette.Button: colors["raised_bg"],
        QPalette.ButtonText: colors["text"],
        QPalette.Link: colors["info"],
        QPalette.Highlight: colors["selection"],
        QPalette.HighlightedText: colors["text"],
    }
    for role, value in roles.items():
        palette.setColor(QPalette.Active, role, QColor(value))
        palette.setColor(QPalette.Inactive, role, QColor(value))
    for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
        palette.setColor(QPalette.Disabled, role, QColor(colors["disabled"]))
    return palette


class ThemeManager(QObject):
    themeChanged = pyqtSignal(object)

    def __init__(self, app: QApplication, settings=None):
        super().__init__(app)
        self.app = app
        self.settings = settings or QSettings()
        self._light_palette = app.style().standardPalette()
        self.current_theme = THEME_LIGHT
        self.tokens = theme_tokens(THEME_LIGHT)
        self._adapters = weakref.WeakSet()

    def register_adapter(self, adapter):
        """Register a non-QSS surface such as a pyqtgraph canvas."""
        self._adapters.add(adapter)
        adapter.apply_theme(self.tokens)

    def unregister_adapter(self, adapter):
        self._adapters.discard(adapter)

    def _apply_adapters(self, tokens):
        for adapter in tuple(self._adapters):
            try:
                adapter.apply_theme(tokens)
            except RuntimeError:
                self._adapters.discard(adapter)
            except Exception:
                logging.error(
                    "界面主题适配器更新失败: %s",
                    type(adapter).__name__,
                    exc_info=True,
                )

    def preferred_theme(self) -> str:
        value = self.settings.value(SETTINGS_KEY, THEME_LIGHT)
        normalized = normalize_theme_id(value)
        if str(value or "").strip().lower() != normalized:
            logging.warning("未知界面主题 %r，已使用清爽浅色", value)
        return normalized

    def apply_preferred_theme(self) -> bool:
        return self.set_theme(self.preferred_theme(), persist=False, force=True)

    def set_theme(self, theme_id, persist=True, force=False) -> bool:
        target = normalize_theme_id(theme_id)
        if target == self.current_theme and not force:
            return True
        previous = (
            self.current_theme,
            self.tokens,
            self.app.palette(),
            self.app.styleSheet(),
        )
        try:
            tokens = theme_tokens(target)
            stylesheet = self._stylesheet(target, tokens)
            palette = self._light_palette if target == THEME_LIGHT else _palette(tokens)
            self.app.setPalette(palette)
            self.app.setStyleSheet(stylesheet)
            self.app.setProperty("appearanceTheme", target)
            self._apply_adapters(tokens)
        except Exception:
            logging.error("应用界面主题失败: %s", target, exc_info=True)
            old_id, old_tokens, old_palette, old_stylesheet = previous
            self.current_theme = old_id
            self.tokens = old_tokens
            self.app.setPalette(old_palette)
            self.app.setStyleSheet(old_stylesheet)
            self.app.setProperty("appearanceTheme", old_id)
            try:
                self._apply_adapters(old_tokens)
            except Exception:
                logging.error("恢复界面主题适配器失败", exc_info=True)
            return False
        self.current_theme = target
        self.tokens = tokens
        if persist:
            self.settings.setValue(SETTINGS_KEY, target)
            self.settings.sync()
        self.themeChanged.emit(tokens)
        return True

    @staticmethod
    def _stylesheet(theme_id, tokens):
        root = _resource_root()
        base = _read_resource(root / "style.qss", ":/appearance/style.qss")
        overlay_name = "dark.qss" if theme_id == THEME_DARK else "light.qss"
        overlay = _read_resource(
            root / "appearance" / overlay_name,
            f":/appearance/{overlay_name}",
        )
        return base + "\n" + Template(overlay).substitute(tokens.colors)


def get_theme_manager(app=None):
    app = app or QApplication.instance()
    return getattr(app, "_lifecalor_theme_manager", None) if app is not None else None


def install_theme_manager(app, settings=None):
    manager = get_theme_manager(app)
    if manager is None:
        manager = ThemeManager(app, settings=settings)
        app._lifecalor_theme_manager = manager
        manager.apply_preferred_theme()
    from .chrome import install_window_chrome_manager

    install_window_chrome_manager(app)
    return manager
