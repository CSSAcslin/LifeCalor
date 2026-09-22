from __future__ import annotations

from .dialog import PreferencesDialog


class PreferencesController:
    PAGE_GENERAL = "常规与更新"
    PAGE_APPEARANCE = "外观与画布"
    PAGE_PLOT = "绘图默认值"
    PAGE_COMPUTE = "计算与加速"
    PAGE_LIFETIME = "寿命拟合默认值"
    PAGE_CACHE = "缓存与存储"

    def __init__(self, window):
        self.window = window
        self.dialog = None

    def show(self, page=None):
        if self.dialog is None:
            self.dialog = PreferencesDialog(self.window, self.window)
            self.dialog.destroyed.connect(self._dialog_destroyed)
            self.dialog.settingsApplied.connect(self.window.save_params)
        if page is None:
            index = self.window.settings.value(
                "preferences/last_page", 0, type=int
            )
            titles = list(self.dialog.pages)
            page = titles[max(0, min(index, len(titles) - 1))]
        self.dialog.show_page(page)
        return self.dialog

    def _dialog_destroyed(self):
        self.dialog = None
