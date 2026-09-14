from __future__ import annotations

from PyQt5.QtCore import QEvent, QObject, QPoint, QRect, QSize, Qt
from PyQt5.QtWidgets import (
    QApplication,
    QColorDialog,
    QDialog,
    QFileDialog,
    QFontDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSizeGrip,
    QSplashScreen,
    QToolButton,
)

from .icons import window_control_icon


TITLE_BAR_HEIGHT = 34
_PROJECT_MODULE_PREFIXES = (
    "MainWindow",
    "ExtraDialog",
    "PlotGraphWidget",
    "ROIdrawDialog",
    "SpatialExtractor",
    "UpdateModule",
    "calculator.",
    "history.",
    "importing.",
    "widget.",
    "display.",
    "tasks.",
)
_EMBEDDED_WINDOW_MODULES = {"ImageDisplayWindow"}
_NATIVE_DIALOG_TYPES = (
    QColorDialog,
    QFileDialog,
    QFontDialog,
    QInputDialog,
    QMessageBox,
    QSplashScreen,
)


class WindowTitleBar(QFrame):
    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self._drag_origin = None
        self.setObjectName("AppTitleBar")
        self.setProperty("windowChrome", True)
        self.setFixedHeight(TITLE_BAR_HEIGHT)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 0, 4, 0)
        layout.setSpacing(6)

        self.icon_label = QLabel(self)
        self.icon_label.setObjectName("AppTitleIcon")
        self.icon_label.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.icon_label.setFixedSize(20, 20)
        layout.addWidget(self.icon_label)

        self.title_label = QLabel(self)
        self.title_label.setObjectName("AppTitleText")
        self.title_label.setAttribute(Qt.WA_TransparentForMouseEvents)
        layout.addWidget(self.title_label, 1)

        self.theme_button = None
        if isinstance(window, QMainWindow):
            self.theme_button = self._button("theme", "切换界面主题")
            self.theme_button.clicked.connect(self.toggle_theme)
            layout.addWidget(self.theme_button)

        self.minimize_button = self._button(
            "minimize", "最小化"
        )
        self.maximize_button = self._button(
            "maximize", "最大化"
        )
        self.close_button = self._button(
            "close", "关闭"
        )
        layout.addWidget(self.minimize_button)
        layout.addWidget(self.maximize_button)
        layout.addWidget(self.close_button)

        self.minimize_button.clicked.connect(window.showMinimized)
        self.maximize_button.clicked.connect(self.toggle_maximized)
        self.close_button.clicked.connect(window.close)
        self.sync_from_window()

    def _button(self, role, tooltip):
        button = QToolButton(self)
        button.setProperty("titleBarButton", True)
        button.setProperty("titleBarRole", role)
        button.setAutoRaise(True)
        button.setFixedSize(40, 32)
        button.setIconSize(QSize(18, 18))
        button.setToolTip(tooltip)
        return button

    def apply_theme(self, tokens):
        if self.theme_button is not None:
            next_is_dark = tokens.theme_id == "light"
            role = "moon" if next_is_dark else "sun"
            self.theme_button.setIcon(window_control_icon(role, tokens))
            self.theme_button.setToolTip(
                "切换到石墨深色" if next_is_dark else "切换到清爽浅色"
            )
        self.minimize_button.setIcon(window_control_icon("minimize", tokens))
        self.close_button.setIcon(window_control_icon("close", tokens))
        self._sync_maximize_icon(tokens)

    def toggle_theme(self):
        from .manager import get_theme_manager

        manager = get_theme_manager()
        if manager is None:
            return
        target = "dark" if manager.current_theme == "light" else "light"
        manager.set_theme(target)

    def sync_from_window(self):
        self.title_label.setText(self.window.windowTitle())
        icon = self.window.windowIcon()
        if icon.isNull():
            app = QApplication.instance()
            icon = app.windowIcon() if app is not None else icon
        self.icon_label.setPixmap(icon.pixmap(18, 18))
        maximized = self.window.isMaximized()
        resizable = self.window.minimumSize() != self.window.maximumSize()
        self.maximize_button.setVisible(resizable)
        self._sync_maximize_icon()
        self.maximize_button.setToolTip("还原" if maximized else "最大化")

    def _sync_maximize_icon(self, tokens=None):
        if tokens is None:
            from .manager import get_theme_manager
            from .tokens import theme_tokens

            manager = get_theme_manager()
            tokens = manager.tokens if manager is not None else theme_tokens("light")
        role = "restore" if self.window.isMaximized() else "maximize"
        self.maximize_button.setIcon(window_control_icon(role, tokens))

    def toggle_maximized(self):
        if self.window.isMaximized():
            self.window.showNormal()
        else:
            self.window.showMaximized()

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.toggle_maximized()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            handle = self.window.windowHandle()
            if handle is not None and hasattr(handle, "startSystemMove"):
                if handle.startSystemMove():
                    event.accept()
                    return
            self._drag_origin = event.globalPos() - self.window.frameGeometry().topLeft()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_origin is not None and event.buttons() & Qt.LeftButton:
            if self.window.isMaximized():
                self.window.showNormal()
            self.window.move(event.globalPos() - self._drag_origin)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._drag_origin = None
        super().mouseReleaseEvent(event)


class WindowChrome(QObject):
    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.original_margins = window.contentsMargins()
        window.setWindowFlags(window.windowFlags() | Qt.FramelessWindowHint)
        window.setContentsMargins(
            self.original_margins.left(),
            self.original_margins.top() + TITLE_BAR_HEIGHT,
            self.original_margins.right(),
            self.original_margins.bottom(),
        )
        self.title_bar = WindowTitleBar(window)
        self.size_grip = QSizeGrip(window)
        self.size_grip.setObjectName("AppWindowSizeGrip")
        window.installEventFilter(self)
        self.update_geometry()
        self.title_bar.show()
        self.size_grip.show()
        from .manager import get_theme_manager
        from .tokens import theme_tokens

        manager = get_theme_manager()
        if manager is not None:
            manager.register_adapter(self)
        else:
            self.apply_theme(theme_tokens("light"))

    def apply_theme(self, tokens):
        self.title_bar.apply_theme(tokens)

    def update_geometry(self):
        width = self.window.width()
        self.title_bar.setGeometry(0, 0, width, TITLE_BAR_HEIGHT)
        grip_size = self.size_grip.sizeHint()
        self.size_grip.setGeometry(
            QRect(
                max(0, width - grip_size.width()),
                max(0, self.window.height() - grip_size.height()),
                grip_size.width(),
                grip_size.height(),
            )
        )
        resizable = (
            not self.window.isMaximized()
            and self.window.minimumSize() != self.window.maximumSize()
        )
        self.size_grip.setVisible(resizable)
        self.title_bar.raise_()
        if resizable:
            self.size_grip.raise_()

    def eventFilter(self, watched, event):
        if watched is self.window:
            if event.type() in (QEvent.Resize, QEvent.Show, QEvent.WindowStateChange):
                self.update_geometry()
                self.title_bar.sync_from_window()
            elif event.type() in (QEvent.WindowTitleChange, QEvent.WindowIconChange):
                self.title_bar.sync_from_window()
        return False


class WindowChromeManager(QObject):
    def __init__(self, app):
        super().__init__(app)
        self.app = app
        app.installEventFilter(self)

    @staticmethod
    def _is_project_window(window):
        module = type(window).__module__
        if module in _EMBEDDED_WINDOW_MODULES:
            return False
        return module.startswith(_PROJECT_MODULE_PREFIXES)

    def decorate(self, window):
        if getattr(window, "_lifecalor_window_chrome", None) is not None:
            return window._lifecalor_window_chrome
        if window.property("useNativeChrome"):
            return None
        if isinstance(window, _NATIVE_DIALOG_TYPES):
            return None
        if not isinstance(window, (QMainWindow, QDialog)) or not window.isWindow():
            return None
        if not self._is_project_window(window):
            return None
        chrome = WindowChrome(window)
        window._lifecalor_window_chrome = chrome
        return chrome

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Polish:
            self.decorate(watched)
        return False


def install_window_chrome_manager(app=None):
    app = app or QApplication.instance()
    if app is None:
        return None
    manager = getattr(app, "_lifecalor_window_chrome_manager", None)
    if manager is None:
        manager = WindowChromeManager(app)
        app._lifecalor_window_chrome_manager = manager
    return manager
