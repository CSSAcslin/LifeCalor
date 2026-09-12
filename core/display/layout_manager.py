import json
import logging

from PyQt5.QtCore import QByteArray, QEvent, QObject, QTimer, Qt
from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


class CanvasLayoutManager(QObject):
    """Arrange and persist canvas docks without touching image data."""

    AUTO = "auto"
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    QUAD = "quad"
    CUSTOM = "custom"

    SETTINGS_ROOT = "canvas_workspace"
    SCHEMA_VERSION = 1
    STATE_VERSION = 1
    SAVE_DELAY_MS = 300

    def __init__(self, window, settings=None):
        super().__init__(window)
        self.window = window
        self.settings = settings
        self.two_canvas_orientation = Qt.Horizontal
        self._layout_mode = self.AUTO
        self._arranging = False
        self._restoring = False
        self._focus_state = None
        self._focus_visibility = {}
        self._focus_floating_geometries = {}
        self._focus_layout_mode = None
        self._focused_layout_key = None
        self._balance_generation = 0
        self._restore_attempted_signatures = set()
        self._warned_restore_failures = set()

        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(self.SAVE_DELAY_MS)
        self._save_timer.timeout.connect(self.save_layout)

        options = (
            QMainWindow.AnimatedDocks
            | QMainWindow.AllowNestedDocks
            | QMainWindow.AllowTabbedDocks
        )
        window.setDockOptions(options)
        window.setDockNestingEnabled(True)
        self.empty_widget = self._build_empty_widget()
        window.setCentralWidget(self.empty_widget)
        self.update_empty_state()

        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self.flush_save)

    @property
    def is_focused(self):
        return self._focus_state is not None

    @property
    def focused_layout_key(self):
        return self._focused_layout_key

    @property
    def layout_mode(self):
        return self._layout_mode

    def _build_empty_widget(self):
        widget = QWidget(self.window)
        widget.setObjectName("CanvasEmptyState")
        layout = QVBoxLayout(widget)
        layout.addStretch()
        label = QLabel("暂无画布")
        label.setAlignment(Qt.AlignCenter)
        button = QPushButton("添加数据")
        button.setObjectName("CanvasEmptyAddButton")
        button.setToolTip("从当前数据或历史数据中选择一项并创建画布")
        button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        button.clicked.connect(self.window.add_canvas_signal.emit)
        row = QHBoxLayout()
        row.addStretch()
        row.addWidget(button)
        row.addStretch()
        layout.addWidget(label)
        layout.addLayout(row)
        layout.addStretch()
        return widget

    def canvases(self):
        return [canvas for canvas in self.window.display_canvas if not canvas._is_closing]

    def layout_signature(self, docks=None):
        docks = self.canvases() if docks is None else docks
        return "+".join(sorted(dock.layout_key for dock in docks))

    def _layout_prefix(self, signature):
        return f"{self.SETTINGS_ROOT}/layouts/{signature}"

    def _settings_ready(self):
        return self.settings is not None and hasattr(self.settings, "value")

    def update_empty_state(self):
        self.empty_widget.setVisible(not bool(self.canvases()))

    def register_canvas(self, canvas):
        if getattr(canvas, "_layout_persistence_bound", False):
            return
        canvas._layout_persistence_bound = True
        canvas.installEventFilter(self)
        canvas.dockLocationChanged.connect(self._on_dock_layout_changed)
        canvas.topLevelChanged.connect(self._on_dock_layout_changed)
        canvas.visibilityChanged.connect(self._on_dock_layout_changed)

    def eventFilter(self, watched, event):
        if (
            getattr(watched, "_layout_persistence_bound", False)
            and event.type() in {QEvent.Move, QEvent.Resize, QEvent.Show, QEvent.Hide}
        ):
            self._on_dock_layout_changed()
        return super().eventFilter(watched, event)

    def _on_dock_layout_changed(self, *args):
        if self._arranging or self._restoring or self.is_focused:
            return
        self._layout_mode = self.CUSTOM
        self.schedule_save()

    def before_structure_change(self):
        if self.is_focused:
            self.restore_focus()

    def canvas_set_changed(self):
        docks = self.canvases()
        for dock in docks:
            self.register_canvas(dock)
        self.update_empty_state()
        if not docks:
            self._save_timer.stop()
            return

        signature = self.layout_signature(docks)
        if signature not in self._restore_attempted_signatures:
            self._restore_attempted_signatures.add(signature)
            if self.restore_saved_layout(signature):
                return
        self.arrange(self.AUTO)

    def arrange(self, mode=AUTO, persist=True):
        docks = self.canvases()
        if not docks:
            self.update_empty_state()
            return False
        if self.is_focused:
            self.restore_focus()

        count = len(docks)
        if mode == self.HORIZONTAL and count != 2:
            return False
        if mode == self.VERTICAL and count != 2:
            return False
        if mode == self.QUAD and count != 4:
            return False
        if mode == self.HORIZONTAL:
            self.two_canvas_orientation = Qt.Horizontal
        elif mode == self.VERTICAL:
            self.two_canvas_orientation = Qt.Vertical
        self._layout_mode = mode

        self._arranging = True
        try:
            self._detach(docks)
            first = docks[0]
            self.window.addDockWidget(Qt.LeftDockWidgetArea, first)
            first.show()

            if count == 2:
                self.window.splitDockWidget(first, docks[1], self.two_canvas_orientation)
            elif count == 3:
                self.window.splitDockWidget(first, docks[1], Qt.Horizontal)
                self.window.splitDockWidget(docks[1], docks[2], Qt.Vertical)
            elif count >= 4:
                self.window.splitDockWidget(first, docks[1], Qt.Horizontal)
                self.window.splitDockWidget(first, docks[2], Qt.Vertical)
                self.window.splitDockWidget(docks[1], docks[3], Qt.Vertical)
            for dock in docks[1:]:
                dock.show()
        finally:
            self._arranging = False

        self.update_empty_state()
        self._schedule_balance(persist)
        return True

    def _detach(self, docks):
        for dock in docks:
            dock.hide()
            if dock.isFloating():
                dock.setFloating(False)
            self.window.removeDockWidget(dock)

    def _schedule_balance(self, persist=True):
        self._balance_generation += 1
        generation = self._balance_generation
        QTimer.singleShot(0, lambda: self._balance(generation, persist))

    def _balance(self, generation, persist):
        if generation != self._balance_generation or self._arranging or self.is_focused:
            return
        docks = self.canvases()
        count = len(docks)
        self._arranging = True
        try:
            if count == 2:
                extent = self.window.width() if self.two_canvas_orientation == Qt.Horizontal else self.window.height()
                self.window.resizeDocks(docks, [max(1, extent // 2)] * 2, self.two_canvas_orientation)
            elif count == 3:
                self.window.resizeDocks(docks[:2], [1, 1], Qt.Horizontal)
                self.window.resizeDocks(docks[1:3], [1, 1], Qt.Vertical)
            elif count >= 4:
                self.window.resizeDocks(docks[:2], [1, 1], Qt.Horizontal)
                self.window.resizeDocks([docks[0], docks[2]], [1, 1], Qt.Vertical)
                self.window.resizeDocks([docks[1], docks[3]], [1, 1], Qt.Vertical)
        finally:
            self._arranging = False
        if persist:
            self.schedule_save()
        else:
            self._save_timer.stop()

    def schedule_save(self):
        if (
            not self._settings_ready()
            or self._arranging
            or self._restoring
            or self.is_focused
            or not self.canvases()
        ):
            return False
        self._save_timer.start()
        return True

    def _visibility(self, docks):
        return {dock.layout_key: dock.isVisible() for dock in docks}

    def _floating_geometries(self, docks):
        geometries = {}
        for dock in docks:
            if dock.isFloating():
                geometry = dock.frameGeometry()
                geometries[dock.layout_key] = [
                    geometry.x(), geometry.y(), geometry.width(), geometry.height()
                ]
        return geometries

    def _write_saved_layout(self, state, visibility, floating_geometries, mode):
        docks = self.canvases()
        if not self._settings_ready() or not docks or not state:
            return False
        signature = self.layout_signature(docks)
        prefix = self._layout_prefix(signature)
        self.settings.setValue(f"{self.SETTINGS_ROOT}/schema_version", self.SCHEMA_VERSION)
        self.settings.setValue(f"{prefix}/layout_keys", json.dumps(sorted(visibility)))
        self.settings.setValue(f"{prefix}/state", state)
        self.settings.setValue(f"{prefix}/mode", mode or self.AUTO)
        self.settings.setValue(f"{prefix}/orientation", int(self.two_canvas_orientation))
        self.settings.setValue(f"{prefix}/visibility", json.dumps(visibility))
        self.settings.setValue(f"{prefix}/floating_geometries", json.dumps(floating_geometries))
        self.settings.sync()
        return True

    def save_layout(self):
        if self.is_focused:
            return False
        docks = self.canvases()
        if not docks:
            return False
        return self._write_saved_layout(
            self.window.saveState(self.STATE_VERSION),
            self._visibility(docks),
            self._floating_geometries(docks),
            self._layout_mode,
        )

    def flush_save(self):
        self._save_timer.stop()
        if self.is_focused:
            return self._write_saved_layout(
                self._focus_state,
                self._focus_visibility,
                self._focus_floating_geometries,
                self._focus_layout_mode,
            )
        return self.save_layout()

    @staticmethod
    def _json_mapping(value, default=None):
        if value in (None, ""):
            return {} if default is None else default
        try:
            result = json.loads(str(value))
        except (TypeError, ValueError, json.JSONDecodeError):
            return {} if default is None else default
        return result if isinstance(result, dict) else ({} if default is None else default)

    @staticmethod
    def _state_bytes(value):
        if isinstance(value, QByteArray):
            return value
        if isinstance(value, (bytes, bytearray)):
            return QByteArray(bytes(value))
        return QByteArray()

    def _warn_restore_failure(self, signature, message):
        if signature in self._warned_restore_failures:
            return
        self._warned_restore_failures.add(signature)
        logging.warning("画布布局恢复失败，已回退到默认排列: %s", message)

    def restore_saved_layout(self, signature=None):
        docks = self.canvases()
        if not self._settings_ready() or not docks:
            return False
        signature = signature or self.layout_signature(docks)
        try:
            schema = int(self.settings.value(f"{self.SETTINGS_ROOT}/schema_version", 0))
        except (TypeError, ValueError):
            schema = 0
        prefix = self._layout_prefix(signature)
        state_value = self.settings.value(f"{prefix}/state")
        if state_value is None or (isinstance(state_value, str) and not state_value):
            return False
        if schema != self.SCHEMA_VERSION:
            self._warn_restore_failure(signature, f"配置版本 {schema} 不受支持")
            return False

        state = self._state_bytes(state_value)
        if state.isEmpty():
            self._warn_restore_failure(signature, "布局状态为空或格式无效")
            return False

        self._restoring = True
        restored = False
        try:
            for dock in docks:
                if self.window.dockWidgetArea(dock) == Qt.NoDockWidgetArea and not dock.isFloating():
                    self.window.addDockWidget(Qt.LeftDockWidgetArea, dock)
                dock.show()
            restored = self.window.restoreState(state, self.STATE_VERSION)
            if restored:
                visibility = self._json_mapping(self.settings.value(f"{prefix}/visibility"))
                if visibility and not any(bool(value) for value in visibility.values()):
                    visibility = {dock.layout_key: True for dock in docks}
                for dock in docks:
                    dock.setVisible(bool(visibility.get(dock.layout_key, True)))
                try:
                    orientation = int(self.settings.value(f"{prefix}/orientation", int(Qt.Horizontal)))
                except (TypeError, ValueError):
                    orientation = int(Qt.Horizontal)
                self.two_canvas_orientation = Qt.Vertical if orientation == int(Qt.Vertical) else Qt.Horizontal
                self._layout_mode = str(self.settings.value(f"{prefix}/mode", self.CUSTOM))
                floating = self._json_mapping(self.settings.value(f"{prefix}/floating_geometries"))
                self._apply_floating_geometries(floating)
        except Exception as exc:
            self._warn_restore_failure(signature, str(exc))
            restored = False
        finally:
            self._restoring = False

        if not restored:
            self._warn_restore_failure(signature, "Qt 无法还原布局状态")
            return False
        self.update_empty_state()
        QTimer.singleShot(0, self._clamp_floating_docks)
        return True

    def _apply_floating_geometries(self, geometries):
        for dock in self.canvases():
            geometry = geometries.get(dock.layout_key)
            if dock.isFloating() and isinstance(geometry, list) and len(geometry) == 4:
                try:
                    dock.setGeometry(*(int(value) for value in geometry))
                except (TypeError, ValueError):
                    continue

    def _clamp_floating_docks(self):
        screens = [screen.availableGeometry() for screen in QApplication.screens()]
        if not screens:
            return
        primary = QApplication.primaryScreen()
        target = primary.availableGeometry() if primary is not None else screens[0]
        for dock in self.canvases():
            if not dock.isFloating():
                continue
            frame = dock.frameGeometry()
            if any(screen.intersects(frame) for screen in screens):
                continue
            width = min(max(dock.minimumWidth(), frame.width()), target.width())
            height = min(max(dock.minimumHeight(), frame.height()), target.height())
            dock.resize(width, height)
            dock.move(target.left(), target.top())

    def focus_current(self):
        current = self.window.current_canvas()
        if current is None or current._is_closing or len(self.canvases()) <= 1:
            return False
        if self.is_focused:
            if self._focused_layout_key == current.layout_key:
                return True
            self.restore_focus()

        self._save_timer.stop()
        docks = self.canvases()
        self._focus_state = self.window.saveState(self.STATE_VERSION)
        self._focus_visibility = self._visibility(docks)
        self._focus_floating_geometries = self._floating_geometries(docks)
        self._focus_layout_mode = self._layout_mode
        self._focused_layout_key = current.layout_key
        for dock in docks:
            dock.setVisible(dock is current)
        if current.isFloating():
            current.setFloating(False)
        current.show()
        current.raise_()
        self.update_empty_state()
        return True

    def restore_focus(self):
        if not self.is_focused:
            return False
        state = self._focus_state
        visibility = self._focus_visibility
        floating = self._focus_floating_geometries
        layout_mode = self._focus_layout_mode
        self._focus_state = None
        self._focus_visibility = {}
        self._focus_floating_geometries = {}
        self._focus_layout_mode = None
        self._focused_layout_key = None

        restored = False
        self._restoring = True
        try:
            for dock in self.canvases():
                dock.show()
            restored = self.window.restoreState(state, self.STATE_VERSION)
            if restored:
                for dock in self.canvases():
                    dock.setVisible(visibility.get(dock.layout_key, True))
                self._apply_floating_geometries(floating)
                self._layout_mode = layout_mode or self.CUSTOM
        finally:
            self._restoring = False
        if not restored:
            logging.warning("画布聚焦前布局恢复失败，已回退到自动排列")
            self.arrange(self.AUTO)
            return False
        self.update_empty_state()
        QTimer.singleShot(0, self._clamp_floating_docks)
        self.schedule_save()
        return True

    def clear_saved_layouts(self):
        if not self._settings_ready():
            return False
        self.settings.remove(self.SETTINGS_ROOT)
        self.settings.sync()
        self._restore_attempted_signatures.clear()
        self._warned_restore_failures.clear()
        return True

    def reset_layout(self):
        if self.is_focused:
            self.restore_focus()
        self.clear_saved_layouts()
        self.two_canvas_orientation = Qt.Horizontal
        restored = self.arrange(self.AUTO, persist=False)
        self._save_timer.stop()
        return restored