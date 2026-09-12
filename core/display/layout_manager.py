import logging

from PyQt5.QtCore import QObject, QTimer, Qt
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


class CanvasLayoutManager(QObject):
    """Arrange existing canvas docks without touching their image data."""

    AUTO = "auto"
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    QUAD = "quad"

    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.two_canvas_orientation = Qt.Horizontal
        self._arranging = False
        self._focus_state = None
        self._focus_visibility = {}
        self._focused_layout_key = None
        self._balance_generation = 0

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

    @property
    def is_focused(self):
        return self._focus_state is not None

    @property
    def focused_layout_key(self):
        return self._focused_layout_key

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

    def update_empty_state(self):
        self.empty_widget.setVisible(not bool(self.canvases()))

    def before_structure_change(self):
        if self.is_focused:
            self.restore_focus()

    def canvas_set_changed(self):
        self.update_empty_state()
        if self.canvases():
            self.arrange(self.AUTO)

    def arrange(self, mode=AUTO):
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
        self._schedule_balance()
        return True

    def _detach(self, docks):
        for dock in docks:
            dock.hide()
            if dock.isFloating():
                dock.setFloating(False)
            self.window.removeDockWidget(dock)

    def _schedule_balance(self):
        self._balance_generation += 1
        generation = self._balance_generation
        QTimer.singleShot(0, lambda: self._balance(generation))

    def _balance(self, generation):
        if generation != self._balance_generation or self._arranging or self.is_focused:
            return
        docks = self.canvases()
        count = len(docks)
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

    def focus_current(self):
        current = self.window.current_canvas()
        if current is None or current._is_closing or len(self.canvases()) <= 1:
            return False
        if self.is_focused:
            if self._focused_layout_key == current.layout_key:
                return True
            self.restore_focus()

        docks = self.canvases()
        self._focus_state = self.window.saveState()
        self._focus_visibility = {dock.layout_key: dock.isVisible() for dock in docks}
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
        self._focus_state = None
        self._focus_visibility = {}
        self._focused_layout_key = None

        for dock in self.canvases():
            dock.show()
        restored = self.window.restoreState(state)
        if not restored:
            logging.warning("画布聚焦前布局恢复失败，已回退到自动排列")
            self.arrange(self.AUTO)
            return False
        for dock in self.canvases():
            dock.setVisible(visibility.get(dock.layout_key, True))
        self.update_empty_state()
        return True

    def reset_layout(self):
        if self.is_focused:
            self.restore_focus()
        self.two_canvas_orientation = Qt.Horizontal
        return self.arrange(self.AUTO)
