from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import QEventLoop, Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)

from app_metadata import APP_NAME, APP_VERSION


class StartupSplash(QWidget):
    """Compact, DPI-aware startup feedback that cannot be dismissed by a click."""

    def __init__(self, icon_path=None, parent=None, total_steps=8):
        super().__init__(
            parent,
            Qt.SplashScreen | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint,
        )
        self.setObjectName("LifeCalorStartupSplash")
        self.setAttribute(Qt.WA_DeleteOnClose, False)
        self.setFixedSize(480, 238)
        self._total_steps = max(1, int(total_steps))
        self._build_ui(Path(icon_path) if icon_path else None)

    def _build_ui(self, icon_path):
        self.setStyleSheet(
            """
            QWidget#LifeCalorStartupSplash {
                background: white;
                border: 1px solid #8BCB8B;
                border-radius: 6px;
            }
            QLabel#StartupName { color: #1F4D2E; font-size: 20pt; font-weight: 600; }
            QLabel#StartupVersion { color: #607066; font-size: 10pt; }
            QLabel#StartupStatus { color: #274B33; font-size: 10pt; }
            QLabel#StartupDetail { color: #6B746E; font-size: 9pt; }
            QProgressBar {
                min-height: 8px;
                max-height: 8px;
                border: 0;
                border-radius: 4px;
                background: #E5EEE7;
                text-align: center;
            }
            QProgressBar::chunk { background: #4FA764; border-radius: 4px; }
            """
        )
        root = QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 22)
        root.setSpacing(14)

        heading = QHBoxLayout()
        heading.setSpacing(16)
        icon = QLabel()
        icon.setObjectName("StartupIcon")
        icon.setFixedSize(64, 64)
        if icon_path and icon_path.exists():
            pixmap = QPixmap(str(icon_path))
        else:
            pixmap = QPixmap(":/LifeCalor.ico")
        if not pixmap.isNull():
            icon.setPixmap(
                pixmap.scaled(
                    64,
                    64,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
            )
        heading.addWidget(icon)

        titles = QVBoxLayout()
        titles.setSpacing(2)
        name = QLabel(APP_NAME)
        name.setObjectName("StartupName")
        version = QLabel(f"版本 {APP_VERSION}")
        version.setObjectName("StartupVersion")
        titles.addWidget(name)
        titles.addWidget(version)
        heading.addLayout(titles, 1)
        root.addLayout(heading)

        self.status_label = QLabel("正在准备启动环境")
        self.status_label.setObjectName("StartupStatus")
        self.status_label.setWordWrap(True)
        root.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setRange(0, self._total_steps)
        self.progress_bar.setValue(0)
        root.addWidget(self.progress_bar)

        self.detail_label = QLabel(f"已完成 0/{self._total_steps} 个初始化步骤")
        self.detail_label.setObjectName("StartupDetail")
        root.addWidget(self.detail_label)

    def show_ready(self, app: QApplication):
        screen = app.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry()
            self.move(available.center() - self.rect().center())
        self.show()
        self.raise_()
        self.activateWindow()
        self.repaint()
        app.processEvents(QEventLoop.ExcludeUserInputEvents)

    def show_busy(self, text, detail=None):
        self.status_label.setText(str(text))
        if detail is not None:
            self.detail_label.setText(str(detail))
        self.repaint()

    def show_step(self, text, current, total):
        total = max(1, int(total))
        current = max(0, min(total, int(current)))
        self.status_label.setText(str(text))
        self.detail_label.setText(f"已完成 {current}/{total} 个初始化步骤")
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(current)
        self.repaint()

    def finish_startup(self, window):
        self.hide()
        self.close()

    def mousePressEvent(self, event):
        event.accept()
