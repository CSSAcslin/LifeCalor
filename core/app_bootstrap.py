from __future__ import annotations

import os

from PyQt5.QtCore import Qt, QCoreApplication
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QApplication

from app_metadata import APP_NAME, ORGANIZATION_NAME


def configure_high_dpi() -> None:
    """Configure Qt DPI behavior before QApplication is constructed."""
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    # QtWebEngine is imported after the splash creates QApplication.
    QApplication.setAttribute(Qt.AA_ShareOpenGLContexts, True)
    if hasattr(QApplication, "setHighDpiScaleFactorRoundingPolicy"):
        QApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )


def configure_application(app: QApplication, point_size: float = 10.0) -> None:
    """Give top-level dialogs and their tooltips a stable point-sized font."""
    font = QFont(app.font())
    current = font.pointSizeF()
    font.setPointSizeF(max(float(point_size), current if current > 0 else 0.0))
    app.setFont(font)
    QCoreApplication.setOrganizationName(ORGANIZATION_NAME)
    QCoreApplication.setApplicationName(APP_NAME)
