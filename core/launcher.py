from __future__ import annotations

import logging
import multiprocessing
import sys
import time
from pathlib import Path

from PyQt5.QtGui import QFontDatabase, QIcon
from PyQt5.QtWidgets import QApplication

from app_bootstrap import configure_application, configure_high_dpi


def _resource_root():
    frozen_root = getattr(sys, "_MEIPASS", None)
    return Path(frozen_root) if frozen_root else Path(__file__).resolve().parent


def _configure_application_icon(app):
    """Register the icon before MainWindow so the early splash can use it."""
    try:
        import resources_rc  # noqa: F401
    except ImportError:
        logging.warning("无法注册 Qt 图标资源", exc_info=True)

    icon_path = _resource_root() / "LifeCalor.ico"
    icon = QIcon(str(icon_path)) if icon_path.exists() else QIcon(":/LifeCalor.ico")
    if not icon.isNull():
        app.setWindowIcon(icon)
    else:
        logging.warning("LifeCalor 应用图标不可用: %s", icon_path)
    return icon_path


def _configure_appearance(app):
    app.setStyle("Fusion")
    for font_path in (
        "C:/Windows/Fonts/NotoSansSC-VF.ttf",
        "C:/Windows/Fonts/calibril.ttf",
    ):
        if Path(font_path).exists():
            QFontDatabase.addApplicationFont(font_path)
    try:
        from appearance import install_theme_manager

        return install_theme_manager(app)
    except Exception:
        logging.error("界面主题初始化失败，使用 Fusion 安全外观", exc_info=True)
        app.setStyleSheet("")
        return None


def main():
    multiprocessing.freeze_support()
    started_at = time.perf_counter()
    configure_high_dpi()

    app = QApplication.instance()
    owns_app = app is None
    if app is None:
        app = QApplication(sys.argv)
    configure_application(app)

    from startup.logging_setup import install_early_logging
    from startup.splash import StartupSplash

    log_file = install_early_logging()
    icon_path = _configure_application_icon(app)
    _configure_appearance(app)
    splash = StartupSplash(icon_path)
    splash.show_ready(app)
    logging.info("启动页首次可见耗时: %.3fs", time.perf_counter() - started_at)

    from startup.bootstrap import launch_main_window

    window = launch_main_window(
        app,
        splash=splash,
        log_file=log_file,
        started_at=started_at,
    )
    app._lifecalor_main_window = window
    if owns_app:
        return app.exec_()
    return 0 if window is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
