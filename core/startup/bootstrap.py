from __future__ import annotations

import logging
import time
import traceback
from pathlib import Path

from PyQt5.QtCore import QEventLoop, QThread, QTimer

from .logging_setup import resolve_log_path
from .splash import StartupSplash


TOTAL_STARTUP_STEPS = 8


class StartupSession:
    """Bridge real MainWindow construction stages to the visible splash."""

    def __init__(self, app, splash, log_file=None, started_at=None):
        self.app = app
        self.splash = splash
        self.log_file = Path(log_file or resolve_log_path())
        self.started_at = started_at if started_at is not None else time.perf_counter()
        self.window = None

    def attach_window(self, window):
        self.window = window

    def busy(self, text, detail="正在加载核心模块"):
        self.splash.show_busy(text, detail)
        logging.info("启动阶段: %s (%.3fs)", text, time.perf_counter() - self.started_at)
        self._refresh()

    def step(self, text, current, total=TOTAL_STARTUP_STEPS):
        self.splash.show_step(text, current, total)
        logging.info(
            "启动阶段 %d/%d: %s (%.3fs)",
            current,
            total,
            text,
            time.perf_counter() - self.started_at,
        )
        self._refresh()

    def _refresh(self):
        self.app.processEvents(QEventLoop.ExcludeUserInputEvents)


def _cleanup_partial_window(window):
    if window is None:
        return
    try:
        window.hide()
    except (AttributeError, RuntimeError):
        pass

    for name in ("imp_thread", "dat_thread", "proc_thread", "cal_thread", "mass_data_processor"):
        worker = getattr(window, name, None)
        for method_name in ("cancel", "stop", "request_stop"):
            method = getattr(worker, method_name, None)
            if callable(method):
                try:
                    method()
                except Exception:
                    logging.debug("启动失败清理 worker 失败: %s.%s", name, method_name, exc_info=True)
                break

    seen = set()
    for name in ("import_thread", "data_thread", "process_thread", "calc_thread", "avi_thread"):
        thread = getattr(window, name, None)
        if not isinstance(thread, QThread) or id(thread) in seen:
            continue
        seen.add(id(thread))
        try:
            thread.requestInterruption()
            thread.quit()
            if thread.isRunning() and not thread.wait(1500):
                logging.warning("启动失败清理线程超时: %s", name)
            thread.deleteLater()
        except RuntimeError:
            pass

    try:
        window.deleteLater()
    except (AttributeError, RuntimeError):
        pass


def _report_startup_failure(error_reporter, exc, log_file):
    details = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).rstrip()
    try:
        from diagnostics import AppError, show_app_error

        app_error = AppError(
            title="LifeCalor 启动失败",
            message="程序初始化未完成，详细信息已写入日志。",
            stage="程序启动",
            severity="critical",
            details=details,
            original=exc,
            context={"log_file": str(log_file)},
        )
        reporter = error_reporter or show_app_error
        reporter(None, app_error)
        return
    except Exception:
        logging.exception("启动错误提交到统一错误队列失败")

    from PyQt5.QtWidgets import QMessageBox

    QMessageBox.critical(
        None,
        "LifeCalor 启动失败",
        f"程序初始化未完成。\n\n日志位置：{log_file}",
    )


def launch_main_window(
    app,
    splash=None,
    window_factory=None,
    log_file=None,
    started_at=None,
    error_reporter=None,
):
    """Show startup feedback, construct MainWindow, and hand over exactly once."""

    started_at = started_at if started_at is not None else time.perf_counter()
    log_file = Path(log_file or resolve_log_path())
    if splash is None:
        icon_path = Path(__file__).resolve().parents[1] / "LifeCalor.ico"
        splash = StartupSplash(icon_path, total_steps=TOTAL_STARTUP_STEPS)

    session = StartupSession(app, splash, log_file=log_file, started_at=started_at)
    try:
        if not splash.isVisible():
            splash.show_ready(app)
            logging.info("启动页首次可见耗时: %.3fs", time.perf_counter() - started_at)
        else:
            app.processEvents(QEventLoop.ExcludeUserInputEvents)
        session.busy("正在加载核心模块")

        if window_factory is None:
            from MainWindow import MainWindow

            window_factory = MainWindow

        window = window_factory(startup_reporter=session)
        if session.window is None:
            session.attach_window(window)
        if not app.windowIcon().isNull():
            window.setWindowIcon(app.windowIcon())

        session.step("正在显示工作区", TOTAL_STARTUP_STEPS)
        window.show()
        app.processEvents(QEventLoop.ExcludeUserInputEvents)
        splash.finish_startup(window)
        QTimer.singleShot(0, getattr(window, "start_deferred_services", lambda: None))
        logging.info("主窗口可见，启动总耗时: %.3fs", time.perf_counter() - started_at)
        return window
    except BaseException as exc:
        _cleanup_partial_window(session.window)
        try:
            splash.hide()
            splash.close()
        except RuntimeError:
            pass
        _report_startup_failure(error_reporter, exc, log_file)
        return None
