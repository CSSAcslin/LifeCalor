import logging
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from PyQt5.QtCore import QThread, Qt
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow

import resources_rc  # Register resources while the real Qt modules are active.

from startup.bootstrap import TOTAL_STARTUP_STEPS, launch_main_window
from startup.logging_setup import install_early_logging, take_startup_messages
from startup.splash import StartupSplash
from launcher import _configure_application_icon, _configure_appearance


class StartupFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_success_uses_existing_application_and_defers_background_services(self):
        events = []
        splash = StartupSplash()

        class FakeWindow(QMainWindow):
            def __init__(self, startup_reporter=None):
                super().__init__()
                startup_reporter.attach_window(self)
                startup_reporter.step("测试设置", 1)
                startup_reporter.step("测试服务", TOTAL_STARTUP_STEPS - 1)

            def start_deferred_services(self):
                events.append("deferred")

        app_before = QApplication.instance()
        window = launch_main_window(
            self.app,
            splash=splash,
            window_factory=FakeWindow,
            started_at=time.perf_counter(),
        )

        self.assertIs(QApplication.instance(), app_before)
        self.assertIsNotNone(window)
        self.assertTrue(window.isVisible())
        self.assertFalse(bool(window.windowFlags() & Qt.WindowStaysOnTopHint))
        self.assertFalse(splash.isVisible())
        self.assertEqual(events, [])
        self.app.processEvents()
        self.assertEqual(events, ["deferred"])
        window.close()

    def test_failure_is_reported_once_and_partial_thread_is_stopped(self):
        reported = []
        splash = StartupSplash()
        created_thread = []

        class FailingWindow(QMainWindow):
            def __init__(self, startup_reporter=None):
                super().__init__()
                startup_reporter.attach_window(self)
                self.import_thread = QThread(self)
                self.import_thread.start()
                created_thread.append(self.import_thread)
                raise RuntimeError("startup test failure")

        window = launch_main_window(
            self.app,
            splash=splash,
            window_factory=FailingWindow,
            error_reporter=lambda parent, error: reported.append((parent, error)),
        )

        self.assertIsNone(window)
        self.assertFalse(splash.isVisible())
        self.assertEqual(len(reported), 1)
        self.assertEqual(reported[0][1].stage, "程序启动")
        self.assertIn("RuntimeError: startup test failure", reported[0][1].details)
        self.assertIn("log_file", reported[0][1].context)
        self.assertFalse(created_thread[0].isRunning())

    def test_embedded_icon_is_used_when_packaged_file_is_absent(self):
        _configure_application_icon(self.app)
        _configure_appearance(self.app)
        splash = StartupSplash(CORE / "missing-packaged-icon.ico")
        icon_label = splash.findChild(QLabel, "StartupIcon")

        self.assertFalse(self.app.windowIcon().isNull())
        self.assertIsNotNone(icon_label)
        self.assertIsNotNone(icon_label.pixmap())
        self.assertFalse(icon_label.pixmap().isNull())
        splash.close()

    def test_splash_is_temporarily_topmost_and_total_progress_never_resets(self):
        splash = StartupSplash(total_steps=8)
        self.assertTrue(bool(splash.windowFlags() & Qt.WindowStaysOnTopHint))
        self.assertEqual((splash.progress_bar.minimum(), splash.progress_bar.maximum()), (0, 8))
        self.assertEqual(splash.progress_bar.value(), 0)

        splash.show_step("第一步", 1, 8)
        splash.show_busy("继续加载")
        self.assertEqual(splash.progress_bar.value(), 1)
        splash.show_step("第二步", 2, 8)
        self.assertEqual(splash.progress_bar.value(), 2)
        self.assertEqual((splash.progress_bar.minimum(), splash.progress_bar.maximum()), (0, 8))
        splash.close()

    def test_click_does_not_dismiss_splash(self):
        splash = StartupSplash()
        splash.show_ready(self.app)

        class Event:
            accepted = False

            def accept(self):
                self.accepted = True

        event = Event()
        splash.mousePressEvent(event)
        self.assertTrue(event.accepted)
        self.assertTrue(splash.isVisible())
        splash.close()

    def test_early_logging_reuses_one_file_handler_and_drains_console_buffer(self):
        root = logging.getLogger()
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "startup.log"
            install_early_logging(log_path)
            install_early_logging(log_path)
            file_handlers = [
                handler
                for handler in root.handlers
                if getattr(handler, "lifecalor_file_handler", False)
            ]
            self.assertEqual(len(file_handlers), 1)
            messages = take_startup_messages()
            self.assertTrue(any("启动引导已开始" in message for message in messages))

            for handler in list(root.handlers):
                if getattr(handler, "lifecalor_file_handler", False):
                    root.removeHandler(handler)
                    handler.close()


if __name__ == "__main__":
    unittest.main()
