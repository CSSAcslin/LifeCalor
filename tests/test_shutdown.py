import os
import subprocess
import sys
import threading
import time
import unittest
from pathlib import Path

try:
    from .qt_test_support import QtTestCase
except ImportError:
    from qt_test_support import QtTestCase

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "core"))

from PyQt5.QtCore import QObject, QThread, QTimer
from PyQt5.QtWidgets import QApplication, QMainWindow

from ConsoleUtils import ConsoleHandler
from startup.shutdown import WindowShutdownController
from tasks import TaskCoordinator
from history.controller import HistoryController


class ShutdownTests(QtTestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_idle_window_closes_after_the_close_event_returns(self):
        window = QMainWindow()
        controller = WindowShutdownController(window)
        window.show()
        window.close()
        deadline = time.monotonic() + 1
        while window.isVisible() and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.005)
        self.assertFalse(window.isVisible())
        self.assertTrue(controller._finished)

    def test_close_cancels_tasks_and_waits_without_blocking_gui(self):
        started = threading.Event()
        release = threading.Event()

        class BusyThread(QThread):
            def run(self):
                started.set()
                release.wait(3)

        window = QMainWindow()
        window.background_thread = BusyThread(window)
        window.task_coordinator = TaskCoordinator(window)
        task = window.task_coordinator.create_task("计算", "processing")
        window.task_coordinator.start(task.task_id)
        controller = WindowShutdownController(window)
        window.show()
        window.background_thread.start()
        self.assertTrue(started.wait(1))
        try:
            before = time.perf_counter()
            window.close()
            self.assertLess(time.perf_counter() - before, 0.2)
            self.assertTrue(window.isVisible())
            self.assertTrue(task.token.is_cancelled)
            self.assertTrue(window.background_thread.isInterruptionRequested())
            responsive = []
            QTimer.singleShot(0, lambda: responsive.append(True))
            self.app.processEvents()
            self.assertTrue(responsive)
        finally:
            release.set()
            self.assertTrue(window.background_thread.wait(2000))
        deadline = time.monotonic() + 1
        while window.isVisible() and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.005)
        self.assertFalse(window.isVisible())
        self.assertTrue(controller._finished)

    def test_console_handler_close_restores_exception_hooks(self):
        class ConsoleWindow(QMainWindow):
            def log_to_console(self, message):
                pass

            def handle_unhandled_exception(self, *args):
                pass

        window = ConsoleWindow()
        previous_sys, previous_thread = sys.excepthook, threading.excepthook
        handler = ConsoleHandler(window)
        try:
            self.assertEqual(sys.excepthook, handler.handle_uncaught_exception)
            handler.close()
            handler.close()
            self.assertIs(sys.excepthook, previous_sys)
            self.assertIs(threading.excepthook, previous_thread)
        finally:
            handler.close()

    def test_close_waits_for_unparented_cache_write_and_maintenance_threads(self):
        release = threading.Event()

        class CacheThread(QThread):
            def run(self):
                release.wait(3)

        class CacheWorker(QObject):
            cancelled = False

            def cancel(self):
                self.cancelled = True

        window = QMainWindow()
        history = HistoryController(window)
        window.history_controller = history
        write_thread, maintenance_thread = CacheThread(), CacheThread()
        write_worker, maintenance_worker = CacheWorker(), CacheWorker()
        history.cache_write_thread = write_thread
        history.cache_write_worker = write_worker
        history.cache_maintenance_threads['cleanup'] = (maintenance_thread, maintenance_worker)
        WindowShutdownController(window)
        window.show()
        write_thread.start()
        maintenance_thread.start()
        try:
            window.close()
            deadline = time.monotonic() + 0.06
            while time.monotonic() < deadline:
                self.app.processEvents()
                time.sleep(0.005)
            self.assertTrue(window.isVisible())
            self.assertTrue(write_worker.cancelled)
            self.assertTrue(maintenance_worker.cancelled)
            self.assertFalse(history._start_cache_maintenance('cleanup'))
        finally:
            release.set()
            self.assertTrue(write_thread.wait(2000))
            self.assertTrue(maintenance_thread.wait(2000))
        deadline = time.monotonic() + 1
        while window.isVisible() and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.005)
        self.assertFalse(window.isVisible())

    def test_close_waits_for_its_preview_pool_completion(self):
        class PreviewPool:
            stopping = False
            pending = True

            def prepare_shutdown(self):
                self.stopping = True

            def shutdown_pending(self):
                return self.pending

        window = QMainWindow()
        pool = PreviewPool()
        window._convolution_preview_pool = pool
        WindowShutdownController(window)
        window.show()
        window.close()
        deadline = time.monotonic() + 0.06
        while time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.005)
        self.assertTrue(pool.stopping)
        self.assertTrue(window.isVisible())
        pool.pending = False
        deadline = time.monotonic() + 1
        while window.isVisible() and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.005)
        self.assertFalse(window.isVisible())

    def test_task_panel_and_theme_cleanup_exit_normally_in_subprocess(self):
        script = """
import ctypes, os, sys, unittest
from pathlib import Path
if sys.platform == 'win32':
    ctypes.windll.kernel32.SetErrorMode(0x0002)
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
sys.path.insert(0, str(Path('tests').resolve()))
for repetition in range(2):
    suite = unittest.TestSuite([
        unittest.defaultTestLoader.loadTestsFromName('test_app_bootstrap_and_task_panel'),
        unittest.defaultTestLoader.loadTestsFromName('test_appearance'),
    ])
    result = unittest.TextTestRunner().run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
"""
        result = subprocess.run(
            [sys.executable, "-X", "faulthandler", "-c", script],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_late_hardware_callbacks_do_not_start_work_during_close(self):
        script = """
import sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, str(Path('core').resolve()))
from app_bootstrap import configure_high_dpi
configure_high_dpi()
from MainWindow import MainWindow
closing = SimpleNamespace(_lifecalor_closing=True)
MainWindow._start_compute_hardware_probe(closing)
MainWindow._compute_hardware_snapshot_ready(closing, None)
MainWindow._start_compute_cuda_self_test(closing)
MainWindow._compute_cuda_capability_ready(closing, None)
assert vars(closing) == {'_lifecalor_closing': True}
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=ROOT, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=20,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_application_logging_cleanup_preserves_qt_object_ownership(self):
        script = """
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path('core').resolve()))
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
from PyQt5 import sip
from PyQt5.QtCore import QCoreApplication, QEvent, QTimer
from PyQt5.QtWidgets import QApplication, QMainWindow, QMenu
from startup.shutdown import ApplicationResources
app = QApplication([])
window = QMainWindow()
popup = QMenu()
window.show()
resources = ApplicationResources(app)
app.aboutToQuit.connect(resources.prepare_quit)
QTimer.singleShot(0, app.quit)
assert app.exec_() == 0
assert not sip.isdeleted(window), 'window deleted inside Qt quit stack'
resources.close()
assert not sip.isdeleted(window), 'logging cleanup must not delete application windows'
assert not sip.isdeleted(popup), 'shared Qt popup must keep its own ownership'
window.close()
window.deleteLater()
popup.deleteLater()
QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=ROOT, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=20,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
