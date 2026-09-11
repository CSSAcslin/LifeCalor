import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from PyQt5.QtWidgets import QApplication, QDialog, QPushButton

from app_bootstrap import configure_application
from tasks import TaskCoordinator, TaskStatus
from tasks.panel import TaskPanel


class ApplicationBootstrapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_dialog_children_inherit_stable_point_sized_application_font(self):
        configure_application(self.app, 10.0)
        dialog = QDialog()
        button = QPushButton("确定", dialog)

        self.assertGreaterEqual(dialog.font().pointSizeF(), 10.0)
        self.assertGreaterEqual(button.font().pointSizeF(), 10.0)

    def test_high_dpi_configuration_precedes_qapplication_creation(self):
        source = (CORE / "launcher.py").read_text(encoding="utf-8")
        self.assertLess(source.index("configure_high_dpi()"), source.index("QApplication(sys.argv)"))
        self.assertLess(source.index("QApplication(sys.argv)"), source.index("configure_application(app)"))

    def test_webengine_context_sharing_is_configured_before_application(self):
        bootstrap = (CORE / "app_bootstrap.py").read_text(encoding="utf-8")
        launcher = (CORE / "launcher.py").read_text(encoding="utf-8")
        self.assertIn("Qt.AA_ShareOpenGLContexts", bootstrap)
        self.assertLess(launcher.index("configure_high_dpi()"), launcher.index("QApplication(sys.argv)"))

    def test_legacy_mainwindow_entry_redirects_before_heavy_imports(self):
        source = (CORE / "MainWindow.py").read_text(encoding="utf-8")
        self.assertLess(source.index("from launcher import main"), source.index("import resources_rc"))


class TaskPanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_large_progress_is_displayed_as_bounded_percentage(self):
        coordinator = TaskCoordinator()
        panel = TaskPanel(coordinator)
        task = coordinator.create_task("大文件导入", "import")
        coordinator.start(task.task_id, total=10**12)
        coordinator.progress(task.task_id, 5 * 10**11, 10**12, "读取中")

        _item, progress, cancel = panel._rows[task.task_id]
        self.assertEqual(progress.minimum(), 0)
        self.assertEqual(progress.maximum(), 100)
        self.assertEqual(progress.value(), 50)
        self.assertTrue(cancel.isEnabled())
        panel.close()

    def test_cancel_button_only_requests_cancellation(self):
        callback_calls = []
        coordinator = TaskCoordinator()
        panel = TaskPanel(coordinator)
        task = coordinator.create_task(
            "计算", "calculation", cancel_callback=lambda: callback_calls.append(True)
        )
        coordinator.start(task.task_id)
        panel._rows[task.task_id][2].click()

        self.assertEqual(task.status, TaskStatus.CANCELLING)
        self.assertTrue(task.token.is_cancelled)
        self.assertEqual(callback_calls, [True])
        panel.close()

    def test_mainwindow_places_tasks_in_the_console_activity_tabs(self):
        source = (CORE / "MainWindow.py").read_text(encoding="utf-8")

        self.assertIn('self._add_activity_panel(self.console_dock, "控制台")', source)
        self.assertIn('self._add_activity_panel(self.task_panel, "任务")', source)
        self.assertIn("result_splitter.addWidget(self.activity_tabs)", source)
        self.assertNotIn("self.addDockWidget(Qt.BottomDockWidgetArea, self.task_panel)", source)


if __name__ == "__main__":
    unittest.main()
