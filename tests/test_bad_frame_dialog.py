import os
import sys
import unittest
try:
    from .qt_test_support import QtTestCase
except ImportError:
    from qt_test_support import QtTestCase
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
CORE = Path(__file__).resolve().parents[1] / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtTest import QSignalSpy
from PyQt5.QtWidgets import QApplication, QWidget

from diagnostics import AppError
from ExtraDialog import BadFrameDialog


class _Worker(QObject):
    bad_frames_detected = pyqtSignal(object)
    bad_frames_fixed = pyqtSignal(object)
    bad_frame_failed = pyqtSignal(object)


class _Window(QWidget):
    detect_bad_frames_auto_signal = pyqtSignal(np.ndarray, float)
    fix_bad_frames_signal = pyqtSignal(object, list, int)

    def __init__(self, data):
        super().__init__()
        self.data = data
        self.proc_thread = _Worker(self)


def _data(name="source"):
    array = np.arange(20, dtype=np.float32).reshape(5, 2, 2)
    return SimpleNamespace(name=name, data_origin=array, datashape=array.shape, timestamp=1.0)


class BadFrameDialogTests(QtTestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.source = _data()
        self.window = _Window(self.source)
        self.dialog = BadFrameDialog(self.window)
        self.dialog.show()
        self.detect_requests = QSignalSpy(self.window.detect_bad_frames_auto_signal)
        self.fix_requests = QSignalSpy(self.window.fix_bad_frames_signal)
        self.accepted = QSignalSpy(self.dialog.accepted)

    def tearDown(self):
        self.dialog.close()
        self.window.close()

    def test_auto_detection_waits_then_repairs_captured_data_once(self):
        self.window.data = _data("changed-current-data")
        self.assertIn("source", self.dialog.source_label.text())
        self.assertIn("(5, 2, 2)", self.dialog.source_label.text())
        self.assertNotIn("changed-current-data", self.dialog.source_label.text())
        self.dialog.apply_fix()
        self.dialog.apply_fix()

        self.assertEqual(len(self.detect_requests), 1)
        self.assertIs(self.detect_requests[0][0], self.source.data_origin)
        self.assertFalse(self.dialog.apply_btn.isEnabled())
        self.assertEqual(len(self.fix_requests), 0)
        self.assertEqual(len(self.accepted), 0)

        self.window.proc_thread.bad_frames_detected.emit([3, 1, 3])
        self.assertEqual(len(self.fix_requests), 1)
        self.assertIs(self.fix_requests[0][0], self.source)
        self.assertEqual(self.fix_requests[0][1], [1, 3])
        self.assertEqual(len(self.accepted), 0)
        self.assertFalse(self.dialog.apply_btn.isEnabled())

        self.window.proc_thread.bad_frames_detected.emit([2])
        self.assertEqual(len(self.fix_requests), 1)
        self.window.proc_thread.bad_frames_fixed.emit(self.source)
        self.assertEqual(len(self.accepted), 1)
        self.assertEqual(self.dialog.result(), self.dialog.Accepted)
        self.assertIs(self.dialog.repaired_data, self.source)
        self.assertEqual(self.dialog._subscriptions, [])

    def test_manual_frames_are_validated_and_finish_only_for_matching_data(self):
        self.dialog.manual_radio.setChecked(True)
        self.dialog.frame_input.setText("3，1;3")
        self.dialog.n_frames_spin.setValue(4)
        self.dialog.apply_fix()

        self.assertEqual(len(self.detect_requests), 0)
        self.assertEqual(self.fix_requests[0][1:], [[1, 3], 4])
        self.assertEqual(len(self.accepted), 0)
        self.window.proc_thread.bad_frames_fixed.emit(_data("another-source"))
        self.assertEqual(len(self.accepted), 0)
        self.assertFalse(self.dialog.apply_btn.isEnabled())
        repaired = _data("repaired")
        repaired.parameters = {"bad_frame_source_timestamp": self.source.timestamp}
        self.window.proc_thread.bad_frames_fixed.emit(repaired)
        self.assertEqual(len(self.accepted), 1)
        self.assertIs(self.dialog.repaired_data, repaired)

    def test_invalid_manual_indices_do_not_submit_work(self):
        self.dialog.manual_radio.setChecked(True)
        with patch("ExtraDialog.report_warning") as warning:
            for text in ("", "-1", "5", "1.2", "one", "0,8"):
                with self.subTest(text=text):
                    self.dialog.frame_input.setText(text)
                    self.dialog.apply_fix()
                    self.assertEqual(len(self.fix_requests), 0)
                    self.assertTrue(self.dialog.apply_btn.isEnabled())
            self.assertEqual(warning.call_count, 6)

    def test_no_detected_frames_keeps_dialog_ready_for_another_selection(self):
        self.dialog.apply_fix()
        self.window.proc_thread.bad_frames_detected.emit([])
        self.assertEqual(len(self.fix_requests), 0)
        self.assertEqual(len(self.accepted), 0)
        self.assertTrue(self.dialog.apply_btn.isEnabled())
        self.assertIn("未检测到", self.dialog.status_label.text())

    def test_failure_restores_controls_and_reports_same_error(self):
        self.dialog.manual_radio.setChecked(True)
        self.dialog.frame_input.setText("1")
        self.dialog.apply_fix()
        error = AppError("坏帧修复失败", "没有可用参考帧", stage="坏帧修复")
        with patch("diagnostics.show_app_error") as report:
            self.window.proc_thread.bad_frame_failed.emit(error)
            report.assert_called_once_with(self.dialog, error)
        self.assertEqual(len(self.accepted), 0)
        self.assertTrue(self.dialog.apply_btn.isEnabled())
        self.assertTrue(self.dialog.manual_radio.isEnabled())
        self.assertEqual(self.dialog.status_label.text(), error.message)

    def test_reject_while_detecting_disconnects_late_results(self):
        self.dialog.apply_fix()
        self.dialog.reject()
        self.assertEqual(self.dialog._subscriptions, [])
        self.window.proc_thread.bad_frames_detected.emit([1])
        self.assertEqual(len(self.fix_requests), 0)
        self.assertEqual(len(self.accepted), 0)

    def test_close_while_fixing_cannot_accept_on_late_completion(self):
        self.dialog.manual_radio.setChecked(True)
        self.dialog.frame_input.setText("1")
        self.dialog.apply_fix()
        self.dialog.close()
        self.window.proc_thread.bad_frames_fixed.emit(self.source)
        self.assertEqual(len(self.accepted), 0)
        self.assertEqual(self.dialog._subscriptions, [])

    def test_explicit_source_overrides_parent_current_data(self):
        explicit = _data("explicit-source")
        dialog = BadFrameDialog(self.window, data=explicit)
        try:
            dialog.apply_fix()
            self.assertIs(self.detect_requests[0][0], explicit.data_origin)
        finally:
            dialog.reject()


if __name__ == "__main__":
    unittest.main()
