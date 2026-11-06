import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
CORE = Path(__file__).resolve().parents[1] / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from app_bootstrap import configure_high_dpi
configure_high_dpi()

from PyQt5.QtCore import QCoreApplication, QEvent, Qt
from PyQt5.QtTest import QSignalSpy
from PyQt5.QtWidgets import (
    QApplication, QComboBox, QDialogButtonBox, QDockWidget,
    QDoubleSpinBox, QGroupBox, QMainWindow, QPushButton, QScrollArea,
    QSlider, QStackedWidget, QVBoxLayout, QWidget,
)

from MainWindow import MainWindow
from PlotGraphWidget import PlotGraphWidget
from analysis.catalog import ACTION_BY_ID
from analysis.controller import AnalysisController
from analysis.dialog import AnalysisInputDialog, validate_source
from selection import SelectionController
from tasks import TaskCoordinator


class _Source:
    def __init__(self, name, timestamp, shape=(5, 3, 2), axes="THW",
                 processed_type="", time_point=None):
        self.name = name
        self.timestamp = timestamp
        self.datashape = shape
        self.datatype = "float32"
        self.framesize = shape[-2:]
        self.parameters = {"source_axes": axes, "fps": 10, "space_step": 1}
        self.out_processed = {}
        self.time_point = (
            np.arange(shape[0], dtype=np.float64)
            if len(shape) == 3 and time_point is None else time_point
        )
        self.history = []
        if processed_type:
            self.type_processed = processed_type

    @property
    def data_origin(self):
        raise AssertionError("Input preparation must use metadata without loading the array")

    @property
    def data_processed(self):
        raise AssertionError("Input preparation must use metadata without loading the array")


class _ImageDisplay:
    def __init__(self):
        self.display_canvas = []
        self.masks = {}
        self.requested_canvases = []

    def get_draw_roi(self, canvas_id):
        self.requested_canvases.append(canvas_id)
        return None, self.masks.get(canvas_id)

    def add_canvas(self, canvas_id, source, mask=None, anchor=None):
        canvas = SimpleNamespace(
            id=canvas_id,
            data=SimpleNamespace(timestamp_inherited=source.timestamp, framesize=source.framesize),
            anchor_mask=anchor,
            v_rect_roi=None,
        )
        self.display_canvas.append(canvas)
        self.masks[canvas_id] = mask
        return canvas


class _Window(QMainWindow):
    show_analysis_page = MainWindow.show_analysis_page

    def __init__(self):
        super().__init__()
        self.mode = 1
        self.data = _Source("current", 1)
        self.history_source = _Source("history", 2)
        self.processed_data = _Source("preprocessed", 3, processed_type="EM_pre_processed")
        self.sources = [self.data, self.history_source, self.processed_data]
        self.image_display = _ImageDisplay()
        self.selection_controller = SelectionController(self)
        self.graph_plot = PlotGraphWidget(self)
        self.setCentralWidget(self.graph_plot)
        self.calls = []
        self.tasks = []
        self.statuses = []
        self.dialog_calls = []
        self.dialog_accepted = True
        self.selected_source = self.history_source
        self.selected_mask = np.ones(self.history_source.framesize, dtype=bool)
        self.handler_error = None
        self.analysis_controller = AnalysisController(self, dialog_factory=self._dialog_factory)
        for spec in ACTION_BY_ID.values():
            setattr(self, spec.handler, self._execute_handler)
        self._setup_parameter_widgets()
        self.preferences_controller = SimpleNamespace(show=self._noop)
        self.console_dock = QWidget(self)
        self.task_panel = QWidget(self)
        for method in (
            "_toggle_activity_panel", "_sync_theme_actions", "bad_frame_edit_dialog",
            "process_math", "data_crop", "process_convolution", "history_cache_manager", "data_history_clear",
            "data_history_view", "process_history_view", "data_plot_add", "help_show",
            "update_dialog",
        ):
            setattr(self, method, self._noop)
        MainWindow.setup_menus(self)

    def _noop(self, *_args):
        return None

    def _setup_parameter_widgets(self):
        self.fuction_select = QComboBox(self)
        self.fuction_select.addItems(["none", "ultrafast", "EM", "generic", "other"])
        self.fuction_select.setCurrentIndex(3)
        self.general_format_selector = QComboBox(self)
        self.general_format_selector.addItems(["auto", "NPY", "TIFF"])
        self.general_format_selector.setCurrentIndex(2)
        self.between_stack = QStackedWidget(self)
        for _index in range(4):
            self.between_stack.addWidget(QWidget())
        self.FS_mode_combo = QComboBox(self)
        self.FS_mode_combo.addItems(["map", "region", "diffusion"])
        self.EM_mode_combo = QComboBox(self)
        self.EM_mode_combo.addItems(["STFT", "CWT"])
        self.after_process_select = QComboBox(self)
        self.after_process_select.addItems(["whole-cell", "single-channel"])
        self.modes_panel = QGroupBox(self)
        self.time_step_input = QDoubleSpinBox(self)
        self.time_step_input.setValue(0.25)
        self.fps_input = QDoubleSpinBox(self)
        self.fps_input.setValue(50)
        self.time_unit_combo = QComboBox(self)
        self.time_unit_combo.addItems(["ms", "s"])
        self.time_unit_combo.setCurrentIndex(1)
        self.left_panel = QScrollArea(self)
        content = QWidget()
        layout = QVBoxLayout(content)
        for name in (
            "analyze_region_btn", "analyze_btn", "heat_transfer_btn",
            "vector_signal_btn", "preprocess_data_btn", "signal_extract",
        ):
            button = QPushButton(name)
            setattr(self, name, button)
            layout.addWidget(button)
        self.left_panel.setWidget(content)
        self.param_dock = QDockWidget(self)
        self.param_dock.setWidget(self.left_panel)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.param_dock)

    def update_status(self, text):
        self.statuses.append(text)

    def calculator_sources(self):
        return list(self.sources)

    def _dialog_factory(self, window, spec, sources, preferred):
        self.dialog_calls.append((spec, list(sources), preferred))
        return SimpleNamespace(
            source=self.selected_source, mask=self.selected_mask,
            method="max", exec_=lambda: self.dialog_accepted,
        )

    def _execute_handler(self):
        controller = self.analysis_controller
        self.calls.append((
            controller.current_action.action_id,
            self.selection_controller.select_data("all"),
            self.selection_controller.select_roi(True),
            controller.current_method,
        ))
        if self.handler_error is not None:
            raise self.handler_error
        self.tasks.append(controller.current_action.action_id)
        return "submitted"


def _leaf_actions(menu):
    result = []
    for action in menu.actions():
        if action.menu() is not None:
            result.extend(_leaf_actions(action.menu()))
        elif not action.isSeparator():
            result.append(action)
    return result


class AnalysisMenuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.window = _Window()

    def tearDown(self):
        self.window.close()
        self.window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)

    def test_menu_covers_computations_without_duplicate_data_editors(self):
        menu_titles = [action.text() for action in self.window.menuBar().actions()]
        analysis_index = menu_titles.index("分析与处理")
        self.assertEqual(menu_titles[analysis_index - 1:analysis_index + 2], [
            "控制台", "分析与处理", "高级操作",
        ])
        leaves = _leaf_actions(self.window.analysis_menu)
        identifiers = [action.objectName() for action in leaves if action.objectName()]
        expected = {
            "preprocess.em", "statistics.roi", "statistics.mean",
            "statistics.distribution", "statistics.time_mean", "transform.stft",
            "transform.stft_quality", "transform.cwt", "transform.cwt_quality",
            "transform.fft2", "transform.ifft2", "lifetime.region", "lifetime.map",
            "physics.diffusion", "physics.heat", "channel.gaussian",
            "channel.simple", "motion.heartbeat",
        }
        self.assertEqual(set(identifiers), expected)
        self.assertEqual(len(identifiers), len(expected))
        text = [action.text() for action in leaves]
        self.assertFalse(any("坏帧" in title or "坏点" in title for title in text))
        self.assertNotIn("数据计算器", text)
        self.assertNotIn("数据切片器", text)
        advanced = next(
            action.menu() for action in self.window.menuBar().actions()
            if action.text() == "高级操作"
        )
        advanced_titles = [action.text() for action in _leaf_actions(advanced)]
        self.assertEqual(advanced_titles, ["坏点处理", "数据计算器", "数据切片器", "时空卷积"])
        self.assertTrue(set(advanced_titles).isdisjoint(text))

    def test_curve_submenu_shares_all_four_existing_actions(self):
        curves = next(action.menu() for action in self.window.analysis_menu.actions()
                      if action.text() == "曲线分析")
        graph = self.window.graph_plot
        self.assertEqual(curves.actions(), [
            graph.action_psd, graph.action_hist, graph.action_time, graph.action_peak,
        ])
        x = np.arange(32) / 32.0
        graph.plot_data(np.column_stack((x, np.sin(2 * np.pi * 4 * x))), time_unit="s")
        curves.actions()[0].trigger()
        self.assertTrue(graph.action_psd.isChecked())
        curves.actions()[2].trigger()
        self.assertTrue(graph.action_time.isChecked())
        self.assertFalse(graph.action_psd.isChecked())
        self.assertEqual(self.window.tasks, [])

    def test_menu_parameter_navigation_keeps_import_and_units_and_creates_no_task(self):
        controller = self.window.analysis_controller
        for action_id, expected_page, expected_fs in (
            ("lifetime.map", 1, 0), ("lifetime.region", 1, 1),
            ("physics.diffusion", 1, 2), ("physics.heat", 1, 0),
            ("preprocess.em", 2, None),
        ):
            with self.subTest(action_id=action_id):
                controller.actions[action_id].trigger()
                self.assertEqual(self.window.between_stack.currentIndex(), expected_page)
                if expected_fs is not None:
                    self.assertEqual(self.window.FS_mode_combo.currentIndex(), expected_fs)
                self.assertEqual(self.window.fuction_select.currentIndex(), 3)
                self.assertEqual(self.window.general_format_selector.currentIndex(), 2)
                self.assertEqual(self.window.time_step_input.value(), 0.25)
                self.assertEqual(self.window.fps_input.value(), 50)
                self.assertEqual(self.window.time_unit_combo.currentText(), "s")
        self.assertEqual(self.window.dialog_calls, [])
        self.assertEqual(self.window.tasks, [])
        self.assertEqual(self.window.calls, [])

    def test_confirmation_submits_exact_source_roi_and_method_then_restores_context(self):
        result = self.window.analysis_controller.execute("statistics.roi")
        self.assertEqual(result, "submitted")
        action_id, source, mask, method = self.window.calls[0]
        self.assertEqual(action_id, "statistics.roi")
        self.assertIs(source, self.window.history_source)
        np.testing.assert_array_equal(mask, self.window.selected_mask)
        self.window.selected_mask[:] = False
        self.assertTrue(mask.all())
        self.assertEqual(method, "max")
        self.assertIs(self.window.selection_controller.select_data("data"), self.window.data)
        self.assertIsNone(self.window.analysis_controller.current_input)

    def test_cancelled_input_does_not_invoke_handler_or_create_task(self):
        self.window.dialog_accepted = False
        self.window.analysis_controller.open("transform.stft")
        self.assertEqual(len(self.window.dialog_calls), 1)
        self.assertIs(self.window.dialog_calls[0][2], self.window.processed_data)
        self.assertEqual(self.window.calls, [])
        self.assertEqual(self.window.tasks, [])
        self.assertEqual(self.window.fuction_select.currentIndex(), 3)

    def test_exception_restores_nested_prepared_context(self):
        controller = self.window.analysis_controller
        outer = _Source("outer", 9)
        outer_mask = np.zeros(outer.framesize, dtype=bool)
        controller.current_action = "outer-action"
        controller.current_input = outer
        controller.current_method = "sum"
        self.window.handler_error = RuntimeError("submission failed")
        with self.window.selection_controller.prepared(outer, outer_mask):
            with self.assertRaisesRegex(RuntimeError, "submission failed"):
                controller.execute("statistics.roi")
            self.assertIs(self.window.selection_controller.select_data("data"), outer)
            self.assertIs(self.window.selection_controller.select_roi(), outer_mask)
        self.assertEqual(controller.current_action, "outer-action")
        self.assertIs(controller.current_input, outer)
        self.assertEqual(controller.current_method, "sum")
        self.assertIs(self.window.selection_controller.select_data("data"), self.window.data)

    def test_empty_history_does_not_open_input_dialog(self):
        self.window.sources = []
        with patch("analysis.controller.report_warning") as warning:
            self.assertFalse(self.window.analysis_controller.execute("statistics.mean"))
        self.assertEqual(self.window.dialog_calls, [])
        self.assertEqual(self.window.tasks, [])
        warning.assert_called_once()

    def test_real_input_dialog_uses_exact_preferred_source_and_only_matching_nonempty_rois(self):
        other = self.window.data
        selected = self.window.history_source
        selected.name = other.name
        valid = np.ones(selected.framesize, dtype=bool)
        anchor = np.zeros(selected.framesize, dtype=bool)
        anchor[0, 0] = True
        display = self.window.image_display
        display.add_canvas(10, other, mask=np.ones(other.framesize, dtype=bool))
        display.add_canvas(20, selected, mask=valid, anchor=anchor)
        display.add_canvas(30, selected, mask=np.ones((1, 1), dtype=bool))
        display.add_canvas(40, selected, mask=np.zeros(selected.framesize, dtype=bool))
        dialog = AnalysisInputDialog(self.window, ACTION_BY_ID["statistics.roi"],
                                     [other, selected], preferred=selected)
        try:
            self.assertIs(dialog.source, selected)
            self.assertEqual(dialog.roi_combo.count(), 2)
            self.assertIs(dialog.mask, valid)
            self.assertNotIn(10, display.requested_canvases)
            dialog.roi_combo.setCurrentIndex(1)
            self.assertIs(dialog.mask, anchor)
            self.assertTrue(dialog.buttons.button(QDialogButtonBox.Ok).isEnabled())
            dialog.source_combo.setCurrentIndex(0)
            self.assertIs(dialog.source, other)
            self.assertEqual(dialog.roi_combo.count(), 1)
            self.assertTrue(dialog.roi_combo.itemText(0).startswith("画布 10"))
        finally:
            dialog.reject()

    def test_missing_matching_roi_blocks_confirmation(self):
        display = self.window.image_display
        display.add_canvas(10, self.window.data,
                           mask=np.ones(self.window.data.framesize, dtype=bool))
        dialog = AnalysisInputDialog(self.window, ACTION_BY_ID["statistics.roi"],
                                     [self.window.history_source])
        try:
            self.assertEqual(dialog.roi_combo.count(), 0)
            self.assertFalse(dialog.buttons.button(QDialogButtonBox.Ok).isEnabled())
        finally:
            dialog.reject()

    def test_video_image_and_complete_spectrum_have_distinct_preconditions(self):
        video = self.window.data
        image = _Source("image", 4, shape=(3, 2), axes="HW")
        self.assertEqual(validate_source(ACTION_BY_ID["statistics.time_mean"], video), "")
        self.assertTrue(validate_source(ACTION_BY_ID["statistics.time_mean"], image))
        self.assertEqual(validate_source(ACTION_BY_ID["transform.fft2"], image), "")
        self.assertTrue(validate_source(ACTION_BY_ID["transform.ifft2"], image))
        image.out_processed["twoD_FFT"] = np.zeros((3, 2), dtype=np.float32)
        self.assertTrue(validate_source(ACTION_BY_ID["transform.ifft2"], image))
        image.out_processed["twoD_FFT"] = np.zeros((3, 2), dtype=np.complex64)
        self.assertEqual(validate_source(ACTION_BY_ID["transform.ifft2"], image), "")

    def test_time_axis_and_preprocessing_requirements_are_checked_before_execution(self):
        for axes in ("HWT", ""):
            source = _Source("time-last", 5, shape=(3, 2, 5), axes=axes,
                             time_point=np.arange(5))
            with self.subTest(axes=axes):
                self.assertTrue(validate_source(ACTION_BY_ID["statistics.time_mean"], source))
                self.assertTrue(validate_source(ACTION_BY_ID["transform.fft2"], source))
        self.assertTrue(validate_source(ACTION_BY_ID["transform.stft"], self.window.data, mode=1))
        self.assertEqual(validate_source(ACTION_BY_ID["transform.stft"], self.window.data, mode=2), "")
        source = _Source("bad-times", 6, time_point=np.array([0, 1, 1, 3, 4]))
        self.assertTrue(validate_source(ACTION_BY_ID["lifetime.map"], source))

    def test_restored_vector_result_drives_slider_without_raw_data(self):
        self.window.data = None
        self.window.vector_source = self.window.history_source
        self.window.vector_array = np.zeros((7, 4, 2), dtype=np.float64)
        slider = QSlider(Qt.Vertical, self.window)
        self.window.time_slider_vertical = slider
        slider.setRange(0, 99)
        slider.setValue(50)
        slider.hide()

        MainWindow._handle_result_tab(self.window, "roi")
        self.assertFalse(slider.isHidden())
        self.assertEqual(slider.maximum(), 6)
        self.assertEqual(slider.value(), 0)

        source = self.window.vector_source
        vector = self.window.vector_array
        for missing_source, missing_array in ((None, vector), (source, None), (None, None)):
            with self.subTest(source=missing_source is None, array=missing_array is None):
                self.window.vector_source = missing_source
                self.window.vector_array = missing_array
                slider.show()
                MainWindow._handle_result_tab(self.window, "roi")
                self.assertTrue(slider.isHidden())

    def test_delayed_submission_records_explicit_action_and_source_without_prepared_context(self):
        coordinator = TaskCoordinator(self.window)
        self.window.task_coordinator = coordinator
        updates = QSignalSpy(coordinator.task_updated)
        created = []

        def begin(thread_name, category, name=None):
            task = coordinator.create_task(name, category)
            task.execution_details["existing"] = "preserved"
            created.append((thread_name, category, task))
            return task

        self.window.processing_controller = SimpleNamespace(begin=begin)
        self.assertIsNone(self.window.analysis_controller.current_action)
        self.assertIsNone(self.window.analysis_controller.current_input)
        for action_id, thread, category, source in (
            ("physics.diffusion", "calc_thread", "calculation", self.window.history_source),
            ("motion.heartbeat", "avi_thread", "em_processing", self.window.processed_data),
        ):
            with self.subTest(action_id=action_id):
                self.assertTrue(MainWindow.ensure_task_thread_running(
                    self.window, thread, category, action_id=action_id, source=source,
                ))
                recorded_thread, recorded_category, task = created[-1]
                self.assertEqual((recorded_thread, recorded_category), (thread, category))
                self.assertEqual(task.name, ACTION_BY_ID[action_id].title.rstrip("…"))
                self.assertEqual(task.execution_details, {
                    "existing": "preserved",
                    "analysis_action_id": action_id,
                    "input_name": source.name,
                    "input_timestamp": source.timestamp,
                })
                self.assertIs(updates[-1][0], task)
        self.assertEqual(len(updates), 2)


if __name__ == "__main__":
    unittest.main()
