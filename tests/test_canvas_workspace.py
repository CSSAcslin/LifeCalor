import os
import sys
import time
import tempfile
import unittest
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from PyQt5.QtCore import QByteArray, QEventLoop, QSettings, Qt
from PyQt5.QtWidgets import QApplication, QDockWidget, QMainWindow

from DataManager import Data, ImagingData
from ImageDisplayWindow import ImageDisplayWindow


TOOL_PARAMS = {
    "pen_size": 2,
    "pen_color": "#008000",
    "fill_color": "#006400",
    "vector_color": "#FFFF00",
    "anchor_select": False,
    "anchor_shape": "square",
    "anchor_size": 5,
    "anchor_method": "mean",
    "angle_step": 0.7853981633974483,
    "auto_fill": False,
    "vector_width": 2,
    "colormap": "Jet",
    "use_colormap": False,
    "auto_boundary_set": True,
    "min_value": "",
    "max_value": "",
}


class HostWindow(QMainWindow):
    def __init__(self, settings=None):
        super().__init__()
        self.focus_canvas = None
        self.settings = settings

    def canvas_signal_connect(self):
        pass


def make_image(name, value=0):
    raw = np.full((4, 9, 13), value, dtype=np.float32)
    raw[:, 2:5, 4:8] += 1
    data = Data(
        data_origin=raw,
        time_point=np.arange(4, dtype=np.float64),
        format_import="unit",
        image_import=raw,
        parameters={"fps": 20},
        name=name,
    )
    return ImagingData.create_image(data)


class CanvasWorkspaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.host = HostWindow()
        self.display = ImageDisplayWindow(dict(TOOL_PARAMS), self.host)
        self.display.resize(1000, 720)
        self.display.show()
        self._settle()

    def tearDown(self):
        self.display.del_canvas(-1)
        deadline = time.perf_counter() + 2.0
        while self.display._retiring_canvases and time.perf_counter() < deadline:
            self._settle()
        self.display.close()
        self.host.close()
        self._settle()

    def _settle(self, rounds=5):
        for _ in range(rounds):
            self.app.processEvents(QEventLoop.AllEvents, 50)

    def add_canvases(self, count):
        canvases = [self.display.add_canvas(make_image(f"data-{index}", index)) for index in range(count)]
        self._settle()
        return canvases

    def assert_near(self, left, right, tolerance=0.08):
        largest = max(abs(left), abs(right), 1)
        self.assertLessEqual(abs(left - right) / largest, tolerance)

    def test_default_four_canvas_layout_is_balanced_two_by_two(self):
        docks = self.add_canvases(4)

        left = sorted(docks, key=lambda dock: dock.geometry().x())[:2]
        right = sorted(docks, key=lambda dock: dock.geometry().x())[2:]
        self.assertLess(max(dock.geometry().x() for dock in left), min(dock.geometry().x() for dock in right))
        self.assert_near(left[0].width(), right[0].width())
        self.assert_near(docks[0].height(), docks[2].height())
        self.assert_near(docks[1].height(), docks[3].height())
        self.assertTrue(all(dock.isVisible() for dock in docks))

    def test_two_canvas_layout_switches_between_horizontal_and_vertical(self):
        first, second = self.add_canvases(2)
        self.assertGreater(second.geometry().x(), first.geometry().x())

        self.assertTrue(self.display.layout_manager.arrange("vertical"))
        self._settle()

        self.assertGreater(second.geometry().y(), first.geometry().y())
        self.assert_near(first.height(), second.height())
        self.assertEqual(self.display.layout_manager.two_canvas_orientation, Qt.Vertical)

    def test_canvas_refits_image_when_layout_reduces_viewport(self):
        first = self.add_canvases(1)[0]
        initial_scale = first.graphics_view.transform().m11()

        self.add_canvases(1)
        self._settle()

        resized_scale = first.graphics_view.transform().m11()
        self.assertLess(resized_scale, initial_scale)
        self.assertAlmostEqual(resized_scale, first.last_scale)

    def test_timeline_occupies_stable_space_before_playback(self):
        canvas = self.add_canvases(1)[0]
        root_layout = canvas.widget().layout()
        before_view = root_layout.itemAt(0).geometry()
        before_timeline = root_layout.itemAt(1).geometry()

        self.assertLessEqual(before_view.bottom(), before_timeline.top())
        try:
            canvas.start_auto_play()
            self.assertTrue(canvas.is_playing)
            self._settle()
        finally:
            canvas.pause_auto_play()

        after_view = root_layout.itemAt(0).geometry()
        after_timeline = root_layout.itemAt(1).geometry()
        self.assertLessEqual(after_view.bottom(), after_timeline.top())
        self.assertEqual(after_timeline, before_timeline)

    def test_three_canvas_layout_uses_one_full_height_and_two_stacked_docks(self):
        first, second, third = self.add_canvases(3)

        self.assertGreater(second.geometry().x(), first.geometry().x())
        self.assertEqual(second.geometry().x(), third.geometry().x())
        self.assertGreater(third.geometry().y(), second.geometry().y())
        self.assert_near(first.height(), second.height() + third.height(), tolerance=0.12)
        self.assert_near(second.height(), third.height())

    def test_focus_and_restore_keep_canvas_instances_and_interaction_state(self):
        docks = self.add_canvases(4)
        target = docks[2]
        target.draw_roi[1:4, 2:6] = 1
        target.current_time_idx = 3
        target.use_colormap = True
        target.graphics_view.scale(1.8, 1.8)
        transform = target.graphics_view.transform()
        before = list(self.display.display_canvas)
        self.display.set_cursor_id(target.id)

        self.assertTrue(self.display.layout_manager.focus_current())
        self._settle()

        self.assertTrue(target.isVisible())
        self.assertTrue(all(not dock.isVisible() for dock in docks if dock is not target))
        self.assertTrue(self.display.layout_manager.is_focused)

        self.assertTrue(self.display.layout_manager.restore_focus())
        self._settle()

        self.assertEqual(self.display.display_canvas, before)
        self.assertTrue(all(dock.isVisible() for dock in docks))
        self.assertEqual(target.current_time_idx, 3)
        self.assertTrue(target.use_colormap)
        self.assertTrue(np.all(target.draw_roi[1:4, 2:6] == 1))
        self.assertEqual(target.graphics_view.transform(), transform)

    def test_focus_restores_a_previously_floating_canvas(self):
        first, second = self.add_canvases(2)
        second.setFloating(True)
        second.resize(420, 300)
        self._settle()
        self.display.set_cursor_id(second.id)

        self.assertTrue(self.display.layout_manager.focus_current())
        self.assertFalse(second.isFloating())
        self.assertTrue(self.display.layout_manager.restore_focus())
        self._settle()

        self.assertTrue(second.isFloating())
        self.assertTrue(first.isVisible())
        self.assertTrue(second.isVisible())

    def test_layout_commands_do_not_replace_sources_or_request_new_frames(self):
        docks = self.add_canvases(4)
        sources = [dock.data.display_source for dock in docks]
        request_ids = [dock.render_controller.state.frame_render_request_id for dock in docks]

        for _ in range(3):
            self.assertTrue(self.display.layout_manager.arrange("quad"))
            self.assertTrue(self.display.layout_manager.arrange("auto"))
            self._settle()

        self.assertEqual([dock.data.display_source for dock in docks], sources)
        self.assertEqual(
            [dock.render_controller.state.frame_render_request_id for dock in docks],
            request_ids,
        )
        self.assertTrue(all(dock.features() & QDockWidget.DockWidgetMovable for dock in docks))
        self.assertTrue(all(dock.features() & QDockWidget.DockWidgetFloatable for dock in docks))

    def test_empty_state_and_layout_action_availability_follow_canvas_count(self):
        self.assertTrue(self.display.layout_manager.empty_widget.isVisible())
        self.display._update_layout_actions()
        self.assertFalse(self.display.auto_layout_action.isEnabled())

        self.add_canvases(2)
        self.display._update_layout_actions()

        self.assertFalse(self.display.layout_manager.empty_widget.isVisible())
        self.assertTrue(self.display.horizontal_layout_action.isEnabled())
        self.assertTrue(self.display.vertical_layout_action.isEnabled())
        self.assertFalse(self.display.quad_layout_action.isEnabled())


    def test_structure_changes_exit_focus_before_add_and_delete(self):
        first, second = self.add_canvases(2)
        self.display.set_cursor_id(second.id)
        self.assertTrue(self.display.layout_manager.focus_current())

        third = self.display.add_canvas(make_image("data-2", 2))
        self._settle()

        self.assertFalse(self.display.layout_manager.is_focused)
        self.assertEqual(len(self.display.display_canvas), 3)
        self.assertTrue(all(canvas.isVisible() for canvas in (first, second, third)))

        self.display.set_cursor_id(third.id)
        self.assertTrue(self.display.layout_manager.focus_current())
        self.assertTrue(self.display.del_canvas(first.id))
        self._settle()

        self.assertFalse(self.display.layout_manager.is_focused)
        self.assertEqual(len(self.display.display_canvas), 2)
        self.assertTrue(all(canvas.isVisible() for canvas in self.display.display_canvas))

    def test_reset_layout_collects_floating_canvases(self):
        first, second = self.add_canvases(2)
        second.setFloating(True)
        self._settle()
        self.assertTrue(second.isFloating())

        self.assertTrue(self.display.layout_manager.reset_layout())
        self._settle()

        self.assertFalse(first.isFloating())
        self.assertFalse(second.isFloating())
        self.assertGreater(second.geometry().x(), first.geometry().x())

    def test_single_canvas_does_not_enter_focus_mode(self):
        self.add_canvases(1)
        self.display._update_layout_actions()

        self.assertFalse(self.display.focus_layout_action.isEnabled())
        self.assertFalse(self.display.layout_manager.focus_current())
        self.assertFalse(self.display.layout_manager.is_focused)

    def test_invalid_global_colormap_boundaries_fall_back_to_data_range(self):
        self.display.tool_parameters.update({
            "use_colormap": True,
            "auto_boundary_set": False,
            "colormap": None,
            "min_value": "",
            "max_value": "",
        })
        self.display._normalize_colormap_parameters(self.display.tool_parameters)
        data = make_image("global-colormap")

        canvas = self.display.add_canvas(data)
        deadline = time.perf_counter() + 1.0
        while canvas.current_image is None and time.perf_counter() < deadline:
            self._settle()
            time.sleep(0.01)

        self.assertEqual(self.display.tool_parameters["colormap"], "Jet")
        self.assertIsNone(self.display.tool_parameters["min_value"])
        self.assertEqual(canvas.min_value, float(data.imagemin))
        self.assertEqual(canvas.max_value, float(data.imagemax))
        self.assertIsNotNone(canvas.current_image)
        self.assertIsNotNone(canvas.colorbar_item)
        self.assertNotEqual(canvas.render_status, "failed")
    def test_long_canvas_title_is_compact_and_full_name_is_in_tooltip(self):
        long_name = "measurement-" * 8
        canvas = self.display.add_canvas(make_image(long_name))

        self.assertLessEqual(len(canvas.windowTitle()), 38)
        self.assertIn(long_name, canvas.toolTip())
        self.assertEqual(canvas.objectName(), "canvas_slot_0")

class CanvasWorkspacePersistenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.settings = QSettings(str(Path(self.temp_dir.name) / "workspace.ini"), QSettings.IniFormat)
        self.settings.clear()
        self.host = None
        self.display = None
        self._create_display()

    def tearDown(self):
        self._destroy_display()
        self.settings.clear()
        self.settings.sync()
        self.temp_dir.cleanup()

    def _settle(self, rounds=8):
        for _ in range(rounds):
            self.app.processEvents(QEventLoop.AllEvents, 50)

    def _create_display(self):
        self.host = HostWindow(self.settings)
        self.display = ImageDisplayWindow(dict(TOOL_PARAMS), self.host)
        self.display.resize(1000, 720)
        self.display.show()
        self._settle()

    def _destroy_display(self):
        if self.display is None:
            return
        self.display.del_canvas(-1)
        deadline = time.perf_counter() + 2.0
        while self.display._retiring_canvases and time.perf_counter() < deadline:
            self._settle()
        self.display.close()
        self.host.close()
        self._settle()
        self.display = None
        self.host = None

    def _recreate_display(self):
        self._destroy_display()
        self._create_display()

    def _add_canvases(self, count):
        canvases = [self.display.add_canvas(make_image(f"saved-{index}", index)) for index in range(count)]
        self._settle()
        return canvases

    def test_matching_canvas_signature_restores_saved_orientation(self):
        self._add_canvases(2)
        self.assertTrue(self.display.layout_manager.arrange("vertical"))
        self._settle()
        self.assertTrue(self.display.layout_manager.flush_save())

        self._recreate_display()
        first, second = self._add_canvases(2)

        self.assertGreater(second.geometry().y(), first.geometry().y())
        self.assertEqual(self.display.layout_manager.two_canvas_orientation, Qt.Vertical)

    def test_unmatched_canvas_signature_uses_default_layout(self):
        self._add_canvases(2)
        self.display.layout_manager.arrange("vertical")
        self._settle()
        self.display.layout_manager.flush_save()

        self._recreate_display()
        first, second, third = self._add_canvases(3)

        self.assertGreater(second.geometry().x(), first.geometry().x())
        self.assertEqual(second.geometry().x(), third.geometry().x())
        self.assertGreater(third.geometry().y(), second.geometry().y())

    def test_corrupt_saved_state_falls_back_to_default_layout(self):
        signature = "canvas_slot_0+canvas_slot_1"
        prefix = f"canvas_workspace/layouts/{signature}"
        self.settings.setValue("canvas_workspace/schema_version", 1)
        self.settings.setValue(f"{prefix}/state", QByteArray(b"invalid-state"))
        self.settings.sync()

        first, second = self._add_canvases(2)

        self.assertGreater(second.geometry().x(), first.geometry().x())
        self.assertFalse(self.display.layout_manager.is_focused)

    def test_flush_during_focus_saves_pre_focus_visibility_and_layout(self):
        first, second = self._add_canvases(2)
        self.display.layout_manager.arrange("vertical")
        self._settle()
        self.display.set_cursor_id(second.id)
        self.assertTrue(self.display.layout_manager.focus_current())
        self.assertTrue(self.display.layout_manager.flush_save())

        self._recreate_display()
        restored_first, restored_second = self._add_canvases(2)

        self.assertTrue(restored_first.isVisible())
        self.assertTrue(restored_second.isVisible())
        self.assertGreater(restored_second.geometry().y(), restored_first.geometry().y())

    def test_reset_layout_clears_saved_preferences(self):
        first, second = self._add_canvases(2)
        self.display.layout_manager.arrange("vertical")
        self._settle()
        self.display.layout_manager.flush_save()
        self.assertIsNotNone(self.settings.value("canvas_workspace/schema_version"))

        self.assertTrue(self.display.layout_manager.reset_layout())
        self._settle()

        self.assertIsNone(self.settings.value("canvas_workspace/schema_version"))
        self.assertGreater(second.geometry().x(), first.geometry().x())

    def test_offscreen_floating_canvas_is_moved_to_an_available_screen(self):
        first, second = self._add_canvases(2)
        second.setFloating(True)
        second.setGeometry(-20000, -20000, 420, 300)
        self._settle()
        self.display.layout_manager.flush_save()

        self._recreate_display()
        self._add_canvases(2)
        self._settle()
        restored = self.display.display_canvas[1]
        available = [screen.availableGeometry() for screen in QApplication.screens()]

        self.assertTrue(restored.isFloating())
        self.assertTrue(any(screen.intersects(restored.frameGeometry()) for screen in available))

if __name__ == "__main__":
    unittest.main()
