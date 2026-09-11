import os
import sys
import time
import unittest
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from PyQt5.QtCore import QEvent, QPointF, QTimer, Qt
from PyQt5.QtGui import QMouseEvent
from PyQt5.QtWidgets import QApplication, QMainWindow

from DataManager import Data, ImagingData
from ImageDisplayWindow import ImageDisplayWindow
from display.canvas_controller import DisplayCanvasController
from display.renderer import RenderedFrame


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
    def __init__(self):
        super().__init__()
        self.focus_canvas = None
        self.rebind_count = 0

    def canvas_signal_connect(self):
        self.rebind_count += 1


def make_data(name, value=0):
    raw = np.full((3, 6, 8), value, dtype=np.float32)
    raw[:, 2:4, 3:5] += 1
    return Data(
        data_origin=raw,
        time_point=np.arange(3, dtype=np.float64),
        format_import="unit",
        image_import=raw,
        parameters={"fps": 20},
        name=name,
    )


def make_image(name, value=0):
    return ImagingData.create_image(make_data(name, value))

class CanvasLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.host = HostWindow()
        self.display = ImageDisplayWindow(dict(TOOL_PARAMS), self.host)
        self.display.resize(800, 600)
        self.display.show()
        self.app.processEvents()

    def tearDown(self):
        self.display.del_canvas(-1)
        self._wait_until(lambda: not self.display._retiring_canvases, timeout=2.0)
        self.display.close()
        self.host.close()
        self.app.processEvents()

    def _wait_until(self, predicate, timeout=1.0):
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            self.app.processEvents()
            if predicate():
                return True
            time.sleep(0.005)
        self.app.processEvents()
        return bool(predicate())

    def test_middle_delete_reindexes_compatibility_id_but_keeps_layout_identity(self):
        first = self.display.add_canvas(make_image("first"))
        middle = self.display.add_canvas(make_image("middle", 1))
        last = self.display.add_canvas(make_image("last", 2))
        self.display.set_cursor_id(last.id)

        self.assertTrue(self.display.del_canvas(middle.id))

        self.assertEqual(self.display.display_canvas, [first, last])
        self.assertEqual([canvas.id for canvas in self.display.display_canvas], [0, 1])
        self.assertEqual([canvas.data.canvas_num for canvas in self.display.display_canvas], [0, 1])
        self.assertEqual(first.layout_key, "canvas_slot_0")
        self.assertEqual(last.layout_key, "canvas_slot_2")
        self.assertEqual(last.windowTitle(), f"1-{last.data.source_name}")
        self.assertEqual(self.display.cursor_id, 1)
        self.assertEqual(self.host.focus_canvas, 1)
        self.assertTrue(bool(last.property("canvasFocused")))

        replacement_slot = self.display.add_canvas(make_image("new", 3))
        self.assertEqual(replacement_slot.layout_key, "canvas_slot_1")

    def test_zero_canvas_cursor_and_roi_are_safe(self):
        self.display.cursor()
        draw_layer, mask = self.display.get_draw_roi()
        self.assertIsNone(draw_layer)
        self.assertIsNone(mask)
        self.assertEqual(self.display.cursor_id, -1)
        self.assertIsNone(self.host.focus_canvas)

    def test_title_and_content_clicks_select_the_same_canvas(self):
        first = self.display.add_canvas(make_image("click-first"))
        second = self.display.add_canvas(make_image("click-second"))
        first.current_canvas_signal.connect(self.display.set_cursor_id)
        second.current_canvas_signal.connect(self.display.set_cursor_id)
        self.assertEqual(self.display.cursor_id, second.id)

        event = QMouseEvent(
            QEvent.MouseButtonPress,
            QPointF(4, 4),
            Qt.LeftButton,
            Qt.LeftButton,
            Qt.NoModifier,
        )
        first.mousePressEvent(event)
        self.assertEqual(self.display.current_canvas(), first)
        self.assertTrue(bool(first.property("canvasFocused")))
        self.assertFalse(bool(second.property("canvasFocused")))

        second.mouse_press_event(event)
        self.assertEqual(self.display.current_canvas(), second)
        self.assertTrue(bool(second.property("canvasFocused")))

    def test_titlebar_close_deletes_canvas_and_moves_focus(self):
        first = self.display.add_canvas(make_image("visible"))
        current = self.display.add_canvas(make_image("to-delete"))
        current.show()
        self.app.processEvents()

        current.close()

        self.assertTrue(self._wait_until(lambda: current not in self.display.display_canvas))
        self.assertIs(self.display.current_canvas(), first)
        self.assertEqual(self.host.focus_canvas, first.id)
        self.assertTrue(bool(first.property("canvasFocused")))
        self.assertTrue(current._is_closing)
    def test_titlebar_close_frees_one_of_four_canvas_slots(self):
        canvases = [
            self.display.add_canvas(make_image(f"canvas-{index}", index))
            for index in range(4)
        ]
        closing = canvases[1]
        closing.show()
        self.app.processEvents()

        closing.close()

        self.assertTrue(self._wait_until(lambda: closing not in self.display.display_canvas))
        self.assertEqual(len(self.display.display_canvas), 3)
        replacement = self.display.add_canvas(make_image("replacement", 9))
        self.assertIsNot(replacement, False)
        self.assertEqual(len(self.display.display_canvas), 4)
        self.assertEqual(replacement.layout_key, "canvas_slot_1")
        self.assertTrue(closing._is_closing)
    def test_import_overwrite_targets_only_the_focused_canvas(self):
        target = self.display.add_canvas(make_image("target"))
        untouched = self.display.add_canvas(make_image("untouched-import", 2))
        self.display.set_cursor_id(target.id)

        class LimitInput:
            def setMaximum(self, value):
                self.maximum = value

        self.host.image_display = self.display
        self.host.tool_params = dict(TOOL_PARAMS)
        self.host.data = make_data("incoming", 5)
        self.host.processed_data = None
        self.host.region_x_input = LimitInput()
        self.host.region_y_input = LimitInput()
        controller = DisplayCanvasController(self.host)
        controller._ask_canvas_action = lambda current: "overwrite"

        self.assertTrue(controller.load_image(origin_data=self.host.data))

        self.assertEqual(len(self.display.display_canvas), 2)
        self.assertIs(self.display.display_canvas[1], untouched)
        current = self.display.current_canvas()
        self.assertIsNot(current, target)
        self.assertIs(current.data.parent_data(), self.host.data)
        self.assertTrue(target._is_closing)

    def test_replace_changes_only_target_and_resets_canvas_interaction_state(self):
        untouched = self.display.add_canvas(make_image("untouched"))
        old = self.display.add_canvas(make_image("old", 1))
        old.draw_roi[1:3, 2:4] = 1
        old.current_time_idx = 2
        old_key = old.layout_key
        self.display.set_cursor_id(old.id)

        new = self.display.replace_canvas(old.id, make_image("replacement", 4))

        self.assertIs(self.display.display_canvas[0], untouched)
        self.assertIs(self.display.display_canvas[1], new)
        self.assertEqual(new.layout_key, old_key)
        self.assertEqual(new.windowTitle(), f"1-{new.data.source_name}")
        self.assertEqual(new.current_time_idx, 0)
        self.assertFalse(np.any(new.draw_roi))
        self.assertEqual(self.display.cursor_id, 1)
        self.assertTrue(old._is_closing)

    def test_slow_render_retirement_returns_immediately_and_gui_keeps_ticking(self):
        canvas = self.display.add_canvas(make_image("slow"))
        self.assertTrue(
            self._wait_until(lambda: not canvas.render_controller.state.render_in_flight)
        )

        def slow_render(source, frame_index, params):
            time.sleep(0.2)
            return RenderedFrame(np.zeros((6, 8), dtype=np.uint8), "L")

        canvas.frame_render_worker.service.render_source = slow_render
        canvas.request_frame_render(1)
        self.app.processEvents()
        time.sleep(0.02)

        ticks = []
        timer = QTimer()
        timer.setInterval(10)
        timer.timeout.connect(lambda: ticks.append(time.perf_counter()))
        timer.start()

        thread = canvas.frame_render_thread
        started = time.perf_counter()
        self.assertTrue(self.display.del_canvas(canvas.id))
        elapsed = time.perf_counter() - started

        self.assertLess(elapsed, 0.1)
        self.assertTrue(self._wait_until(lambda: not thread.isRunning(), timeout=1.0))
        timer.stop()
        self.assertGreater(len(ticks), 0)
        self.assertNotIn(canvas, self.display.display_canvas)


if __name__ == "__main__":
    unittest.main()
