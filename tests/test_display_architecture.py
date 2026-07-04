import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"


class DisplayArchitectureTests(unittest.TestCase):
    def test_imaging_create_image_does_not_copy_full_source_stack(self):
        source = (CORE / "DataManager.py").read_text(encoding="utf-8")
        create_image = source[source.index("def create_image"):source.index("    def apply_ROI", source.index("def create_image"))]
        self.assertNotIn("data_obj.data_origin.copy()", create_image)
        self.assertNotIn("data_obj.data_processed.copy()", create_image)
        self.assertIn("DisplaySourceFactory", source)

    def test_image_display_uses_frame_accessors_for_temporal_preview_data(self):
        source = (CORE / "ImageDisplayWindow.py").read_text(encoding="utf-8")
        self.assertIn("def frame_for_display", source)
        self.assertIn("def raw_frame", source)
        self.assertNotIn("self.data.image_data[0]", source)
        self.assertNotIn("self.data.image_data[idx]", source)

    def test_image_display_delegates_grayscale_frame_rendering_to_service(self):
        source = (CORE / "ImageDisplayWindow.py").read_text(encoding="utf-8")
        self.assertIn("FrameRenderService", source)
        self.assertIn("self.frame_render_service", source)
        self.assertIn("render_source", source)

    def test_image_display_uses_worker_and_stale_request_guard_for_async_frames(self):
        widget_source = (CORE / "ImageDisplayWindow.py").read_text(encoding="utf-8")
        source = (CORE / "display" / "render_controller.py").read_text(encoding="utf-8")
        self.assertIn("FrameRenderWorker", source)
        self.assertIn("frame_render_requested", source)
        self.assertIn("latest_frame_render_request_id", source)
        self.assertIn("def on_rendered", source)
        self.assertIn("self.state.is_stale(request_id)", source)
        self.assertIn("def on_frame_rendered", widget_source)

    def test_image_display_centralizes_qimage_creation(self):
        source = (CORE / "ImageDisplayWindow.py").read_text(encoding="utf-8")
        self.assertIn("def display_array_to_qimage", source)
        self.assertIn("np.ascontiguousarray", source)

    def test_image_display_routes_colormap_rendering_through_frame_service(self):
        source = (CORE / "ImageDisplayWindow.py").read_text(encoding="utf-8")
        self.assertIn("def render_params_for_display", source)
        self.assertIn("use_colormap=self.use_colormap", source)
        self.assertIn("colormap=self.colormap", source)
        self.assertIn("min_value=self.min_value", source)
        self.assertNotIn("color_map_manager.apply_colormap", source)

    def test_legacy_frame_path_applies_colormap_through_frame_renderer(self):
        source = (CORE / "ImageDisplayWindow.py").read_text(encoding="utf-8")
        frame_for_display = source[source.index("def frame_for_display"):source.index("def update_time_slice")]
        self.assertIn("FrameRenderer.render", frame_for_display)
        self.assertIn("self.raw_frame(idx)", frame_for_display)

    def test_image_display_schedules_initial_frame_without_mouse_movement(self):
        source = (CORE / "ImageDisplayWindow.py").read_text(encoding="utf-8")
        self.assertIn("def schedule_initial_display", source)
        self.assertIn("QTimer.singleShot(0, self._show_initial_frame)", source)
        mouse_move = source[source.index("def mouse_move_event"):source.index("def mouse_release_event")]
        self.assertNotIn("display_image()", mouse_move)

    def test_display_source_initial_frame_is_rendered_asynchronously(self):
        source = (CORE / "ImageDisplayWindow.py").read_text(encoding="utf-8")
        display_image = source[source.index("def display_image"):source.index("def update_display")]
        self.assertIn("self.request_frame_render(0)", display_image)
        self.assertIn("return", display_image)
        self.assertNotIn("self.frame_for_display(0)", display_image.split("self.request_frame_render(0)")[0])
        self.assertIn("def initialize_display_scene", source)
        update_display = source[source.index("def update_display"):source.index("def add_colorbar")]
        self.assertIn("self.initialize_display_scene(image_data)", update_display)

    def test_render_status_bar_only_reports_render_failures(self):
        source = (CORE / "MainWindow.py").read_text(encoding="utf-8")
        status_source = (CORE / "display" / "status.py").read_text(encoding="utf-8")
        handler = source[source.index("def handle_render_status"):source.index("def update_status")]
        self.assertIn("render_status_update(status, message)", handler)
        self.assertIn('status == "failed"', status_source)
        self.assertNotIn("'rendering': 'working'", handler)
        self.assertNotIn("'completed': 'idle'", handler)

    def test_canvas_signal_connect_does_not_disconnect_internal_render_worker(self):
        source = (CORE / "MainWindow.py").read_text(encoding="utf-8")
        binder_source = (CORE / "display" / "canvas_signals.py").read_text(encoding="utf-8")
        connect_block = source[source.index("def canvas_signal_connect"):source.index("def get_data_all")]
        self.assertNotIn("canvas.disconnect()", connect_block)
        self.assertIn("self.canvas_signal_binder.rebind_all()", connect_block)
        self.assertIn("disconnect_canvas_signal(signal, slot)", binder_source)

    def test_render_status_is_forwarded_to_main_status_bar(self):
        display_source = (CORE / "ImageDisplayWindow.py").read_text(encoding="utf-8")
        main_source = (CORE / "MainWindow.py").read_text(encoding="utf-8")
        self.assertIn("render_status_signal = pyqtSignal(str, str)", display_source)
        self.assertIn("def set_render_status", display_source)
        self.assertIn("render_status_signal.connect(self.handle_render_status)", main_source)
        self.assertIn("def handle_render_status", main_source)

    def test_canvas_removal_stops_render_worker_before_delete_later(self):
        source = (CORE / "ImageDisplayWindow.py").read_text(encoding="utf-8")
        remove_block = source[source.index("def _remove_single_canvas"):source.index("def del_canvas")]
        self.assertIn("prepare_for_removal", remove_block)
        self.assertLess(remove_block.index("prepare_for_removal"), remove_block.index("deleteLater"))

    def test_frame_render_requests_are_coalesced_to_latest_frame(self):
        source = (CORE / "display" / "render_controller.py").read_text(encoding="utf-8")
        self.assertIn("render_in_flight", source)
        self.assertIn("pending_frame_index", source)
        start = source.index("def request_frame_render")
        request_block = source[start:source.index("def _start_frame_render", start)]
        self.assertIn("state.start_or_queue(idx)", request_block)
        self.assertIn("return", request_block)
        self.assertIn("def start_pending_frame_render", source)

    def test_render_callbacks_ignore_closing_canvas(self):
        widget_source = (CORE / "ImageDisplayWindow.py").read_text(encoding="utf-8")
        source = (CORE / "display" / "render_controller.py").read_text(encoding="utf-8")
        rendered_block = source[source.index("def on_rendered"):source.index("def on_failed")]
        failed_start = source.index("def on_failed")
        failed_block = source[failed_start:]
        self.assertIn("_is_closing", rendered_block)
        self.assertIn("_is_closing", failed_block)
        self.assertIn("def prepare_for_removal", widget_source)

    def test_display_qimage_owns_its_buffer(self):
        source = (CORE / "ImageDisplayWindow.py").read_text(encoding="utf-8")
        qimage_block = source[source.index("def display_array_to_qimage"):source.index("def initialize_display_scene")]
        self.assertIn("qimage.copy()", qimage_block)

    def test_display_modules_live_in_display_package(self):
        display_dir = CORE / "display"
        self.assertTrue((display_dir / "__init__.py").exists())
        self.assertTrue((display_dir / "source.py").exists())
        self.assertTrue((display_dir / "renderer.py").exists())
        self.assertTrue((display_dir / "cache.py").exists())
        self.assertTrue((display_dir / "service.py").exists())
        self.assertTrue((display_dir / "worker.py").exists())

    def test_mainwindow_delegates_render_status_policy(self):
        source = (CORE / "MainWindow.py").read_text(encoding="utf-8")
        self.assertIn("from display.status import render_status_update", source)
        handler = source[source.index("def handle_render_status"):source.index("def update_status")]
        self.assertIn("render_status_update(status, message)", handler)
        self.assertNotIn("status == 'failed'", handler)

    def test_mainwindow_delegates_canvas_signal_wiring(self):
        source = (CORE / "MainWindow.py").read_text(encoding="utf-8")
        self.assertIn("from display.canvas_signals import CanvasSignalBinder", source)
        self.assertIn("self.canvas_signal_binder = CanvasSignalBinder(self)", source)
        block = source[source.index("def canvas_signal_connect"):source.index("def get_data_all")]
        self.assertIn("self.canvas_signal_binder.rebind_all()", block)
        self.assertNotIn("canvas.mouse_position_signal.connect", block)

    def test_mainwindow_delegates_canvas_creation_controller(self):
        source = (CORE / "MainWindow.py").read_text(encoding="utf-8")
        self.assertIn("from display.canvas_controller import DisplayCanvasController", source)
        self.assertIn("self.display_canvas_controller = DisplayCanvasController(self)", source)
        add_block = source[source.index("def add_new_canvas"):source.index("def other_imports")]
        self.assertIn("self.display_canvas_controller.add_new_canvas(assign_data)", add_block)
        self.assertIn("self.display_canvas_controller.load_image(data_type, other_params, origin_data)", add_block)
        self.assertNotIn("DataViewAndSelectPop", add_block)
        self.assertNotIn("ImagingData.create_image", add_block)
        upgrade_block = source[source.index("def upgrade_and_imaging"):source.index("'''其他功能'''")]
        self.assertIn("self.display_canvas_controller.upgrade_and_imaging", upgrade_block)

    def test_subimage_widget_delegates_render_lifecycle(self):
        source = (CORE / "ImageDisplayWindow.py").read_text(encoding="utf-8")
        self.assertIn("from display.render_controller import RenderController", source)
        self.assertIn("self.render_controller = RenderController(self)", source)
        self.assertIn("self.render_controller.request_frame_render(idx)", source)
        self.assertIn("self.render_controller.start_worker()", source)

    def test_mainwindow_display_methods_remain_thin(self):
        source = (CORE / "MainWindow.py").read_text(encoding="utf-8")
        add_block = source[source.index("def add_new_canvas"):source.index("def other_imports")]
        self.assertLess(len(add_block.splitlines()), 12)
        self.assertNotIn("DataViewAndSelectPop", add_block)
        self.assertNotIn("ImagingData.create_image", add_block)

    def test_image_display_render_methods_remain_wrappers(self):
        source = (CORE / "ImageDisplayWindow.py").read_text(encoding="utf-8")
        start = source.index("def request_frame_render")
        request_block = source[start:source.index("def _start_frame_render", start)]
        self.assertIn("self.render_controller.request_frame_render(idx)", request_block)
        self.assertLess(len(request_block.splitlines()), 8)


if __name__ == "__main__":
    unittest.main()
