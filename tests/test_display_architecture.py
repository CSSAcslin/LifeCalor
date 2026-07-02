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
        source = (CORE / "ImageDisplayWindow.py").read_text(encoding="utf-8")
        self.assertIn("FrameRenderWorker", source)
        self.assertIn("frame_render_requested", source)
        self.assertIn("_latest_frame_render_request_id", source)
        self.assertIn("def on_frame_rendered", source)
        self.assertIn("request_id != self._latest_frame_render_request_id", source)


if __name__ == "__main__":
    unittest.main()
