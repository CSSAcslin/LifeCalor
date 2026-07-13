import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from DataManager import Data, ImagingData
from display.playback_policy import timeline_label
from display.renderer import FrameRenderer
from importing import default_importer_registry


class ImporterRegistryTests(unittest.TestCase):
    def setUp(self):
        self.registry = default_importer_registry()

    def test_timeline_handles_missing_fps_and_prefers_real_axis(self):
        self.assertEqual(timeline_label(3, None, None), "3")
        self.assertEqual(timeline_label(1, 20, np.array([0.0, 0.25])), "0.25")

    def test_auto_import_reads_grayscale_tiff_stack_as_tyx(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stack.tif"
            source = np.arange(24, dtype=np.uint16).reshape(3, 2, 4)
            tifffile.imwrite(path, source, photometric="minisblack")

            probe = self.registry.probe("auto", path)
            payload = self.registry.read("auto", path, {"time_step": 0.5})

            self.assertEqual(probe.axes, "TYX")
            np.testing.assert_array_equal(payload.data, source)
            np.testing.assert_allclose(payload.time_point, [0.0, 0.5, 1.0])

    def test_palette_tiff_preserves_indices_and_builds_color_display(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "palette.tif"
            indices = np.array([[0, 1], [2, 3]], dtype=np.uint8)
            colormap = np.zeros((3, 256), dtype=np.uint16)
            colormap[0, :4] = [0, 65535, 0, 65535]
            colormap[1, :4] = [0, 0, 65535, 65535]
            colormap[2, :4] = [0, 0, 0, 65535]
            tifffile.imwrite(path, indices, photometric="palette", colormap=colormap)

            payload = self.registry.read("tiff", path, {"color_policy": "preserve"})

            np.testing.assert_array_equal(payload.data, indices)
            self.assertEqual(payload.display_data.shape, (2, 2, 3))
            self.assertEqual(payload.parameters["display_axes"], "YXC")
            self.assertTrue(payload.parameters["scientific_values_reconstructed"])

    def test_rgb_tiff_is_single_color_frame_with_explicit_luminance_analysis(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rgb.tif"
            rgb = np.zeros((3, 4, 3), dtype=np.uint8)
            rgb[..., 0] = 255
            tifffile.imwrite(path, rgb, photometric="rgb")

            payload = self.registry.read("tiff", path, {"color_policy": "preserve"})
            data = Data(payload.data, payload.time_point, payload.format_import, payload.display_data,
                        payload.parameters, payload.name)
            image = ImagingData.create_image(data)
            rendered = FrameRenderer.render(image.display_source.get_frame(0))

            self.assertEqual(payload.data.shape, (3, 4))
            self.assertEqual(image.totalframes, 1)
            self.assertEqual(image.framesize, (3, 4))
            self.assertEqual(rendered.image.shape, (3, 4, 4))
            self.assertFalse(payload.parameters["scientific_values_reconstructed"])
            self.assertIn("亮度", payload.parameters["import_warning"])

    def test_explicit_format_rejects_wrong_extension(self):
        with self.assertRaisesRegex(ValueError, "扩展名"):
            self.registry.probe("npy", "sample.tif")


    def test_mainwindow_exposes_generic_import_in_left_panel_only(self):
        source = (CORE / "MainWindow.py").read_text(encoding="utf-8")
        self.assertIn("通用文件导入", source)
        self.assertIn("self.general_file_btn.clicked.connect(self.load_general_file)", source)
        self.assertNotIn('addAction("导入 NPY 数据")', source)

    def test_timeline_setters_keep_optional_metadata_safe(self):
        source = (CORE / "widget" / "AdvancedTimeline.py").read_text(encoding="utf-8")
        self.assertIn("self.fps = normalized_fps(fps)", source)
        self.assertIn("self.time_point = valid_time_axis(time_point", source)
        self.assertNotIn("if self.fps > 0", source)


if __name__ == "__main__":
    unittest.main()
