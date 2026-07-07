import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))


def _install_missing_dependency_stubs():
    optional_modules = [
        "sif_parser", "cv2", "tifffile", "matplotlib", "matplotlib.cm", "matplotlib.colors",
        "matplotlib.pyplot", "PIL", "PIL.Image", "scipy", "scipy.ndimage", "scipy.ndimage.interpolation",
    ]
    for name in optional_modules:
        sys.modules.setdefault(name, types.ModuleType(name))
    colors = sys.modules["matplotlib.colors"]
    if not hasattr(colors, "LinearSegmentedColormap"):
        colors.LinearSegmentedColormap = type("LinearSegmentedColormap", (), {})
    scipy_interp = sys.modules["scipy.ndimage.interpolation"]
    if not hasattr(scipy_interp, "zoom"):
        scipy_interp.zoom = lambda data, *args, **kwargs: data
    qtcore = types.ModuleType("PyQt5.QtCore")
    class _QObject:
        def __init__(self, *args, **kwargs):
            pass

    qtcore.QObject = _QObject
    qtcore.QThread = object
    qtcore.pyqtSignal = lambda *args, **kwargs: None
    qtcore.pyqtSlot = lambda *args, **kwargs: (lambda func: func)
    pyqt5 = types.ModuleType("PyQt5")
    pyqt5.QtCore = qtcore
    sys.modules.setdefault("PyQt5", pyqt5)
    sys.modules.setdefault("PyQt5.QtCore", qtcore)


_install_missing_dependency_stubs()
import DataManager as data_manager_module
from DataManager import (
    DataManager,
    export_array_from_data,
    export_as_temporal_images,
    prepare_still_frame,
    prepare_video_frames,
    video_fps_for_duration,
)


class VideoExportPolicyTests(unittest.TestCase):
    def test_temporal_canvas_export_uses_original_stack_not_preview_frame(self):
        source = types.SimpleNamespace(
            image_backup=np.zeros((5, 72, 72), dtype=np.uint16),
            image_data=np.zeros((72, 72), dtype=np.uint8),
        )

        result = export_array_from_data(source, "tif", is_temporal=True)

        self.assertEqual(result.shape, (5, 72, 72))
        self.assertIs(result, source.image_backup)

    def test_still_canvas_export_uses_display_frame(self):
        source = types.SimpleNamespace(
            image_backup=np.zeros((5, 72, 72), dtype=np.uint16),
            image_data=np.zeros((72, 72), dtype=np.uint8),
        )

        result = export_array_from_data(source, "png", is_temporal=False)

        self.assertEqual(result.shape, (72, 72))
        self.assertIs(result, source.image_data)

    def test_two_dimensional_preview_is_not_iterated_as_temporal_stack(self):
        image = np.zeros((72, 72), dtype=np.uint8)

        self.assertFalse(export_as_temporal_images(image, is_temporal=True))

    def test_tiff_export_treats_two_dimensional_preview_as_single_image(self):
        manager = DataManager.__new__(DataManager)
        manager.data_progress_signal = types.SimpleNamespace(emit=lambda *args: None)
        image = np.zeros((4, 5), dtype=np.uint8)
        writes = []
        old_imwrite = getattr(data_manager_module.tiff, "imwrite", None)
        data_manager_module.tiff.imwrite = lambda path, frame, photometric=None: writes.append((path, frame.shape, photometric))
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                files = manager.export_as_tif(image, tmpdir, "preview", is_temporal=True)
        finally:
            if old_imwrite is None:
                delattr(data_manager_module.tiff, "imwrite")
            else:
                data_manager_module.tiff.imwrite = old_imwrite

        self.assertEqual(len(files), 1)
        self.assertEqual(len(writes), 1)
        self.assertEqual(writes[0][1], (4, 5))
        self.assertEqual(writes[0][2], "minisblack")

    def test_two_dimensional_image_becomes_single_frame_video(self):
        image = np.arange(12, dtype=np.float32).reshape(3, 4)

        frames, is_color = prepare_video_frames(image)

        self.assertEqual(frames.shape, (1, 3, 4))
        self.assertFalse(is_color)

    def test_temporal_grayscale_stack_is_preserved(self):
        stack = np.arange(24, dtype=np.float32).reshape(2, 3, 4)

        frames, is_color = prepare_video_frames(stack)

        self.assertEqual(frames.shape, (2, 3, 4))
        self.assertFalse(is_color)

    def test_rgba_stack_is_trimmed_to_rgb_color_video(self):
        stack = np.zeros((2, 3, 4, 4), dtype=np.uint8)

        frames, is_color = prepare_video_frames(stack)

        self.assertEqual(frames.shape, (2, 3, 4, 3))
        self.assertTrue(is_color)

    def test_one_dimensional_data_is_rejected_with_clear_error(self):
        with self.assertRaisesRegex(ValueError, "图像序列"):
            prepare_video_frames(np.arange(4, dtype=np.float32))

    def test_single_channel_frame_is_written_as_grayscale(self):
        frame = np.zeros((3, 4, 1), dtype=np.uint8)

        prepared, photometric = prepare_still_frame(frame)

        self.assertEqual(prepared.shape, (3, 4))
        self.assertEqual(photometric, "minisblack")

    def test_rgb_frame_keeps_rgb_photometric(self):
        frame = np.zeros((3, 4, 3), dtype=np.uint8)

        prepared, photometric = prepare_still_frame(frame)

        self.assertEqual(prepared.shape, (3, 4, 3))
        self.assertEqual(photometric, "rgb")

    def test_video_fps_never_drops_below_one(self):
        self.assertEqual(video_fps_for_duration(10, 60), 1)
        self.assertEqual(video_fps_for_duration(120, 60), 2)
        self.assertEqual(video_fps_for_duration(10, 0), 10)


if __name__ == "__main__":
    unittest.main()
