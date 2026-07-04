import sys
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
    qtcore.QObject = object
    qtcore.QThread = object
    qtcore.pyqtSignal = lambda *args, **kwargs: None
    qtcore.pyqtSlot = lambda *args, **kwargs: (lambda func: func)
    pyqt5 = types.ModuleType("PyQt5")
    pyqt5.QtCore = qtcore
    sys.modules.setdefault("PyQt5", pyqt5)
    sys.modules.setdefault("PyQt5.QtCore", qtcore)


_install_missing_dependency_stubs()
from DataManager import prepare_still_frame, prepare_video_frames, video_fps_for_duration


class VideoExportPolicyTests(unittest.TestCase):
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
