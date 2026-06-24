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
        "sif_parser",
        "cv2",
        "tifffile",
        "matplotlib",
        "matplotlib.cm",
        "matplotlib.colors",
        "matplotlib.pyplot",
        "PIL",
        "PIL.Image",
        "scipy",
        "scipy.ndimage",
        "scipy.ndimage.interpolation",
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

from DataManager import Data, ImagingData, ProcessedData

class DataModelTests(unittest.TestCase):
    def setUp(self):
        Data.clear_history()
        ProcessedData.clear_history()

    def test_data_records_metadata_and_history(self):
        raw = np.arange(12, dtype=np.float32).reshape(3, 2, 2)
        time_point = np.array([0.0, 0.5, 1.0])

        data = Data(
            data_origin=raw,
            time_point=time_point,
            format_import="unit",
            image_import=raw,
            parameters={"time_step": 0.5, "time_unit": "ms"},
            name="sample",
        )

        self.assertEqual(data.datashape, (3, 2, 2))
        self.assertEqual(data.timelength, 3)
        self.assertEqual(data.framesize, (2, 2))
        self.assertEqual(len(Data.history), 1)
        self.assertEqual(Data.find_history(data.timestamp).name, data.name)

    def test_data_time_trim_updates_shape_and_time_origin(self):
        raw = np.arange(20, dtype=np.float32).reshape(5, 2, 2)
        data = Data(
            data_origin=raw,
            time_point=np.arange(5, dtype=np.float32),
            format_import="unit",
            image_import=raw,
        )

        data.trim_time(1, 4)

        self.assertEqual(data.datashape, (3, 2, 2))
        np.testing.assert_array_equal(data.time_point, np.array([0.0, 1.0, 2.0], dtype=np.float32))
        self.assertIn("trimmed", data.name)

    def test_processed_data_upgrade_extracts_array_payload(self):
        source = ProcessedData(
            timestamp_inherited=123.0,
            name="processed",
            type_processed="stft_quality",
            time_point=np.array([0.0, 1.0]),
            data_processed=np.ones((2, 2), dtype=np.float32),
            out_processed={
                "fps": 360,
                "time_step": 1.0,
                "time_unit": "s",
                "target": np.arange(8, dtype=np.float32).reshape(2, 2, 2),
            },
        )

        extracted = source.upgrade_processed("target")

        self.assertIsInstance(extracted, ProcessedData)
        self.assertEqual(extracted.type_processed, "extracted_target")
        self.assertEqual(extracted.out_processed["fps"], 360)
        np.testing.assert_array_equal(extracted.data_processed, source.out_processed["target"])

    def test_imaging_data_create_image_normalizes_uint8_preview(self):
        raw = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dtype=np.float32)
        data = Data(
            data_origin=raw,
            time_point=np.array([0.0, 1.0]),
            format_import="unit",
            image_import=raw,
            parameters={"fps": 2},
            name="movie",
        )

        image = ImagingData.create_image(data)

        self.assertEqual(image.source_type, "Data")
        self.assertEqual(image.totalframes, 2)
        self.assertEqual(image.framesize, (2, 2))
        self.assertEqual(image.fps, 2)
        self.assertEqual(image.image_data.dtype, np.uint8)
        self.assertEqual(int(image.image_data.max()), 255)


if __name__ == "__main__":
    unittest.main()


