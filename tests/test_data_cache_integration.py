import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

import types


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
from ArrayCache import ArrayCacheConfig, ArrayRef
from DataManager import Data, ProcessedData, configure_array_cache, get_array_store, materialize_cached_arrays


class DataCacheIntegrationTests(unittest.TestCase):
    def setUp(self):
        Data.clear_history()
        ProcessedData.clear_history()
        self.cache_dir = Path(tempfile.mkdtemp())
        configure_array_cache(ArrayCacheConfig(cache_dir=self.cache_dir, threshold_bytes=8))

    def test_data_history_caches_large_data_origin_but_keeps_image_import_in_memory(self):
        source = np.arange(12, dtype=np.float32).reshape(3, 2, 2)
        image = source.mean(axis=0)
        data = Data(source, np.arange(3), "test", image)

        history_item = Data.history[-1]

        self.assertFalse(isinstance(data._data_origin_storage, ArrayRef))
        self.assertIsInstance(history_item._data_origin_storage, ArrayRef)
        self.assertIsInstance(history_item._image_import_storage, np.ndarray)
        np.testing.assert_array_equal(history_item.data_origin, source)
        np.testing.assert_array_equal(history_item.image_import, image)
        np.testing.assert_array_equal(data.data_origin, source)

    def test_processed_history_caches_large_data_processed_and_out_processed_arrays(self):
        source = np.arange(12, dtype=np.float32).reshape(3, 2, 2)
        processed = ProcessedData(
            1.0,
            "processed",
            "ROI_stft",
            time_point=np.arange(3),
            data_processed=source,
            out_processed={"fps": 360, "large_extra": source.copy()},
        )

        history_item = ProcessedData.history[-1]

        self.assertFalse(isinstance(processed._data_processed_storage, ArrayRef))
        self.assertIsInstance(history_item._data_processed_storage, ArrayRef)
        self.assertIsInstance(history_item.out_processed["large_extra"], ArrayRef)
        self.assertEqual(history_item.out_processed["fps"], 360)
        np.testing.assert_array_equal(history_item.data_processed, source)
        np.testing.assert_array_equal(history_item.out_processed_array("large_extra"), source)

    def test_history_lookup_returns_usable_cached_processed_data(self):
        source = np.arange(12, dtype=np.float32).reshape(3, 2, 2)
        processed = ProcessedData(1.0, "processed", "ROI_cwt", np.arange(3), source)

        selected = ProcessedData.find_history(processed.timestamp)

        self.assertIsNotNone(selected)
        np.testing.assert_array_equal(selected.data_processed, source)
        self.assertEqual(selected.framesize, (2, 2))

    def test_clear_history_removes_cached_npy_files(self):
        source = np.arange(12, dtype=np.float32).reshape(3, 2, 2)
        Data(source, np.arange(3), "test", source.mean(axis=0))
        ProcessedData(1.0, "processed", "ROI_stft", np.arange(3), source, out_processed={"large_extra": source.copy()})

        cached_files = list(self.cache_dir.glob("*.npy"))
        self.assertGreaterEqual(len(cached_files), 2)

        Data.clear_history(remove_cache=True)
        ProcessedData.clear_history(remove_cache=True)

        self.assertEqual(list(self.cache_dir.glob("*.npy")), [])

    def test_clear_all_cache_removes_files_even_after_history_is_gone(self):
        source = np.arange(12, dtype=np.float32).reshape(3, 2, 2)
        Data(source, np.arange(3), "test", source.mean(axis=0))
        Data.clear_history(remove_cache=False)

        self.assertGreaterEqual(len(list(self.cache_dir.glob("*.npy"))), 1)
        deleted = get_array_store().clear_all()

        self.assertGreaterEqual(deleted, 1)
        self.assertEqual(list(self.cache_dir.glob("*.npy")), [])

    def test_materialize_cached_arrays_replaces_refs_and_reports_progress(self):
        events = []
        source = np.arange(12, dtype=np.float32).reshape(3, 2, 2)
        Data(source, np.arange(3), "test", source.mean(axis=0))
        processed = ProcessedData(1.0, "processed", "ROI_stft", np.arange(3), source, out_processed={"large_extra": source.copy()})
        history_item = ProcessedData.history[-1]
        self.assertIsInstance(history_item._data_processed_storage, ArrayRef)
        self.assertIsInstance(history_item.out_processed["large_extra"], ArrayRef)

        materialize_cached_arrays(history_item, progress_callback=lambda current, total, message: events.append((current, total, message)))

        self.assertIsInstance(history_item._data_processed_storage, np.ndarray)
        self.assertIsInstance(history_item.out_processed["large_extra"], np.ndarray)
        np.testing.assert_array_equal(history_item.data_processed, source)
        np.testing.assert_array_equal(history_item.out_processed_array("large_extra"), source)
        self.assertTrue(any("读取缓存" in event[2] for event in events))


if __name__ == "__main__":
    unittest.main()

