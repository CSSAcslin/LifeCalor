import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from ArrayCache import (
    ArrayCacheConfig,
    ArrayRef,
    ArrayStore,
    array_nbytes,
    cache_large_arrays_in_mapping,
    resolve_array,
    should_cache_array,
)


class ArrayCacheTests(unittest.TestCase):
    def test_array_nbytes_respects_dtype(self):
        self.assertEqual(array_nbytes(np.zeros((4,), dtype=np.uint8)), 4)
        self.assertEqual(array_nbytes(np.zeros((4,), dtype=np.float32)), 16)
        self.assertEqual(array_nbytes(np.zeros((4,), dtype=np.complex128)), 64)

    def test_should_cache_array_uses_configurable_threshold(self):
        config = ArrayCacheConfig(cache_dir=Path(tempfile.mkdtemp()), threshold_bytes=16)
        self.assertFalse(should_cache_array(np.zeros((4,), dtype=np.float32), config))
        self.assertTrue(should_cache_array(np.zeros((5,), dtype=np.float32), config))

    def test_store_writes_npy_and_resolves_memmap(self):
        config = ArrayCacheConfig(cache_dir=Path(tempfile.mkdtemp()), threshold_bytes=1)
        store = ArrayStore(config)
        source = np.arange(12, dtype=np.float32).reshape(3, 4)

        ref = store.put_array(source, owner_id="abc", field_name="data_processed")
        loaded = resolve_array(ref, mmap_mode="r+")

        self.assertIsInstance(ref, ArrayRef)
        self.assertEqual(ref.shape, (3, 4))
        self.assertEqual(ref.dtype, "float32")
        self.assertEqual(ref.nbytes, source.nbytes)
        self.assertTrue(ref.path.exists())
        self.assertTrue(isinstance(loaded, np.memmap))
        np.testing.assert_array_equal(loaded, source)

    def test_store_reports_progress_for_cache_writes(self):
        events = []
        config = ArrayCacheConfig(cache_dir=Path(tempfile.mkdtemp()), threshold_bytes=1)
        store = ArrayStore(config, progress_callback=lambda current, total, message: events.append((current, total, message)))

        store.put_array(np.arange(4, dtype=np.float32), owner_id="abc", field_name="data_origin")

        self.assertEqual(events[0][0], 0)
        self.assertEqual(events[-1][0], events[-1][1])
        self.assertIn("缓存", events[-1][2])

    def test_cache_large_arrays_in_mapping_keeps_small_metadata(self):
        config = ArrayCacheConfig(cache_dir=Path(tempfile.mkdtemp()), threshold_bytes=8)
        store = ArrayStore(config)
        mapping = {
            "fps": 360,
            "whole_mean": np.arange(2, dtype=np.float32),
            "unfolded_data": np.arange(12, dtype=np.float32).reshape(3, 4),
        }

        cached = cache_large_arrays_in_mapping(mapping, store, owner_id="abc")

        self.assertEqual(cached["fps"], 360)
        self.assertIsInstance(cached["whole_mean"], np.ndarray)
        self.assertIsInstance(cached["unfolded_data"], ArrayRef)
        np.testing.assert_array_equal(resolve_array(cached["unfolded_data"]), mapping["unfolded_data"])

    def test_store_can_delete_refs_and_clear_all_cache_files(self):
        config = ArrayCacheConfig(cache_dir=Path(tempfile.mkdtemp()), threshold_bytes=1)
        store = ArrayStore(config)
        ref_one = store.put_array(np.arange(4, dtype=np.float32), owner_id="abc", field_name="one")
        ref_two = store.put_array(np.arange(4, dtype=np.float32), owner_id="abc", field_name="two")

        self.assertTrue(ref_one.path.exists())
        self.assertTrue(ref_two.path.exists())

        self.assertTrue(store.delete_ref(ref_one))
        self.assertFalse(ref_one.path.exists())
        deleted = store.clear_all()

        self.assertEqual(deleted, 1)
        self.assertFalse(ref_two.path.exists())

    def test_store_cleans_orphan_cache_files_without_active_refs(self):
        config = ArrayCacheConfig(cache_dir=Path(tempfile.mkdtemp()), threshold_bytes=1)
        store = ArrayStore(config)
        active = store.put_array(np.arange(4, dtype=np.float32), owner_id="abc", field_name="active")
        orphan = store.put_array(np.arange(4, dtype=np.float32), owner_id="abc", field_name="orphan")

        deleted = store.cleanup_orphans([active])

        self.assertEqual(deleted, 1)
        self.assertTrue(active.path.exists())
        self.assertFalse(orphan.path.exists())

    def test_store_logs_slow_cache_writes_with_array_metadata(self):
        config = ArrayCacheConfig(cache_dir=Path(tempfile.mkdtemp()), threshold_bytes=1, slow_write_seconds=0.1)
        store = ArrayStore(config)

        with patch("ArrayCache.time.perf_counter", side_effect=[10.0, 10.25]), self.assertLogs(level="WARNING") as logs:
            store.put_array(np.arange(4, dtype=np.complex64), owner_id="abc", field_name="complex_data")

        message = "\n".join(logs.output)
        self.assertIn("complex_data", message)
        self.assertIn("complex64", message)
        self.assertIn("shape=(4,)", message)

    def test_store_load_ref_reports_progress_and_materializes_array(self):
        events = []
        config = ArrayCacheConfig(cache_dir=Path(tempfile.mkdtemp()), threshold_bytes=1)
        store = ArrayStore(config, progress_callback=lambda current, total, message: events.append((current, total, message)))
        source = np.arange(12, dtype=np.float32).reshape(3, 4)
        ref = store.put_array(source, owner_id="abc", field_name="data_origin")
        events.clear()

        loaded = store.load_ref(ref, mmap_mode=None)

        self.assertIsInstance(loaded, np.ndarray)
        self.assertFalse(isinstance(loaded, np.memmap))
        np.testing.assert_array_equal(loaded, source)
        self.assertEqual(events[0][0], 0)
        self.assertEqual(events[-1][0], events[-1][1])
        self.assertIn("读取缓存", events[0][2])


if __name__ == "__main__":
    unittest.main()
