import json
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

from ArrayCache import ArrayCacheConfig, ArrayRef, ArrayStore
from history.manifest import (
    HistoryManifestStore,
    build_manifest_item,
    cache_status_for_history_item,
)


class HistoryCacheManagerTests(unittest.TestCase):
    def setUp(self):
        self.cache_dir = Path(tempfile.mkdtemp())

    def test_manifest_round_trips_items_and_marks_missing_files(self):
        cache_file = self.cache_dir / "raw_data.npy"
        np.save(cache_file, np.zeros((2, 3), dtype=np.float32))
        ref = ArrayRef(
            path=cache_file,
            shape=(2, 3),
            dtype="float32",
            nbytes=24,
            created_at=1.0,
            field_name="data_origin",
        )
        item = types.SimpleNamespace(
            name="raw",
            timestamp=10.0,
            serial_number=1,
            datashape=(2, 3),
            datatype=np.dtype("float32"),
            _data_origin_storage=ref,
            parameters={"fps": 100},
        )
        store = HistoryManifestStore(self.cache_dir)

        stored = store.upsert(build_manifest_item(item, "Data"))
        loaded = store.load()

        self.assertEqual(loaded["items"][0]["id"], stored["id"])
        self.assertEqual(loaded["items"][0]["arrays"]["data_origin"]["path"], str(cache_file))
        self.assertTrue(store.validate_item_files(loaded["items"][0])["ok"])

        cache_file.unlink()

        self.assertFalse(store.validate_item_files(loaded["items"][0])["ok"])

    def test_cache_status_counts_array_refs_and_memory_arrays(self):
        ref = ArrayRef(
            path=self.cache_dir / "cached.npy",
            shape=(2, 3),
            dtype="float32",
            nbytes=24,
            created_at=1.0,
            field_name="data_origin",
        )
        item = types.SimpleNamespace(
            _data_origin_storage=ref,
            out_processed={"small": np.zeros((2,), dtype=np.float32)},
        )

        status = cache_status_for_history_item(item)

        self.assertEqual(status["cached_count"], 1)
        self.assertEqual(status["cached_bytes"], 24)
        self.assertGreaterEqual(status["memory_bytes"], 8)

    def test_manifest_file_is_json_and_uses_schema_version(self):
        store = HistoryManifestStore(self.cache_dir)
        store.save({"schema_version": 1, "items": []})

        content = json.loads((self.cache_dir / "history_manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(content["schema_version"], 1)
        self.assertEqual(content["items"], [])


class HistoryCacheArchitectureTests(unittest.TestCase):
    def test_mainwindow_and_history_controller_expose_unified_manager(self):
        main_source = (CORE / "MainWindow.py").read_text(encoding="utf-8")
        history_source = (CORE / "history" / "controller.py").read_text(encoding="utf-8")

        self.assertIn("history_cache_manager", main_source)
        self.assertIn("HistoryCacheManagerDialog", history_source)
        self.assertIn("force_cache_history_item", history_source)


if __name__ == "__main__":
    unittest.main()
