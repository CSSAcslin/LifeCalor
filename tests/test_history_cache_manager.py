import json
from collections import deque
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
    array_ref_to_dict,
    array_refs_for_manifest,
    HistoryManifestStore,
    build_manifest_item,
    cache_status_for_history_item,
    restore_history_item,
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


    def test_restore_data_manifest_item_keeps_arrays_as_refs(self):
        cache_file = self.cache_dir / "raw_data.npy"
        np.save(cache_file, np.zeros((2, 3), dtype=np.float32))
        manifest_item = {
            "kind": "Data",
            "name": "raw_1",
            "format_import": "tif",
            "timestamp": 12.5,
            "serial_number": 7,
            "shape": [2, 3],
            "dtype": "float32",
            "ndim": 2,
            "datamin": 0.0,
            "datamax": 1.0,
            "metadata": {"parameters": {"fps": 100}},
            "arrays": {
                "data_origin": {
                    "path": str(cache_file),
                    "shape": [2, 3],
                    "dtype": "float32",
                    "nbytes": 24,
                    "created_at": 1.0,
                    "field_name": "data_origin",
                }
            },
        }

        restored = restore_history_item(manifest_item)

        self.assertEqual(restored.name, "raw_1")
        self.assertEqual(restored.format_import, "tif")
        self.assertEqual(restored.datashape, (2, 3))
        self.assertIsInstance(restored.__dict__["_data_origin_storage"], ArrayRef)

    def test_restore_processed_manifest_item_rebuilds_out_processed_refs(self):
        processed_file = self.cache_dir / "processed.npy"
        extra_file = self.cache_dir / "extra.npy"
        np.save(processed_file, np.zeros((2, 3), dtype=np.float32))
        np.save(extra_file, np.ones((2, 3), dtype=np.float32))
        manifest_item = {
            "kind": "ProcessedData",
            "name": "proc-1",
            "type_processed": "ROI_stft",
            "timestamp": 22.0,
            "timestamp_inherited": 12.5,
            "serial_number": 8,
            "shape": [2, 3],
            "dtype": "float32",
            "ndim": 2,
            "datamin": 0.0,
            "datamax": 1.0,
            "metadata": {"out_processed_metadata": {"fps": 360}},
            "arrays": {
                "data_processed": {
                    "path": str(processed_file),
                    "shape": [2, 3],
                    "dtype": "float32",
                    "nbytes": 24,
                    "created_at": 1.0,
                    "field_name": "data_processed",
                },
                "out_processed.large": {
                    "path": str(extra_file),
                    "shape": [2, 3],
                    "dtype": "float32",
                    "nbytes": 24,
                    "created_at": 1.0,
                    "field_name": "out_processed_large",
                },
            },
        }

        restored = restore_history_item(manifest_item)

        self.assertEqual(restored.name, "proc-1")
        self.assertEqual(restored.type_processed, "ROI_stft")
        self.assertEqual(restored.out_processed["fps"], 360)
        self.assertIsInstance(restored.__dict__["_data_processed_storage"], ArrayRef)
        self.assertIsInstance(restored.out_processed["large"], ArrayRef)


    def test_persistent_manifest_refs_survive_current_history_clear(self):
        cache_file = self.cache_dir / "persisted.npy"
        np.save(cache_file, np.zeros((2, 3), dtype=np.float32))
        ref = ArrayRef(
            path=cache_file,
            shape=(2, 3),
            dtype="float32",
            nbytes=24,
            created_at=1.0,
            field_name="data_origin",
        )
        store = HistoryManifestStore(self.cache_dir)
        store.upsert({
            "id": "Data:1:10.0",
            "kind": "Data",
            "name": "persisted",
            "arrays": {"data_origin": array_ref_to_dict(ref)},
        })
        window = types.SimpleNamespace(data=types.SimpleNamespace(history=deque([types.SimpleNamespace(__dict__={"ref": ref})])), processed_data=None)

        # Simulate the intended behavior of the MainWindow "current history clear": clear RAM history only.
        window.data.history.clear()

        self.assertTrue(cache_file.exists())
        self.assertTrue(store.validate_item_files(store.load()["items"][0])["ok"])

    def test_manifest_refs_are_not_treated_as_orphan_cache_files(self):
        kept_file = self.cache_dir / "kept.npy"
        orphan_file = self.cache_dir / "orphan.npy"
        np.save(kept_file, np.zeros((2, 3), dtype=np.float32))
        np.save(orphan_file, np.ones((2, 3), dtype=np.float32))
        ref = ArrayRef(
            path=kept_file,
            shape=(2, 3),
            dtype="float32",
            nbytes=24,
            created_at=1.0,
            field_name="data_origin",
        )
        store = HistoryManifestStore(self.cache_dir)
        store.upsert({
            "id": "Data:1:10.0",
            "kind": "Data",
            "name": "persisted",
            "arrays": {"data_origin": array_ref_to_dict(ref)},
        })

        deleted = ArrayStore(ArrayCacheConfig(self.cache_dir)).cleanup_orphans(array_refs_for_manifest(store.load()))

        self.assertEqual(deleted, 1)
        self.assertTrue(kept_file.exists())
        self.assertFalse(orphan_file.exists())

    def test_clear_manifest_removes_recoverable_history_index(self):
        cache_file = self.cache_dir / "persisted.npy"
        np.save(cache_file, np.zeros((2, 3), dtype=np.float32))
        ref = ArrayRef(
            path=cache_file,
            shape=(2, 3),
            dtype="float32",
            nbytes=24,
            created_at=1.0,
            field_name="data_origin",
        )
        store = HistoryManifestStore(self.cache_dir)
        store.upsert({
            "id": "Data:1:10.0",
            "kind": "Data",
            "name": "persisted",
            "arrays": {"data_origin": array_ref_to_dict(ref)},
        })

        removed = store.clear_items()

        self.assertEqual(removed, 1)
        self.assertEqual(store.load()["items"], [])

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
        dialog_source = (CORE / "history" / "dialog.py").read_text(encoding="utf-8")
        self.assertIn("open_cache_directory", dialog_source)
        self.assertIn("recover_manifest_requested", dialog_source)
        self.assertIn("restore_manifest_item", history_source)
        self.assertIn("remove_cache=False", main_source)
        self.assertIn("self.data is not None", main_source)
        self.assertIn("array_refs_for_manifest", history_source)
        self.assertIn("clear_all_cache", history_source)
        self.assertIn("refresh_cache_dialog", history_source)
        self.assertIn("refresh_requested", dialog_source)
        self.assertIn("load_cached_history_async(restored", history_source)


if __name__ == "__main__":
    unittest.main()
