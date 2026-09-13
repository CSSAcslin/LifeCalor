import json
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from DataManager import Data
from history.annotations import (
    assign_annotations,
    compact_display_name,
    display_name_for,
    history_identity,
    normalize_display_name,
    normalize_tags,
    same_history_identity,
    tags_for,
    updated_annotations,
    validate_emoji_tags,
)
from history.manifest import HistoryManifestStore, build_manifest_item, restore_history_item


class MetadataOnlyObject:
    name = "generated-name"
    serial_number = 7
    timestamp = 10.25
    annotations = None

    @property
    def data_origin(self):
        raise AssertionError("metadata access must not resolve arrays")


class HistoryAnnotationsTests(unittest.TestCase):
    def test_compact_display_name_keeps_head_and_tail(self):
        self.assertEqual(compact_display_name("1234567890", 10), "1234567890")
        self.assertEqual(compact_display_name("12345678901", 10), "1234...901")

    def test_display_name_is_separate_from_generated_original_name(self):
        target = MetadataOnlyObject()
        assign_annotations(target, updated_annotations(None, display_name="  Sample A  "))

        self.assertEqual(target.name, "generated-name")
        self.assertEqual(display_name_for(target), "Sample A")

        assign_annotations(target, updated_annotations(target.annotations, display_name=""))
        self.assertEqual(display_name_for(target), "generated-name")
        self.assertEqual(target.name, "generated-name")

    def test_display_name_rejects_control_characters_and_overflow(self):
        with self.assertRaises(ValueError):
            normalize_display_name("bad\nname")
        with self.assertRaises(ValueError):
            normalize_display_name("x" * 129)

    def test_tags_are_normalized_deduplicated_and_limited(self):
        family = "👨‍👩‍👧‍👦"
        self.assertEqual(normalize_tags([family, family, "✅"]), [family, "✅"])
        with self.assertRaises(ValueError):
            normalize_tags(["1", "2", "3", "4"])

    def test_emoji_validation_accepts_complete_sequences(self):
        tags = ["👨‍👩‍👧‍👦", "🇨🇳", "1️⃣"]
        self.assertEqual(validate_emoji_tags(tags), tags)
        with self.assertRaises(ValueError):
            validate_emoji_tags(["实验"])

    def test_child_tags_can_follow_parent_or_be_explicitly_empty(self):
        annotations = updated_annotations(None, tags=["✅"])
        path = ["out_processed", "spectrum.with.dot"]
        self.assertEqual(tags_for({"annotations": annotations}, path), ["✅"])

        annotations = updated_annotations(annotations, path=path, tags=[])
        self.assertEqual(tags_for({"annotations": annotations}, path), [])

        annotations = updated_annotations(annotations, path=path, follow_parent_tags=True)
        self.assertEqual(tags_for({"annotations": annotations}, path), ["✅"])
        self.assertEqual(display_name_for({"annotations": annotations}, path), "spectrum.with.dot")

    def test_identity_does_not_treat_same_serial_from_other_session_as_equal(self):
        first = SimpleNamespace(serial_number=3, timestamp=10.0)
        second = SimpleNamespace(serial_number=3, timestamp=20.0)

        self.assertNotEqual(history_identity(first, "Data"), history_identity(second, "Data"))
        self.assertFalse(same_history_identity(first, second, "Data"))

    def test_history_snapshot_does_not_share_mutable_annotations(self):
        old_history = Data.history
        Data.history = old_history.__class__(maxlen=old_history.maxlen)
        try:
            array = np.zeros((2, 3), dtype=np.float32)
            data = Data(array, np.arange(2), "unit", array, name="raw")
            data.annotations = updated_annotations(data.annotations, tags=["✅"])
            snapshot = data._history_snapshot()

            snapshot.annotations["tags"].append("⭐")
            self.assertEqual(data.annotations["tags"], ["✅"])
        finally:
            Data.history = old_history


class HistoryManifestAnnotationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.cache_dir = Path(self.temp_dir.name)
        self.store = HistoryManifestStore(self.cache_dir)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _item(self, item_id="Data:1:10.0", name="generated"):
        return {
            "id": item_id,
            "kind": "Data",
            "name": name,
            "serial_number": 1,
            "timestamp": 10.0,
            "shape": [2, 3],
            "dtype": "float32",
            "ndim": 2,
            "metadata": {"parameters": {}},
            "arrays": {},
        }

    def test_manifest_round_trip_preserves_annotations_and_source_binding(self):
        item = self._item()
        item["annotations"] = updated_annotations(None, display_name="Result", tags=["✅"])
        self.store.upsert(item)

        restored = restore_history_item(self.store.load()["items"][0], self.cache_dir)

        self.assertEqual(restored.name, "generated")
        self.assertEqual(display_name_for(restored), "Result")
        self.assertEqual(restored._history_manifest_dir, str(self.cache_dir))
        self.assertEqual(restored._history_manifest_id, item["id"])

    def test_old_manifest_defaults_to_original_name_and_empty_tags(self):
        self.store.upsert(self._item())
        restored = restore_history_item(self.store.load()["items"][0], self.cache_dir)

        self.assertEqual(display_name_for(restored), "generated")
        self.assertEqual(tags_for(restored), [])

    def test_metadata_update_preserves_original_name_saved_time_and_unknown_fields(self):
        item = self._item()
        item["saved_at"] = 123.0
        item["future_field"] = {"keep": True}
        self.store.upsert(item)

        updated = self.store.update_annotations(
            item["id"], updated_annotations(None, display_name="Alias")
        )

        self.assertEqual(updated["name"], "generated")
        self.assertEqual(updated["saved_at"], 123.0)
        self.assertEqual(updated["future_field"], {"keep": True})
        self.assertEqual(display_name_for(updated), "Alias")

    def test_atomic_replace_failure_keeps_previous_manifest(self):
        self.store.upsert(self._item())
        previous = self.store.path.read_bytes()

        with patch("history.manifest.os.replace", side_effect=OSError("denied")):
            with self.assertRaises(OSError):
                self.store.update_annotations(
                    "Data:1:10.0", updated_annotations(None, display_name="Alias")
                )

        self.assertEqual(self.store.path.read_bytes(), previous)
        self.assertFalse(list(self.cache_dir.glob("*.tmp")))

    def test_higher_schema_is_not_silently_downgraded(self):
        raw = {"schema_version": 99, "items": [self._item()], "future": True}
        self.store.path.write_text(json.dumps(raw), encoding="utf-8")

        self.assertEqual(self.store.load()["schema_version"], 99)
        with self.assertRaises(ValueError):
            self.store.update_annotations("Data:1:10.0", updated_annotations(None, display_name="Alias"))
        self.assertEqual(json.loads(self.store.path.read_text(encoding="utf-8")), raw)

    def test_concurrent_updates_of_different_items_do_not_lose_entries(self):
        self.store.upsert(self._item("Data:1:10.0", "one"))
        second = self._item("Data:2:20.0", "two")
        second["serial_number"] = 2
        second["timestamp"] = 20.0
        self.store.upsert(second)
        errors = []

        def update(item_id, alias):
            try:
                self.store.update_annotations(item_id, updated_annotations(None, display_name=alias))
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=update, args=("Data:1:10.0", "First")),
            threading.Thread(target=update, args=("Data:2:20.0", "Second")),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(errors, [])
        aliases = {item["id"]: display_name_for(item) for item in self.store.load()["items"]}
        self.assertEqual(aliases, {"Data:1:10.0": "First", "Data:2:20.0": "Second"})

    def test_build_manifest_item_keeps_generated_name_and_annotations_separate(self):
        target = SimpleNamespace(
            name="generated",
            serial_number=4,
            timestamp=5.0,
            annotations=updated_annotations(None, display_name="Alias"),
            parameters={},
            out_processed={},
            time_point=None,
            datashape=(),
            datatype="float32",
        )

        item = build_manifest_item(target, "Data")

        self.assertEqual(item["name"], "generated")
        self.assertEqual(item["annotations"]["display_name"], "Alias")


if __name__ == "__main__":
    unittest.main()
