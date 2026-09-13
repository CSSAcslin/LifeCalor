import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

from DataManager import Data
from ExtraDialog import DataTreeViewDialog, DataViewAndSelectPop
from history.annotations import assign_annotations, updated_annotations
from history.dialog import HistoryCacheManagerDialog
from widget.DataTagEditor import DataTagEditorDialog
from widget.EmojiPicker import EmojiPickerDialog, EmojiTagSlot
from widget.DataTreeWidget import DataHistoryTreeWidget


class HistoryMetadataUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.old_history = Data.history
        Data.history = self.old_history.__class__(maxlen=self.old_history.maxlen)

    def tearDown(self):
        Data.history = self.old_history

    @staticmethod
    def data(name="generated"):
        array = np.zeros((2, 3), dtype=np.float32)
        return Data(array, np.arange(2), "unit", array, name=name)

    def test_editor_keeps_generated_name_immutable(self):
        target = self.data()
        original_name = target.name
        dialog = DataTagEditorDialog(target)
        self.addCleanup(dialog.close)
        dialog.display_name.setText("显示别名")
        dialog.tag_edits[0].setText("✅")
        dialog._accept_changes()

        self.assertEqual(target.name, original_name)
        self.assertEqual(dialog.annotations()["display_name"], "显示别名")
        self.assertEqual(dialog.annotations()["tags"], ["✅"])

    def test_emoji_slot_uses_picker_and_displays_selected_icon(self):
        slot = EmojiTagSlot()
        self.addCleanup(slot.close)
        with patch.object(EmojiPickerDialog, "pick", return_value=("🔬", True)):
            slot.click()
        self.assertEqual(slot.text(), "🔬")
        self.assertEqual(slot.property("hasEmoji"), True)

    def test_shared_tree_filters_display_original_tag_and_category(self):
        target = self.data()
        assign_annotations(target, updated_annotations(None, display_name="Alias", tags=["✅"]))
        tree = DataHistoryTreeWidget()
        self.addCleanup(tree.close)
        tree.refresh_data([target], [])
        item = tree.topLevelItem(0)

        self.assertEqual(item.text(0), "Alias")
        self.assertEqual(item.text(1), "✅")
        tree.filter_entries(query="generated")
        self.assertFalse(item.isHidden())
        tree.filter_entries(query="missing")
        self.assertTrue(item.isHidden())

    def test_history_manager_combines_name_tag_and_type_filters(self):
        dialog = HistoryCacheManagerDialog(params={}, current_items=[{
            "kind": "Data", "name": "Alias", "original_name": "generated",
            "tags": ["✅"], "data_category": "图片", "timestamp": 1.0,
        }])
        self.addCleanup(dialog.close)
        item = dialog.current_tree.topLevelItem(0)

        dialog.current_search.setText("generated")
        self.assertFalse(item.isHidden())
        dialog.current_type_filter.setCurrentIndex(dialog.current_type_filter.findData("图片"))
        self.assertFalse(item.isHidden())
        dialog.current_search.setText("missing")
        self.assertTrue(item.isHidden())

    def test_history_manager_multi_select_counts_only_visible_items(self):
        dialog = HistoryCacheManagerDialog(params={}, current_items=[
            {
                "kind": "Data", "name": "One", "original_name": "one",
                "identity": ("Data", 1, 1.0), "timestamp": 1.0,
            },
            {
                "kind": "Data", "name": "Two", "original_name": "two",
                "identity": ("Data", 2, 2.0), "timestamp": 2.0,
            },
        ])
        self.addCleanup(dialog.close)
        dialog.current_search.setText("One")
        dialog._select_visible(dialog.current_tree)

        self.assertEqual(dialog._selected_current_identities(), [("Data", 1, 1.0)])
        self.assertIn("已选 1 项", dialog.current_count_label.text())
        self.assertIn("显示 1 项", dialog.current_count_label.text())
        self.assertTrue(dialog.edit_current_btn.isEnabled())

    def test_history_manager_checkbox_and_double_click_edit_use_identity(self):
        identity = ("Data", 7, 3.0)
        dialog = HistoryCacheManagerDialog(params={}, current_items=[{
            "kind": "Data", "name": "Alias", "original_name": "generated",
            "identity": identity, "timestamp": 3.0, "tags": ["✅"],
        }])
        self.addCleanup(dialog.close)
        item = dialog.current_tree.topLevelItem(0)
        item.setCheckState(0, Qt.Checked)
        emitted = []
        dialog.edit_current_requested.connect(emitted.append)
        dialog._current_item_double_clicked(item, 2)

        self.assertEqual(dialog._selected_current_identities(), [identity])
        self.assertEqual(emitted, [identity])

    def test_data_flow_tree_filters_by_alias_and_tag(self):
        target = self.data()
        assign_annotations(target, updated_annotations(None, display_name="Microscope", tags=["🔬"]))
        Data.history.clear()
        Data.history.append(target)
        dialog = DataTreeViewDialog()
        self.addCleanup(dialog.close)
        item = dialog.tree.topLevelItem(0)

        dialog.search_filter.setText("Microscope")
        self.assertFalse(item.isHidden())
        dialog.tag_filter._tag_actions["🔬"].setChecked(True)
        self.assertFalse(item.isHidden())
        dialog.search_filter.setText("missing")
        self.assertTrue(item.isHidden())

    def test_legacy_data_picker_filters_by_tag(self):
        dialog = DataViewAndSelectPop(datadict=[
            {"type": "Data", "name": "one", "标签": "✅", "timestamp": 1.0},
            {"type": "Data", "name": "two", "标签": "🔬", "timestamp": 2.0},
        ])
        self.addCleanup(dialog.close)
        table = dialog.tables[0]
        table._tag_filter._tag_actions["🔬"].setChecked(True)

        visible_names = [
            table.item(row, 1).text()
            for row in range(table.rowCount())
            if not table.isRowHidden(row)
        ]
        self.assertEqual(visible_names, ["two"])

    def test_legacy_history_table_selection_survives_sorting(self):
        dialog = DataViewAndSelectPop(datadict=[
            {"type": "Data", "name": "B", "timestamp": 2.0},
            {"type": "Data", "name": "A", "timestamp": 1.0},
        ])
        self.addCleanup(dialog.close)
        table = dialog.tables[0]
        table.sortItems(1, Qt.AscendingOrder)
        self.app.processEvents()
        first = table.item(0, 0)
        payload = first.data(Qt.UserRole)
        dialog.on_row_button_clicked(0, table)

        self.assertEqual(dialog.selected_timestamp, payload["timestamp"])


if __name__ == "__main__":
    unittest.main()
