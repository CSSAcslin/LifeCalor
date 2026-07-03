import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from selection import SelectionController


class FakeDialog:
    def __init__(self, selected_timestamp=None, selected_table="data", accepted=True):
        self.selected_timestamp = selected_timestamp
        self.selected_table = selected_table
        self.accepted = accepted

    def exec_(self):
        return self.accepted

    def get_selected_timestamp(self):
        return self.selected_timestamp, self.selected_table


class FakeWindow:
    def __init__(self):
        self.mode = 1
        self.data = SimpleNamespace(name="raw-current", timestamp=1.0, history=[])
        self.processed_data = SimpleNamespace(name="processed-current", type_processed="filtered", timestamp=2.0, history=[])
        self.warnings = []

    def get_data_all(self):
        return []

    def get_processed_data_all(self):
        return []


class SelectionControllerTests(unittest.TestCase):
    def test_select_data_delegates_default_mode_to_current_data(self):
        window = FakeWindow()
        controller = SelectionController(window, warning=lambda *args: window.warnings.append(args))

        result = controller.select_data("data")

        self.assertIs(result, window.data)

    def test_pick_data_returns_selected_history_item(self):
        window = FakeWindow()
        older = SimpleNamespace(name="older", timestamp=10.0)
        window.data.history = [older]
        controller = SelectionController(
            window,
            data_dialog_factory=lambda **kwargs: FakeDialog(10.0, "data"),
            warning=lambda *args: window.warnings.append(args),
        )

        result = controller.pick_data()

        self.assertIs(result, older)

    def test_pick_data_warns_when_selection_cannot_be_resolved(self):
        window = FakeWindow()
        controller = SelectionController(
            window,
            data_dialog_factory=lambda **kwargs: FakeDialog(99.0, "data"),
            warning=lambda *args: window.warnings.append(args),
        )

        result = controller.pick_data()

        self.assertIsNone(result)
        self.assertEqual(window.warnings[-1][1], "没有选取数据")


if __name__ == "__main__":
    unittest.main()
