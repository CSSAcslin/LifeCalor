import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))


class FakeColumns:
    def __init__(self, values):
        self.values = list(values)

    def __ne__(self, other):
        return [value != other for value in self.values]

    def get_level_values(self, index):
        return FakeColumns([value[index] for value in self.values])


class FakeLoc:
    def __init__(self, frame):
        self.frame = frame

    def __getitem__(self, key):
        row_selector, column_mask = key
        return self.frame.select_columns(column_mask)


class FakeDataFrame:
    def __init__(self, columns):
        self.columns = FakeColumns(columns)
        self.loc = FakeLoc(self)
        self.selected_mask = None

    def select_columns(self, mask):
        selected = FakeDataFrame([column for column, keep in zip(self.columns.values, mask) if keep])
        selected.selected_mask = list(mask)
        return selected


class ExportPolicyTests(unittest.TestCase):
    def test_prepare_curve_export_drops_fit_curve_when_requested(self):
        from ExportPolicy import prepare_dataframe_for_export

        frame = FakeDataFrame(["time", "signal", "fit_curve"])
        prepared = prepare_dataframe_for_export(frame, current_mode="curve", include_fitting=False)

        self.assertEqual(prepared.columns.values, ["time", "signal"])

    def test_prepare_diff_export_drops_fit_curve_level_when_requested(self):
        from ExportPolicy import prepare_dataframe_for_export

        frame = FakeDataFrame([("x", "原始数据"), ("x", "拟合曲线")])
        prepared = prepare_dataframe_for_export(frame, current_mode="diff", include_fitting=False)

        self.assertEqual(prepared.columns.values, [("x", "原始数据")])

    def test_em_export_requires_frequency_processed_data(self):
        from ExportPolicy import can_export_em_data

        self.assertTrue(can_export_em_data("ROI_stft"))
        self.assertTrue(can_export_em_data("ROI_cwt"))
        self.assertFalse(can_export_em_data("lifetime_distribution"))
        self.assertFalse(can_export_em_data(None))


if __name__ == "__main__":
    unittest.main()
