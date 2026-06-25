import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

import SelectionPolicy as selection_policy


class SelectionPolicyTests(unittest.TestCase):
    def test_selects_current_data_when_data_requested(self):
        data = SimpleNamespace(name="raw", history=[])
        processed = SimpleNamespace(type_processed="filtered", history=[])

        result = selection_policy.select_data(
            mode=1,
            raw_data=data,
            processed_data=processed,
            aim_type="data",
            picker=lambda: None,
        )

        self.assertIs(result, data)

    def test_falls_back_to_processed_history_by_type(self):
        older = SimpleNamespace(type_processed="ROI_stft")
        current = SimpleNamespace(type_processed="other", history=[older])

        result = selection_policy.select_data(
            mode=1,
            raw_data=None,
            processed_data=current,
            aim_type=["ROI_stft", "ROI_cwt"],
            picker=lambda: None,
        )

        self.assertIs(result, older)

    def test_roi_rect_mask_uses_canvas_frame_size(self):
        canvas = SimpleNamespace(
            data=SimpleNamespace(framesize=(4, 5)),
            v_rect_roi=((1, 1), 2, 2),
        )

        mask = selection_policy.rect_mask_from_canvas(canvas)

        self.assertEqual(mask.shape, (4, 5))
        self.assertTrue(mask[1, 1])
        self.assertTrue(mask[2, 2])
        self.assertFalse(mask[0, 0])


if __name__ == "__main__":
    unittest.main()
