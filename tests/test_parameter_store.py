import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))
import unittest


class ParameterStoreTests(unittest.TestCase):
    def test_load_group_preserves_default_types_and_string_fallbacks(self):
        from settings.parameter_store import load_param_group

        raw_settings = {
            "enabled": "true",
            "count": "5",
            "ratio": "2.5",
            "label": "",
        }
        defaults = {
            "enabled": False,
            "count": 1,
            "ratio": 1.0,
            "label": "fallback",
        }

        params = load_param_group(raw_settings.get, defaults)

        self.assertIs(params["enabled"], True)
        self.assertEqual(params["count"], 5)
        self.assertEqual(params["ratio"], 2.5)
        self.assertEqual(params["label"], "fallback")

    def test_load_group_uses_defaults_for_invalid_numeric_values(self):
        from settings.parameter_store import load_param_group

        params = load_param_group({"count": "bad", "ratio": None}.get, {"count": 3, "ratio": 1.5})

        self.assertEqual(params["count"], 3)
        self.assertEqual(params["ratio"], 1.5)


if __name__ == "__main__":
    unittest.main()

