import os
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))


class SafetyFoundationTests(unittest.TestCase):
    def test_mainwindow_does_not_embed_github_pat(self):
        mainwindow_source = (CORE / "MainWindow.py").read_text(encoding="utf-8")

        self.assertNotIn("ghp_", mainwindow_source)
        self.assertNotIn("Bearer ghp_", mainwindow_source)

    def test_github_auth_header_uses_environment_token_when_present(self):
        from AppConfig import get_github_auth_header

        previous = os.environ.get("LIFECALOR_GITHUB_TOKEN")
        try:
            os.environ["LIFECALOR_GITHUB_TOKEN"] = "example-token"

            self.assertEqual(get_github_auth_header(), "Bearer example-token")
        finally:
            if previous is None:
                os.environ.pop("LIFECALOR_GITHUB_TOKEN", None)
            else:
                os.environ["LIFECALOR_GITHUB_TOKEN"] = previous

    def test_github_auth_header_is_empty_without_environment_token(self):
        from AppConfig import get_github_auth_header

        previous = os.environ.pop("LIFECALOR_GITHUB_TOKEN", None)
        try:
            self.assertEqual(get_github_auth_header(), "")
        finally:
            if previous is not None:
                os.environ["LIFECALOR_GITHUB_TOKEN"] = previous


    def test_github_headers_omit_authorization_without_token(self):
        from AppConfig import build_github_headers

        previous = os.environ.pop("LIFECALOR_GITHUB_TOKEN", None)
        try:
            headers = build_github_headers({"Accept": "application/vnd.github+json"})

            self.assertEqual(headers["User-Agent"], "Carrier-Lifetime-Calculator")
            self.assertEqual(headers["Accept"], "application/vnd.github+json")
            self.assertNotIn("Authorization", headers)
        finally:
            if previous is not None:
                os.environ["LIFECALOR_GITHUB_TOKEN"] = previous
    def test_em_export_type_guard_only_accepts_roi_frequency_results(self):
        from AppConfig import is_em_frequency_result

        self.assertTrue(is_em_frequency_result("ROI_stft"))
        self.assertTrue(is_em_frequency_result("ROI_cwt"))
        self.assertFalse(is_em_frequency_result("lifetime_distribution"))
        self.assertFalse(is_em_frequency_result(""))
        self.assertFalse(is_em_frequency_result(None))


if __name__ == "__main__":
    unittest.main()

