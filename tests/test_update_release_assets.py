import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
CORE = Path(__file__).resolve().parents[1] / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from UpdateModule import UpdateChecker, current_application_path, update_restart_path


class UpdateReleaseAssetTests(unittest.TestCase):
    @staticmethod
    def asset(name):
        return {"name": name, "browser_download_url": f"https://example.test/{name}"}

    def select(self, executable, assets):
        checker = UpdateChecker("owner", "repo", "1.1.1", "")
        with patch.object(sys, "frozen", True, create=True), patch.object(sys, "executable", executable):
            return checker.find_download_asset(assets)

    def test_each_build_selects_its_asset_regardless_of_order_and_old_version_suffix(self):
        full = self.asset("LifeCalor-v1.1.2.exe")
        cpu = self.asset("LifeCalor-CPU-v1.1.2.exe")
        for name, expected in (("LifeCalor", full), ("LifeCalor-CPU", cpu)):
            for old_suffix in ("", "-v1.1.1"):
                for assets in ([full, cpu], [cpu, full]):
                    with self.subTest(name=name, old_suffix=old_suffix, order=assets):
                        actual = self.select(str(Path("application") / f"{name}{old_suffix}.exe"), assets)
                        self.assertEqual(actual, {"name": expected["name"], "url": expected["browser_download_url"]})

    def test_cpu_build_does_not_fall_back_to_full_distribution(self):
        self.assertIsNone(self.select("LifeCalor-CPU.exe", [self.asset("LifeCalor-v1.1.2.exe")]))

    def test_full_build_does_not_fall_back_to_cpu_distribution(self):
        self.assertIsNone(self.select("LifeCalor.exe", [self.asset("LifeCalor-CPU-v1.1.2.exe")]))

    def test_zip_assets_keep_the_build_variant(self):
        full = self.asset("LifeCalor-v1.1.2.zip")
        cpu = self.asset("LifeCalor-CPU-v1.1.2.zip")
        self.assertEqual(self.select("LifeCalor-CPU.exe", [full, cpu])["name"], cpu["name"])
        self.assertEqual(self.select("LifeCalor.exe", [cpu, full])["name"], full["name"])

    def test_irrelevant_assets_and_missing_urls_are_not_updates(self):
        assets = [self.asset("SHA256SUMS.txt"), self.asset("unrelated.exe"), {"name": "LifeCalor.exe"}]
        self.assertIsNone(self.select("LifeCalor.exe", assets))
        self.assertIsNone(self.select("LifeCalor.exe", []))

    def test_case_insensitive_cpu_asset_matching(self):
        asset = self.asset("LIFECALOR-CPU-V1.1.2.EXE")
        self.assertEqual(self.select("LifeCalor-cpu.exe", [asset])["name"], asset["name"])

    def test_exe_restart_uses_downloaded_filename_with_or_without_old_version(self):
        for build in ("LifeCalor", "LifeCalor-CPU"):
            for old_suffix in ("", "-v1.1.1"):
                with self.subTest(build=build, old_suffix=old_suffix):
                    app_path = str(Path("installed") / f"{build}{old_suffix}.exe")
                    downloaded = str(Path("downloads") / f"{build}-v1.1.2.exe")
                    self.assertEqual(
                        update_restart_path(app_path, downloaded, "1.1.2"),
                        str(Path("installed") / f"{build}-v1.1.2.exe"),
                    )

    def test_zip_restart_keeps_existing_executable_layout(self):
        for build in ("LifeCalor", "LifeCalor-CPU"):
            for old_suffix, new_suffix in (("", ""), ("-v1.1.1", "-v1.1.2")):
                with self.subTest(build=build, old_suffix=old_suffix):
                    app_path = str(Path("installed") / f"{build}{old_suffix}.exe")
                    self.assertEqual(
                        update_restart_path(app_path, f"{build}-v1.1.2.zip", "1.1.2"),
                        str(Path("installed") / f"{build}{new_suffix}.exe"),
                    )

    def test_frozen_application_path_uses_sys_executable(self):
        executable = str(Path("installed") / "LifeCalor-CPU.exe")
        with patch.object(sys, "frozen", True, create=True), patch.object(sys, "executable", executable), patch.object(sys, "argv", ["MainWindow.py"]):
            self.assertEqual(current_application_path(), os.path.abspath(executable))

    def test_source_application_path_uses_entry_script(self):
        script = str(CORE / "MainWindow.py")
        with patch.object(sys, "frozen", False, create=True), patch.object(sys, "argv", [script]):
            self.assertEqual(current_application_path(), script)


if __name__ == "__main__":
    unittest.main()
