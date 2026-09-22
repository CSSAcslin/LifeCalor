import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from PyQt5.QtCore import QSettings
from PyQt5.QtWidgets import QApplication

from compute.capabilities import HardwareSnapshot
from compute.model import CapabilityStatus, DeviceCapability
from compute.dialog import ComputeSettingsDialog
from ExtraDialog import CWTComputePop, STFTComputePop


class ComputeDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_dialog_probes_in_background_and_persists_defaults(self):
        snapshot = HardwareSnapshot(
            captured_at=0,
            cpu_name="Test CPU",
            physical_cores=4,
            logical_cores=8,
            cpu_percent=12.0,
            ram_total_bytes=16 * 1024 ** 3,
            ram_available_bytes=8 * 1024 ** 3,
            devices=(),
        )
        with tempfile.TemporaryDirectory() as directory, patch(
            "compute.dialog.probe_hardware", return_value=snapshot
        ):
            settings = QSettings(str(Path(directory) / "settings.ini"), QSettings.IniFormat)
            dialog = ComputeSettingsDialog(settings)
            self.assertTrue(dialog._probe_thread.wait(3000))
            self.app.processEvents()
            self.assertEqual(dialog.probe_status.text(), "检测完成")
            self.assertEqual(dialog.tabs.count(), 2)
            gpu_index = dialog.backend_combo.findData("gpu")
            self.assertFalse(dialog.backend_combo.model().item(gpu_index).isEnabled())
            dialog._save()
            self.assertEqual(settings.value("compute/precision"), "compatibility")
            dialog.close()

    def test_auto_resource_fields_show_effective_values_and_restore_manual_values(self):
        snapshot = HardwareSnapshot(
            captured_at=0,
            cpu_name="Test CPU",
            physical_cores=8,
            logical_cores=16,
            cpu_percent=10.0,
            ram_total_bytes=16 * 1024 ** 3,
            ram_available_bytes=10 * 1024 ** 3,
            devices=(),
        )
        with tempfile.TemporaryDirectory() as directory, patch(
            "compute.dialog.probe_hardware", return_value=snapshot
        ):
            settings = QSettings(str(Path(directory) / "settings.ini"), QSettings.IniFormat)
            settings.setValue("compute/auto_cpu", True)
            settings.setValue("compute/auto_memory", True)
            settings.setValue("compute/auto_gpu", True)
            settings.setValue("compute/cpu_workers", 2)
            settings.setValue("compute/host_memory_limit_mb", 1024)
            settings.setValue("compute/gpu_memory_percent", 55)
            dialog = ComputeSettingsDialog(settings)
            self.assertTrue(dialog._probe_thread.wait(3000))
            self.app.processEvents()

            recommendation = dialog.recommendation
            self.assertFalse(dialog.cpu_workers_spin.isEnabled())
            self.assertEqual(dialog.cpu_workers_spin.value(), recommendation.cpu_workers)
            self.assertFalse(dialog.host_memory_spin.isEnabled())
            self.assertEqual(
                dialog.host_memory_spin.value(),
                recommendation.host_memory_limit_mb,
            )
            self.assertFalse(dialog.gpu_memory_spin.isEnabled())
            self.assertEqual(dialog.gpu_memory_spin.value(), 0)
            self.assertEqual(dialog.gpu_memory_spin.text(), "不可用")

            dialog.auto_cpu_check.setChecked(False)
            dialog.auto_memory_check.setChecked(False)
            dialog.auto_gpu_check.setChecked(False)
            self.assertEqual(dialog.cpu_workers_spin.value(), 2)
            self.assertEqual(dialog.host_memory_spin.value(), 1024)
            self.assertEqual(dialog.gpu_memory_spin.value(), 55)
            dialog.close()
    def test_cuda_result_updates_supported_algorithm_status(self):
        snapshot = HardwareSnapshot(
            captured_at=0,
            cpu_name="Test CPU",
            physical_cores=4,
            logical_cores=8,
            cpu_percent=12.0,
            ram_total_bytes=16 * 1024 ** 3,
            ram_available_bytes=8 * 1024 ** 3,
            devices=(),
        )
        with tempfile.TemporaryDirectory() as directory, patch(
            "compute.dialog.probe_hardware", return_value=snapshot
        ):
            settings = QSettings(str(Path(directory) / "settings.ini"), QSettings.IniFormat)
            dialog = ComputeSettingsDialog(settings)
            self.assertTrue(dialog._probe_thread.wait(3000))
            self.app.processEvents()
            dialog._apply_cuda_capability(DeviceCapability(
                "nvidia:0", "gpu", "Test GPU",
                status=CapabilityStatus.AVAILABLE,
                detail="CUDA FFT 快速自检通过",
                supported_algorithms=("stft", "fft2"),
            ))
            self.assertIn("自检通过", dialog.algorithm_status_items["stft"].text())
            self.assertIn("自检通过", dialog.algorithm_status_items["fft2"].text())
            dialog.close()

    def test_algorithm_status_distinguishes_real_gpu_task_validation(self):
        snapshot = HardwareSnapshot(
            captured_at=0,
            cpu_name="Test CPU",
            physical_cores=4,
            logical_cores=8,
            cpu_percent=12.0,
            ram_total_bytes=16 * 1024 ** 3,
            ram_available_bytes=8 * 1024 ** 3,
            devices=(),
        )
        with tempfile.TemporaryDirectory() as directory, patch(
            "compute.dialog.probe_hardware", return_value=snapshot
        ):
            settings = QSettings(str(Path(directory) / "settings.ini"), QSettings.IniFormat)
            dialog = ComputeSettingsDialog(settings)
            self.assertTrue(dialog._probe_thread.wait(3000))
            self.app.processEvents()
            dialog._apply_cuda_capability(DeviceCapability(
                "nvidia:0", "gpu", "Test GPU",
                status=CapabilityStatus.AVAILABLE,
                detail="CUDA 可用；已完成实际任务验证：STFT",
                supported_algorithms=("stft", "fft2"),
                verified_algorithms=("stft",),
            ))
            self.assertIn(
                "实际任务验证", dialog.algorithm_status_items["stft"].text()
            )
            self.assertIn("等待实际任务验证", dialog.algorithm_status_items["fft2"].text())
            self.assertIn(
                "尚无该算法的 GPU 实现",
                dialog.algorithm_status_items["cwt"].text(),
            )
            dialog.close()

    def test_stft_dialog_returns_task_backend_and_precision(self):
        params = {
            "target_freq": 8.0,
            "EM_fps": 64,
            "stft_scale_range": 0,
            "stft_window_size": 32,
            "stft_noverlap": 16,
            "custom_nfft": 32,
        }
        dialog = STFTComputePop(
            params,
            "process",
            compute_options={"backend": "gpu", "precision": "single", "cpu_workers": 2},
        )
        options = dialog.selected_compute_options()
        self.assertEqual(options["backend"], "gpu")
        self.assertEqual(options["precision"], "single")
        self.assertFalse(dialog.multiprocess_check.isVisible())
        dialog.close()

    def test_cwt_dialog_returns_task_backend_and_precision(self):
        params = {
            "target_freq": 8.0,
            "EM_fps": 128,
            "cwt_scale_range": 2.0,
            "cwt_type": "morl",
        }
        dialog = CWTComputePop(
            params,
            "signal",
            compute_options={"backend": "cpu", "precision": "double"},
        )
        options = dialog.selected_compute_options()
        self.assertEqual(options["backend"], "cpu")
        self.assertEqual(options["precision"], "double")
        self.assertEqual(dialog.cwt_size_input.minimum(), 1)
        dialog.close()

    def test_cwt_dialog_normalizes_legacy_wavelet_whitespace(self):
        params = {
            "target_freq": 8.0,
            "EM_fps": 100,
            "cwt_total_scales": 4,
            "cwt_scale_range": 2.0,
            "cwt_type": "cmor8-3 ",
        }
        dialog = CWTComputePop(params, "signal")
        self.addCleanup(dialog.close)

        self.assertEqual(dialog.wavelet.currentText(), "cmor8-3")
        self.assertNotIn("cmor8-3 ", [
            dialog.wavelet.itemText(index)
            for index in range(dialog.wavelet.count())
        ])


if __name__ == "__main__":
    unittest.main()
