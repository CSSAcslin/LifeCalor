import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from compute.capabilities import (
    HardwareSnapshot,
    mark_algorithm_verified,
    merge_device_capability,
    probe_hardware,
    reconcile_probe_snapshot,
    set_cached_snapshot,
)
from compute.model import (
    BackendPreference,
    CapabilityStatus,
    DeviceCapability,
    PrecisionPolicy,
)
from compute.settings import ALGORITHM_KEYS, ComputePreferences, ComputeSettingsStore


class _Settings:
    def __init__(self):
        self.values = {}
        self.synced = False

    def value(self, key, default=None):
        return self.values.get(key, default)

    def setValue(self, key, value):
        self.values[key] = value

    def sync(self):
        self.synced = True


class ComputeSettingsTests(unittest.TestCase):
    def tearDown(self):
        set_cached_snapshot(None)

    def test_defaults_are_cpu_only_safe_and_compatibility_precision(self):
        settings = _Settings()
        values = ComputeSettingsStore(settings).load()
        self.assertEqual(values.default_backend, BackendPreference.AUTO)
        self.assertEqual(values.precision, PrecisionPolicy.COMPATIBILITY)
        self.assertTrue(values.allow_cpu_fallback)
        self.assertGreaterEqual(values.cpu_workers, 1)

    def test_settings_persist_and_are_clamped(self):
        settings = _Settings()
        store = ComputeSettingsStore(settings)
        saved = store.save(ComputePreferences(
            default_backend=BackendPreference.GPU,
            precision=PrecisionPolicy.DOUBLE,
            allow_cpu_fallback=False,
            cpu_workers=10000,
            host_memory_limit_mb=10,
            gpu_memory_percent=99,
            preferred_device="nvidia:0",
            algorithm_backends=tuple(
                (key, BackendPreference.CPU) for key in ALGORITHM_KEYS
            ),
        ))

        self.assertTrue(settings.synced)
        self.assertEqual(saved.host_memory_limit_mb, 256)
        self.assertEqual(saved.gpu_memory_percent, 90)
        self.assertEqual(store.load().preferred_device, "nvidia:0")
        self.assertEqual(store.load().backend_for("stft"), BackendPreference.CPU)

    def test_hardware_probe_does_not_import_cupy(self):
        sys.modules.pop("cupy", None)
        with patch("compute.capabilities._nvidia_devices", return_value=[]), patch(
            "compute.capabilities._windows_display_devices", return_value=[]
        ):
            snapshot = probe_hardware()

        self.assertIsInstance(snapshot, HardwareSnapshot)
        self.assertNotIn("cupy", sys.modules)

    def test_mainwindow_exposes_shared_compute_settings_dialog(self):
        source = (CORE / "MainWindow.py").read_text(encoding="utf-8")
        self.assertIn('edit_menu.addAction("计算与加速")', source)
        self.assertIn("ComputeSettingsDialog(self.settings, parent=self)", source)

    def test_hardware_refresh_preserves_session_cuda_validation(self):
        validated = DeviceCapability(
            device_id="nvidia:0",
            kind="gpu",
            name="Validated GPU",
            vendor="NVIDIA",
            status=CapabilityStatus.AVAILABLE,
            backend="CuPy/CUDA",
            detail="CUDA FFT 快速自检通过",
            supported_algorithms=("stft", "fft2"),
        )
        set_cached_snapshot(HardwareSnapshot(
            1.0, "CPU", 4, 8, 0.0, 16, 8, (validated,)
        ))
        refreshed = reconcile_probe_snapshot(HardwareSnapshot(
            2.0,
            "CPU",
            4,
            8,
            0.0,
            16,
            8,
            (DeviceCapability(
                device_id="nvidia:0",
                kind="gpu",
                name="Fresh GPU Name",
                vendor="NVIDIA",
                status=CapabilityStatus.UNSUPPORTED,
                total_memory_bytes=32,
                free_memory_bytes=24,
                driver="new-driver",
                backend="CuPy/CUDA",
            ),),
        ))

        device = refreshed.devices[0]
        self.assertEqual(device.status, CapabilityStatus.AVAILABLE)
        self.assertEqual(device.name, "Fresh GPU Name")
        self.assertEqual(device.driver, "new-driver")
        self.assertEqual(device.supported_algorithms, ("stft", "fft2"))

    def test_real_gpu_task_marks_algorithm_verified_for_session(self):
        device = DeviceCapability(
            device_id="nvidia:0",
            kind="gpu",
            name="GPU",
            status=CapabilityStatus.AVAILABLE,
            supported_algorithms=("stft", "fft2"),
        )
        set_cached_snapshot(HardwareSnapshot(
            1.0, "CPU", 4, 8, 0.0, 16, 8, (device,)
        ))

        verified = mark_algorithm_verified("nvidia:0", "stft")

        self.assertEqual(verified.verified_algorithms, ("stft",))
        self.assertIn("实际任务验证", verified.detail)

    def test_retest_does_not_erase_real_task_verification(self):
        verified = DeviceCapability(
            device_id="nvidia:0",
            kind="gpu",
            name="GPU",
            status=CapabilityStatus.AVAILABLE,
            supported_algorithms=("stft", "fft2"),
            verified_algorithms=("stft",),
        )
        snapshot = HardwareSnapshot(
            1.0, "CPU", 4, 8, 0.0, 16, 8, (verified,)
        )
        retested = DeviceCapability(
            device_id="nvidia:0",
            kind="gpu",
            name="GPU",
            status=CapabilityStatus.AVAILABLE,
            supported_algorithms=("stft", "fft2"),
        )

        merged = merge_device_capability(snapshot, retested)

        self.assertEqual(merged.devices[0].verified_algorithms, ("stft",))


if __name__ == "__main__":
    unittest.main()
