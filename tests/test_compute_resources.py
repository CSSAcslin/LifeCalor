import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from PyQt5.QtCore import QSettings
from PyQt5.QtWidgets import QApplication

from compute.capabilities import HardwareSnapshot
from compute.model import (
    BackendPreference,
    CapabilityStatus,
    ComputeRequest,
    DeviceCapability,
    PrecisionPolicy,
    ResourceBudget,
    plan_execution_details,
)
from compute.planner import plan_compute
from compute.resources import recommend_resources
from compute.settings import ComputePreferences, ComputeSettingsStore
from tasks import TaskCoordinator
from tasks.panel import TaskPanel


class ComputeResourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    @staticmethod
    def snapshot(cpu_percent=10.0, devices=()):
        return HardwareSnapshot(
            captured_at=0,
            cpu_name="Test CPU",
            physical_cores=12,
            logical_cores=20,
            cpu_percent=cpu_percent,
            ram_total_bytes=24 * 1024 ** 3,
            ram_available_bytes=16 * 1024 ** 3,
            devices=tuple(devices),
        )

    def test_auto_budget_leaves_cpu_and_memory_headroom(self):
        recommendation = recommend_resources(self.snapshot())
        self.assertEqual(recommendation.cpu_workers, 9)
        self.assertGreaterEqual(recommendation.host_memory_limit_mb, 256)
        self.assertLess(
            recommendation.host_memory_limit_mb,
            16 * 1024,
        )

    def test_auto_budget_reduces_workers_when_machine_is_busy(self):
        idle = recommend_resources(self.snapshot(cpu_percent=10))
        busy = recommend_resources(self.snapshot(cpu_percent=90))
        self.assertLess(busy.cpu_workers, idle.cpu_workers)

    def test_available_gpu_is_selected_and_budgeted(self):
        gpu = DeviceCapability(
            "nvidia:0",
            "gpu",
            "Test GPU",
            vendor="NVIDIA",
            status=CapabilityStatus.AVAILABLE,
            total_memory_bytes=8 * 1024 ** 3,
            free_memory_bytes=6 * 1024 ** 3,
            supported_algorithms=("stft", "fft2"),
        )
        recommendation = recommend_resources(self.snapshot(devices=(gpu,)))
        self.assertEqual(recommendation.selected_device, "nvidia:0")
        self.assertGreaterEqual(recommendation.gpu_memory_percent, 10)
        self.assertLessEqual(recommendation.gpu_memory_percent, 75)

    def test_auto_backend_uses_only_validated_algorithm_capability(self):
        gpu = DeviceCapability(
            "nvidia:0",
            "gpu",
            "Test GPU",
            status=CapabilityStatus.AVAILABLE,
            supported_algorithms=("stft",),
        )
        request = ComputeRequest(
            task_id="task",
            attempt_id=0,
            algorithm="stft",
            data_id="data",
            shape=(32, 4, 4),
            dtype="float32",
            axes="THW",
            source=object(),
            backend=BackendPreference.AUTO,
            precision=PrecisionPolicy.COMPATIBILITY,
        )
        plan = plan_compute(
            request,
            ResourceBudget(
                host_limit_bytes=256 * 1024 ** 2,
                device_limit_bytes=128 * 1024 ** 2,
                cpu_workers=4,
            ),
            capabilities=(gpu,),
        )
        self.assertEqual(plan.actual_backend, "gpu")
        self.assertEqual(plan.selected_device, "Test GPU")
        details = plan_execution_details(plan)
        self.assertIn("CPU 4", details["resource_summary"])
        self.assertIn("VRAM", details["resource_summary"])

    def test_automatic_switches_persist_independently_from_manual_values(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = QSettings(
                str(Path(directory) / "settings.ini"), QSettings.IniFormat
            )
            store = ComputeSettingsStore(settings)
            store.save(ComputePreferences(
                auto_cpu=False,
                auto_memory=True,
                auto_gpu=False,
                cpu_workers=3,
                host_memory_limit_mb=2048,
                gpu_memory_percent=55,
            ))
            loaded = store.load()
            self.assertFalse(loaded.auto_cpu)
            self.assertTrue(loaded.auto_memory)
            self.assertFalse(loaded.auto_gpu)
            self.assertEqual(loaded.cpu_workers, 3)
            self.assertEqual(loaded.gpu_memory_percent, 55)

    def test_task_panel_shows_execution_device_and_resource_details(self):
        coordinator = TaskCoordinator()
        panel = TaskPanel(coordinator)
        task = coordinator.create_task("STFT", "em_processing")
        coordinator.configure_execution(
            task.task_id,
            requested_backend="auto",
            actual_backend="gpu",
            precision="compatibility",
            device="Test GPU",
            backend_reason="自动策略选择",
            resource_summary="CPU 4 · RAM 2.0 GB · VRAM 4.0 GB",
        )
        self.app.processEvents()
        item = panel._rows[task.task_id][0]
        self.assertIn("GPU", item.text(2))
        self.assertIn("Test GPU", item.text(2))
        self.assertIn("RAM 2.0 GB", item.text(3))
        self.assertIn("RAM 2.0 GB", item.toolTip(2))
        panel.close()


if __name__ == "__main__":
    unittest.main()
