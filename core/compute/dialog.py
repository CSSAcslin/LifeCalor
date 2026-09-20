from __future__ import annotations

import logging
from dataclasses import replace

from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .capabilities import (
    HardwareSnapshot,
    get_cached_snapshot,
    merge_device_capability,
    probe_hardware,
    reconcile_probe_snapshot,
    set_cached_snapshot,
)
from .model import BackendPreference, CapabilityStatus, PrecisionPolicy
from .resources import recommend_resources
from .settings import ALGORITHM_KEYS, ComputePreferences, ComputeSettingsStore


_ACTIVE_PROBE_THREADS = set()

BACKEND_ITEMS = (
    ("跟随全局", BackendPreference.FOLLOW_GLOBAL),
    ("自动", BackendPreference.AUTO),
    ("CPU", BackendPreference.CPU),
    ("GPU", BackendPreference.GPU),
)
ALGORITHM_LABELS = {
    "stft": "STFT",
    "cwt": "CWT",
    "lifetime_single": "单指数寿命",
    "lifetime_double": "双指数寿命",
    "em_preprocess": "EM 预处理",
    "fft2": "FFT / IFFT",
}


def _format_bytes(value):
    value = max(0, int(value or 0))
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{value} B"
        value /= 1024


def _set_combo_item_enabled(combo, data, enabled, tooltip=""):
    index = combo.findData(data)
    if index < 0:
        return
    item = combo.model().item(index)
    if item is not None:
        item.setEnabled(bool(enabled))
        item.setToolTip(tooltip)


class HardwareProbeThread(QThread):
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)

    def run(self):
        try:
            self.completed.emit(probe_hardware())
        except Exception as exc:
            logging.exception("硬件能力探测失败")
            self.failed.emit(f"{type(exc).__name__}: {exc}")


def track_probe_thread(thread):
    _ACTIVE_PROBE_THREADS.add(thread)
    thread.finished.connect(
        lambda current=thread: _ACTIVE_PROBE_THREADS.discard(current)
    )
    return thread


class CudaSelfTestThread(QThread):
    completed = pyqtSignal(object)

    def __init__(self, device_index=0, parent=None):
        super().__init__(parent)
        self.device_index = max(0, int(device_index))

    def run(self):
        from compute.worker import probe_cuda_capability_isolated

        self.completed.emit(
            probe_cuda_capability_isolated(self.device_index, timeout=15.0)
        )


class ComputeSettingsDialog(QDialog):
    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("计算与加速")
        self.setMinimumSize(820, 600)
        self.store = ComputeSettingsStore(settings)
        self.preferences = self.store.load()
        self.snapshot = get_cached_snapshot()
        self.recommendation = recommend_resources(self.snapshot)
        self._loading_preferences = False
        self._manual_cpu_workers = self.preferences.cpu_workers
        self._manual_host_memory_mb = self.preferences.host_memory_limit_mb
        self._manual_gpu_percent = self.preferences.gpu_memory_percent
        self._probe_thread = None
        self._cuda_test_thread = None
        self._cuda_test_started = False
        self._cuda_device_index = 0
        self.algorithm_combos = {}
        self.algorithm_status_items = {}
        self._build_ui()
        self._load_preferences()
        if self.snapshot is not None:
            self._apply_snapshot(self.snapshot, allow_auto_test=False)
        self._start_probe()

    def _build_ui(self):
        root = QVBoxLayout(self)
        self.tabs = QTabWidget()
        root.addWidget(self.tabs, 1)
        self.tabs.addTab(self._build_hardware_strategy_tab(), "硬件与策略")
        self.tabs.addTab(self._build_algorithms_tab(), "算法设置")

        self.button_box = QDialogButtonBox(
            QDialogButtonBox.Save | QDialogButtonBox.Cancel
        )
        self.button_box.button(QDialogButtonBox.Save).setText("保存")
        self.button_box.button(QDialogButtonBox.Cancel).setText("取消")
        self.button_box.accepted.connect(self._save)
        self.button_box.rejected.connect(self.reject)
        root.addWidget(self.button_box)

    def _build_hardware_strategy_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        hardware_group = QGroupBox("硬件状态与自动建议")
        hardware_layout = QVBoxLayout(hardware_group)
        self.hardware_table = QTableWidget(0, 5)
        self.hardware_table.setHorizontalHeaderLabels(
            ("硬件", "设备 / 容量", "当前状态", "自动建议", "可用性")
        )
        self.hardware_table.verticalHeader().setVisible(False)
        self.hardware_table.setEditTriggers(QTableWidget.NoEditTriggers)
        for column in range(4):
            self.hardware_table.horizontalHeader().setSectionResizeMode(
                column, QHeaderView.ResizeToContents
            )
        self.hardware_table.horizontalHeader().setSectionResizeMode(
            4, QHeaderView.Stretch
        )
        hardware_layout.addWidget(self.hardware_table)

        controls = QHBoxLayout()
        self.probe_status = QLabel("等待检测")
        controls.addWidget(self.probe_status, 1)
        self.cuda_test_button = QPushButton("CUDA 重新自检")
        self.cuda_test_button.setToolTip("在隔离进程中验证 CUDA 运行库和一次小型 FFT")
        self.cuda_test_button.clicked.connect(self._start_cuda_self_test)
        self.cuda_test_button.setEnabled(False)
        controls.addWidget(self.cuda_test_button)
        self.copy_button = QPushButton("复制诊断摘要")
        self.copy_button.clicked.connect(self._copy_diagnostic)
        controls.addWidget(self.copy_button)
        self.refresh_button = QPushButton("刷新检测")
        self.refresh_button.clicked.connect(self._start_probe)
        controls.addWidget(self.refresh_button)
        hardware_layout.addLayout(controls)
        layout.addWidget(hardware_group)

        policy_group = QGroupBox("默认执行策略")
        form = QFormLayout(policy_group)

        self.backend_combo = QComboBox()
        for label, value in BACKEND_ITEMS[1:]:
            self.backend_combo.addItem(label, value.value)
        self.backend_combo.setToolTip(
            "自动模式只会选择已通过隔离自检且支持当前算法的 GPU，否则使用 CPU。"
        )
        form.addRow("默认后端", self.backend_combo)

        self.precision_combo = QComboBox()
        for label, value in (
            ("兼容现有（推荐）", PrecisionPolicy.COMPATIBILITY),
            ("保留输入精度", PrecisionPolicy.PRESERVE_INPUT),
            ("单精度", PrecisionPolicy.SINGLE),
            ("双精度", PrecisionPolicy.DOUBLE),
        ):
            self.precision_combo.addItem(label, value.value)
        form.addRow("默认精度", self.precision_combo)

        self.fallback_check = QCheckBox("GPU 不可用或资源不足时回退到 CPU")
        form.addRow("回退策略", self.fallback_check)

        self.auto_cpu_check = QCheckBox("根据物理核心数和当前负载自动分配")
        self.auto_cpu_check.toggled.connect(self._sync_resource_controls)
        form.addRow("CPU 自动分配", self.auto_cpu_check)
        self.cpu_workers_spin = QSpinBox()
        self.cpu_workers_spin.setRange(1, 256)
        self.cpu_workers_spin.setToolTip("关闭自动分配后，限制单个新任务的 CPU 并行工作数。")
        form.addRow("CPU 配额", self.cpu_workers_spin)

        self.auto_memory_check = QCheckBox("根据当前可用内存并保留系统余量自动分配")
        self.auto_memory_check.toggled.connect(self._sync_resource_controls)
        form.addRow("内存自动分配", self.auto_memory_check)
        self.host_memory_spin = QSpinBox()
        self.host_memory_spin.setRange(256, 1024 * 1024)
        self.host_memory_spin.setSuffix(" MB")
        self.host_memory_spin.setToolTip("关闭自动分配后，设置单个新任务的工作内存上限。")
        form.addRow("内存配额", self.host_memory_spin)

        self.auto_gpu_check = QCheckBox("根据空闲显存并保留安全余量自动分配")
        self.auto_gpu_check.toggled.connect(self._sync_resource_controls)
        form.addRow("显存自动分配", self.auto_gpu_check)
        self.gpu_memory_spin = QSpinBox()
        self.gpu_memory_spin.setRange(0, 90)
        self.gpu_memory_spin.setSpecialValueText("不可用")
        self.gpu_memory_spin.setSuffix(" %")
        self.gpu_memory_spin.setToolTip("关闭自动分配后，设置 GPU 任务的显存比例上限。")
        form.addRow("显存配额", self.gpu_memory_spin)

        self.device_combo = QComboBox()
        self.device_combo.addItem("自动选择", "")
        self.device_combo.setToolTip("自动模式优先选择已通过自检且可用显存最多的设备。")
        form.addRow("首选设备", self.device_combo)

        layout.addWidget(policy_group)
        return tab

    def _build_algorithms_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.algorithm_table = QTableWidget(len(ALGORITHM_KEYS), 3)
        self.algorithm_table.setHorizontalHeaderLabels(
            ("算法", "后端偏好", "当前状态")
        )
        self.algorithm_table.verticalHeader().setVisible(False)
        self.algorithm_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.algorithm_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeToContents
        )
        self.algorithm_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeToContents
        )
        self.algorithm_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.Stretch
        )
        for row, algorithm in enumerate(ALGORITHM_KEYS):
            self.algorithm_table.setItem(
                row, 0, QTableWidgetItem(ALGORITHM_LABELS[algorithm])
            )
            combo = QComboBox()
            for label, value in BACKEND_ITEMS:
                combo.addItem(label, value.value)
            combo.setToolTip(
                "跟随全局使用默认策略；不可用或尚未验证的 GPU 选项会被禁用。"
            )
            self.algorithm_table.setCellWidget(row, 1, combo)
            status_item = QTableWidgetItem("CPU 可用；等待硬件检测")
            self.algorithm_table.setItem(row, 2, status_item)
            self.algorithm_status_items[algorithm] = status_item
            self.algorithm_combos[algorithm] = combo
        layout.addWidget(self.algorithm_table)
        return tab

    def _load_preferences(self):
        self._loading_preferences = True
        self._select_data(self.backend_combo, self.preferences.default_backend.value)
        self._select_data(self.precision_combo, self.preferences.precision.value)
        self.fallback_check.setChecked(self.preferences.allow_cpu_fallback)
        self.auto_cpu_check.setChecked(self.preferences.auto_cpu)
        self.auto_memory_check.setChecked(self.preferences.auto_memory)
        self.auto_gpu_check.setChecked(self.preferences.auto_gpu)
        self.cpu_workers_spin.setValue(self._manual_cpu_workers)
        self.host_memory_spin.setValue(self._manual_host_memory_mb)
        self.gpu_memory_spin.setValue(self._manual_gpu_percent)
        for algorithm, backend in self.preferences.algorithm_backends:
            combo = self.algorithm_combos.get(algorithm)
            if combo is not None:
                self._select_data(combo, backend.value)
        self._loading_preferences = False
        self._sync_resource_controls()

    @staticmethod
    def _select_data(combo, value):
        index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)

    def _sync_resource_controls(self):
        if self._loading_preferences:
            return
        recommendation = self.recommendation or recommend_resources(self.snapshot)

        if self.auto_cpu_check.isChecked():
            if self.cpu_workers_spin.isEnabled():
                self._manual_cpu_workers = self.cpu_workers_spin.value()
            self.cpu_workers_spin.setValue(recommendation.cpu_workers)
            self.cpu_workers_spin.setEnabled(False)
            self.cpu_workers_spin.setToolTip("当前自动 CPU 并行配额。")
        else:
            if not self.cpu_workers_spin.isEnabled():
                self.cpu_workers_spin.setValue(self._manual_cpu_workers)
            self.cpu_workers_spin.setEnabled(True)
            self.cpu_workers_spin.setToolTip("单个新任务的手动 CPU 并行工作数。")

        if self.auto_memory_check.isChecked():
            if self.host_memory_spin.isEnabled():
                self._manual_host_memory_mb = self.host_memory_spin.value()
            self.host_memory_spin.setValue(recommendation.host_memory_limit_mb)
            self.host_memory_spin.setEnabled(False)
            self.host_memory_spin.setToolTip("根据当前可用内存计算的自动任务配额。")
        else:
            if not self.host_memory_spin.isEnabled():
                self.host_memory_spin.setValue(self._manual_host_memory_mb)
            self.host_memory_spin.setEnabled(True)
            self.host_memory_spin.setToolTip("单个新任务的手动工作内存上限。")

        if self.auto_gpu_check.isChecked():
            if self.gpu_memory_spin.isEnabled():
                self._manual_gpu_percent = max(10, self.gpu_memory_spin.value())
            automatic_gpu = (
                recommendation.gpu_memory_percent
                if recommendation.selected_device else 0
            )
            self.gpu_memory_spin.setValue(automatic_gpu)
            self.gpu_memory_spin.setEnabled(False)
            self.gpu_memory_spin.setToolTip(
                "当前自动显存配额；无可用 GPU 时显示不可用。"
            )
        else:
            if not self.gpu_memory_spin.isEnabled():
                self.gpu_memory_spin.setValue(self._manual_gpu_percent)
            self.gpu_memory_spin.setEnabled(True)
            self.gpu_memory_spin.setToolTip("GPU 任务的手动显存比例上限。")

    def _start_probe(self):
        if self._probe_thread is not None and self._probe_thread.isRunning():
            return
        self.refresh_button.setEnabled(False)
        self.probe_status.setText("正在后台检测硬件...")
        thread = HardwareProbeThread()
        self._probe_thread = thread
        track_probe_thread(thread)
        thread.completed.connect(self._apply_snapshot)
        thread.failed.connect(self._probe_failed)
        thread.finished.connect(lambda: self.refresh_button.setEnabled(True))
        thread.start()

    def _apply_snapshot(self, snapshot: HardwareSnapshot, allow_auto_test=True):
        if allow_auto_test:
            snapshot = reconcile_probe_snapshot(snapshot)
        self.snapshot = set_cached_snapshot(snapshot)
        recommendation = recommend_resources(snapshot)
        self.recommendation = recommendation
        self._sync_resource_controls()
        rows = [
            (
                "CPU",
                f"{snapshot.cpu_name} · {snapshot.physical_cores} 核 / "
                f"{snapshot.logical_cores} 线程",
                "占用不可用" if snapshot.cpu_percent is None
                else f"当前占用 {snapshot.cpu_percent:.0f}%",
                f"{recommendation.cpu_workers} 个并行工作",
                "可用",
            ),
            (
                "内存",
                f"总计 {_format_bytes(snapshot.ram_total_bytes)}",
                f"可用 {_format_bytes(snapshot.ram_available_bytes)}",
                f"{recommendation.host_memory_limit_mb} MB / 任务",
                "可用",
            ),
        ]
        for device in snapshot.devices:
            available = device.status is CapabilityStatus.AVAILABLE
            rows.append((
                "GPU",
                f"{device.name} · {_format_bytes(device.total_memory_bytes)}",
                f"可用显存 {_format_bytes(device.free_memory_bytes)} · "
                f"驱动 {device.driver or '未知'}",
                (
                    f"{recommendation.gpu_memory_percent}% 显存"
                    if available else "不参与自动分配"
                ),
                device.detail or device.status.value,
            ))
        if not snapshot.devices:
            rows.append(("GPU", "未检测到计算设备", "-", "不参与自动分配", "不可用"))

        self.hardware_table.setRowCount(len(rows))
        for row, values in enumerate(rows):
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setToolTip(str(value))
                self.hardware_table.setItem(row, column, item)

        preferred = str(
            self.device_combo.currentData() or self.preferences.preferred_device
        )
        self.device_combo.clear()
        self.device_combo.addItem("自动选择", "")
        for device in snapshot.devices:
            self.device_combo.addItem(device.name, device.device_id)
            index = self.device_combo.count() - 1
            item = self.device_combo.model().item(index)
            if item is not None:
                item.setEnabled(device.status is CapabilityStatus.AVAILABLE)
                item.setToolTip(device.detail)
        self._select_data(self.device_combo, preferred)
        self._apply_backend_availability()

        candidates = [
            device for device in snapshot.devices
            if device.vendor.upper() == "NVIDIA" and bool(device.backend)
        ]
        candidate = (
            max(candidates, key=lambda device: int(device.free_memory_bytes or 0))
            if candidates else None
        )
        if candidate is not None:
            try:
                self._cuda_device_index = int(candidate.device_id.rsplit(":", 1)[-1])
            except ValueError:
                self._cuda_device_index = 0
        self.cuda_test_button.setEnabled(candidate is not None)
        self.cuda_test_button.setToolTip(
            "在隔离进程中重新验证 CUDA 运行库和一次小型 FFT"
            if candidate is not None
            else "当前安装包未包含可用的 CUDA Python 后端"
        )
        self.probe_status.setText(
            "检测完成" if not snapshot.probe_error
            else f"部分信息不可用：{snapshot.probe_error}"
        )
        if (
            allow_auto_test
            and candidate is not None
            and candidate.status in (
                CapabilityStatus.UNAVAILABLE,
                CapabilityStatus.UNSUPPORTED,
            )
            and not self._cuda_test_started
        ):
            self._start_cuda_self_test()

    def _apply_backend_availability(self):
        devices = tuple(self.snapshot.devices if self.snapshot is not None else ())
        any_gpu = any(
            device.status is CapabilityStatus.AVAILABLE
            and bool(device.supported_algorithms)
            for device in devices
        )
        _set_combo_item_enabled(
            self.backend_combo,
            BackendPreference.GPU.value,
            any_gpu,
            "至少一个 GPU 已通过隔离自检" if any_gpu
            else "没有已通过隔离自检的 GPU",
        )
        if not any_gpu and self.backend_combo.currentData() == BackendPreference.GPU.value:
            self._select_data(self.backend_combo, BackendPreference.AUTO.value)

        for algorithm, combo in self.algorithm_combos.items():
            supported = any(
                device.status is CapabilityStatus.AVAILABLE
                and algorithm in device.supported_algorithms
                for device in devices
            )
            verified = any(
                device.status is CapabilityStatus.AVAILABLE
                and algorithm in device.verified_algorithms
                for device in devices
            )
            _set_combo_item_enabled(
                combo,
                BackendPreference.GPU.value,
                supported,
                "该算法 GPU 后端已通过自检" if supported
                else "该算法尚无可用且已验证的 GPU 后端",
            )
            if not supported and combo.currentData() == BackendPreference.GPU.value:
                self._select_data(combo, BackendPreference.FOLLOW_GLOBAL.value)
            if verified:
                status = "CPU 可用；GPU 已完成实际任务验证"
            elif supported:
                status = "CPU 可用；GPU 自检通过，等待实际任务验证"
            elif any_gpu:
                status = "CPU 可用；当前版本尚无该算法的 GPU 实现"
            else:
                status = "CPU 可用；GPU 不可用或尚未验证"
            self.algorithm_status_items[algorithm].setText(status)

    def _start_cuda_self_test(self):
        if self._cuda_test_thread is not None and self._cuda_test_thread.isRunning():
            return
        self._cuda_test_started = True
        self.cuda_test_button.setEnabled(False)
        self.probe_status.setText("正在隔离进程中执行 CUDA 快速自检...")
        thread = CudaSelfTestThread(self._cuda_device_index)
        self._cuda_test_thread = thread
        track_probe_thread(thread)
        thread.completed.connect(self._apply_cuda_capability)
        thread.start()

    def _apply_cuda_capability(self, capability):
        snapshot = merge_device_capability(self.snapshot, capability)
        self._apply_snapshot(snapshot, allow_auto_test=False)
        if capability.status is CapabilityStatus.AVAILABLE:
            self.probe_status.setText("CUDA FFT 快速自检通过")
            logging.info(
                "CUDA 自检通过: device=%s backend=%s memory_total=%s memory_free=%s",
                capability.name,
                capability.backend,
                capability.total_memory_bytes,
                capability.free_memory_bytes,
            )
        else:
            self.probe_status.setText(f"CUDA 自检未通过：{capability.detail}")
            logging.warning("CUDA 自检未通过: %s", capability.detail)

    def _probe_failed(self, message):
        self.probe_status.setText(f"检测失败：{message}")
        logging.warning("硬件检测失败: %s", message)

    def _copy_diagnostic(self):
        if self.snapshot is None:
            text = self.probe_status.text()
        else:
            recommendation = recommend_resources(self.snapshot)
            lines = [
                f"CPU: {self.snapshot.cpu_name}",
                f"Cores: {self.snapshot.physical_cores}/{self.snapshot.logical_cores}",
                f"RAM: {_format_bytes(self.snapshot.ram_total_bytes)} total, "
                f"{_format_bytes(self.snapshot.ram_available_bytes)} available",
                f"Auto: workers={recommendation.cpu_workers}; "
                f"RAM={recommendation.host_memory_limit_mb} MB; "
                f"GPU={recommendation.gpu_memory_percent}%",
            ]
            for device in self.snapshot.devices:
                lines.append(
                    f"GPU: {device.name}; vendor={device.vendor}; "
                    f"driver={device.driver}; "
                    f"memory={_format_bytes(device.total_memory_bytes)}; "
                    f"status={device.detail or device.status.value}"
                )
            if self.snapshot.probe_error:
                lines.append(f"Probe warning: {self.snapshot.probe_error}")
            text = "\n".join(lines)
        QApplication.clipboard().setText(text)
        self.probe_status.setText("诊断摘要已复制")

    def _save(self):
        overrides = tuple(
            (algorithm, BackendPreference(combo.currentData()))
            for algorithm, combo in self.algorithm_combos.items()
        )
        if not self.auto_cpu_check.isChecked():
            self._manual_cpu_workers = self.cpu_workers_spin.value()
        if not self.auto_memory_check.isChecked():
            self._manual_host_memory_mb = self.host_memory_spin.value()
        if not self.auto_gpu_check.isChecked():
            self._manual_gpu_percent = max(10, self.gpu_memory_spin.value())
        preferences = ComputePreferences(
            default_backend=BackendPreference(self.backend_combo.currentData()),
            precision=PrecisionPolicy(self.precision_combo.currentData()),
            allow_cpu_fallback=self.fallback_check.isChecked(),
            auto_cpu=self.auto_cpu_check.isChecked(),
            auto_memory=self.auto_memory_check.isChecked(),
            auto_gpu=self.auto_gpu_check.isChecked(),
            cpu_workers=self._manual_cpu_workers,
            host_memory_limit_mb=self._manual_host_memory_mb,
            gpu_memory_percent=self._manual_gpu_percent,
            preferred_device=(
                str(self.device_combo.currentData() or "")
                if self.snapshot is not None
                else self.preferences.preferred_device
            ),
            algorithm_backends=overrides,
        )
        self.preferences = self.store.save(preferences)
        logging.info(
            "计算设置已保存: backend=%s precision=%s fallback=%s "
            "auto_cpu=%s cpu_workers=%s auto_memory=%s memory_mb=%s "
            "auto_gpu=%s gpu_percent=%s device=%s",
            self.preferences.default_backend.value,
            self.preferences.precision.value,
            self.preferences.allow_cpu_fallback,
            self.preferences.auto_cpu,
            self.preferences.cpu_workers,
            self.preferences.auto_memory,
            self.preferences.host_memory_limit_mb,
            self.preferences.auto_gpu,
            self.preferences.gpu_memory_percent,
            self.preferences.preferred_device or "auto",
        )
        self.accept()
