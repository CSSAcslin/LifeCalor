"""Verify a Windows package through its GUI and frozen CUDA worker entry points."""

from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import json
import multiprocessing
from multiprocessing import spawn
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import psutil
from scipy.signal import get_window


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "core"))


def verify_worker(executable, backend):
    from compute.worker import CudaBackendError, CudaWorkerClient
    from compute.backends.cpu import stft_block_cpu

    # Use the frozen spawn command, so the package imports its own worker code.
    original_frozen = getattr(sys, "frozen", None)
    original_preparation = spawn.get_preparation_data

    def package_preparation(name):
        data = original_preparation(name)
        # Keep the package's import paths, rather than the source virtualenv.
        for key in ("sys_path", "init_main_from_name", "init_main_from_path"):
            data.pop(key, None)
        return data

    multiprocessing.set_executable(str(executable))
    sys.frozen = True
    spawn.get_preparation_data = package_preparation
    try:
        with CudaWorkerClient() as worker:
            try:
                device = worker.self_test(timeout=120)
            except CudaBackendError as error:
                if backend == "cpu" and "No module named" in str(error):
                    return {"cuda_available": False}
                print(error.details, file=sys.stderr)
                raise
            if backend == "cpu":
                raise AssertionError("CPU package unexpectedly contains CUDA")
            values = np.sin(2 * np.pi * 8 * np.arange(128) / 64).astype(np.float32)
            block = np.broadcast_to(values[:, None, None], (128, 2, 3)).copy()
            params = dict(fps=64, window=get_window("hann", 32), window_size=32, noverlap=16,
                          nfft=64, target_freq=8, scale_range=2)
            expected = stft_block_cpu(block, params, compute_dtype="float32",
                                      output_dtype="float32")[0]
            actual = worker.execute("stft", block=block, params=params,
                                    compute_dtype="float32", output_dtype="float32")[0]
            np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
            times = np.linspace(0.0, 30.0, 200)
            fit_params = dict(from_start_cal=True, r_squared_min=0.4,
                              peak_range=(0, 200), tau_range=(1e-3, 100.0))
            fitted = {}
            for model in ("single", "double"):
                signal = 30 * np.exp(-times / 2.5) + 0.5
                expected_tau = [2.5]
                if model == "double":
                    signal += 15 * np.exp(-times / 12)
                    expected_tau = [2.5, 12.0]
                outputs = worker.execute(
                    "lifetime_" + model, block=signal[:, None, None],
                    time_points=times, data_type=None, fit_params=fit_params,
                    model_type=model, compute_dtype="float64", output_dtype="float64",
                )
                assert int(outputs["fit_status"][0, 0]) == 1
                fields = ("lifetime_map",) if model == "single" else ("tau1_map", "tau2_map")
                fitted[model] = sorted(float(outputs[key][0, 0]) for key in fields)
                np.testing.assert_allclose(fitted[model], expected_tau, rtol=1e-4, atol=1e-5)
            return {"cuda_available": True, "device": device["name"],
                    "stft_max_absolute_error": float(np.max(np.abs(actual - expected))),
                    "fitted_lifetimes": fitted}
    finally:
        spawn.get_preparation_data = original_preparation
        multiprocessing.set_executable(sys.executable)
        if original_frozen is None:
            del sys.frozen
        else:
            sys.frozen = original_frozen


def verify_window(executable, timeout, close_delay):
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    user32.EnumWindows.argtypes = [callback_type, wintypes.LPARAM]
    user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user32.IsWindowVisible.argtypes = [wintypes.HWND]
    user32.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
    startup = subprocess.STARTUPINFO()
    startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startup.wShowWindow = 0
    env = dict(os.environ, QT_QPA_PLATFORM="windows")
    process = subprocess.Popen([str(executable)], env=env, startupinfo=startup)
    deadline = time.monotonic() + timeout
    closed_window = False
    owned = []
    try:
        while process.poll() is None and time.monotonic() < deadline:
            owned = psutil.Process(process.pid).children(recursive=True)
            pids = {process.pid, *(child.pid for child in owned)}
            windows = []

            @callback_type
            def visit(hwnd, _):
                pid = wintypes.DWORD()
                user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
                if pid.value in pids and user32.IsWindowVisible(hwnd):
                    title = ctypes.create_unicode_buffer(512)
                    user32.GetWindowTextW(hwnd, title, len(title))
                    if title.value.startswith("成像数据分析工具箱 v"):
                        windows.append(hwnd)
                return True

            user32.EnumWindows(visit, 0)
            if windows:
                # Allow deferred hardware discovery to start before normal close.
                time.sleep(close_delay)
                user32.PostMessageW(windows[0], 0x0010, 0, 0)
                closed_window = True
                break
            time.sleep(0.2)
        if not closed_window:
            raise RuntimeError(f"Main window did not become ready, exit={process.poll()}")
        code = process.wait(timeout=timeout)
        if code != 0:
            raise RuntimeError(f"Package exited with code {code}")
        return {"main_window_ready": True, "normal_close_exit_code": code}
    finally:
        if process.poll() is None:
            # Only clean up the process tree started by this verification run.
            for child in reversed(owned):
                try:
                    child.terminate()
                except psutil.NoSuchProcess:
                    pass
            process.terminate()
            process.wait(timeout=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("--backend", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--close-delay", type=float, default=2)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--worker-only", action="store_true")
    mode.add_argument("--window-only", action="store_true")
    args = parser.parse_args()
    executable = args.executable.resolve(strict=True)
    cache = ROOT / ".venv" / "packaged-cupy-cache"
    os.environ["CUPY_CACHE_DIR"] = str(cache)
    result = {"executable": str(executable), "backend": args.backend}
    if not args.window_only:
        result.update(verify_worker(executable, args.backend))
    if not args.worker_only:
        result.update(verify_window(executable, args.timeout, args.close_delay))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
