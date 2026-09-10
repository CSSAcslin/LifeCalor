"""Repeatable data-path benchmark for LifeCalor development builds."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pywt
import tifffile
from scipy import signal

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from ArrayCache import ArrayCacheConfig, ArrayStore
from display.renderer import FrameRenderer
from importing import default_importer_registry
from performance import PerformanceRecorder


SIZES = {
    "small": (32, 128, 128),
    "medium": (128, 256, 256),
    "large": (256, 512, 512),
}


def run(size_name):
    shape = SIZES[size_name]
    rng = np.random.default_rng(20260909)
    source = rng.normal(size=shape).astype(np.float32)
    rows = []
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / f"{size_name}.npy"
        started = time.perf_counter()
        np.save(path, source, allow_pickle=False)
        rows.append({"operation": "npy_write", "seconds": time.perf_counter() - started})

        registry = default_importer_registry()
        with PerformanceRecorder("npy_import", output_bytes=source.nbytes) as recorder:
            payload = registry.read("npy", path, {"time_step": 1.0})
        rows.append({"operation": "npy_import", "seconds": recorder.metrics.elapsed_seconds})

        with PerformanceRecorder("first_frame_render") as recorder:
            FrameRenderer.render(payload.display_data[0])
        rows.append({"operation": "first_frame_render", "seconds": recorder.metrics.elapsed_seconds})

        with PerformanceRecorder("middle_frame_render") as recorder:
            FrameRenderer.render(payload.display_data[len(payload.display_data) // 2])
        rows.append({"operation": "middle_frame_render", "seconds": recorder.metrics.elapsed_seconds})

        store = ArrayStore(ArrayCacheConfig(Path(directory) / "cache", threshold_bytes=1))
        with PerformanceRecorder("cache_write", output_bytes=source.nbytes) as recorder:
            ref = store.put_array(payload.data, "benchmark", "data")
        rows.append({"operation": "cache_write", "seconds": recorder.metrics.elapsed_seconds})
        with PerformanceRecorder("history_cache_restore", output_bytes=source.nbytes) as recorder:
            restored = store.load_ref(ref)
        rows.append({"operation": "history_cache_restore", "seconds": recorder.metrics.elapsed_seconds})
        if restored.shape != source.shape:
            raise RuntimeError("cache restore shape mismatch")

        sample = payload.data.reshape(payload.data.shape[0], -1).T[:256]
        window = min(32, sample.shape[-1])
        with PerformanceRecorder("stft_sample") as recorder:
            signal.stft(sample, nperseg=window, noverlap=max(0, window // 2), axis=-1)
        rows.append({"operation": "stft_sample", "seconds": recorder.metrics.elapsed_seconds})
        with PerformanceRecorder("cwt_sample") as recorder:
            pywt.cwt(sample, np.arange(1, 17), "morl", axis=-1)
        rows.append({"operation": "cwt_sample", "seconds": recorder.metrics.elapsed_seconds})

        export_path = Path(directory) / "export.tif"
        with PerformanceRecorder("tiff_export", output_bytes=source.nbytes) as recorder:
            tifffile.imwrite(export_path, payload.data, photometric="minisblack")
        rows.append({"operation": "tiff_export", "seconds": recorder.metrics.elapsed_seconds})

    return {"size": size_name, "shape": shape, "dtype": str(source.dtype), "bytes": source.nbytes, "operations": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=SIZES, default="small")
    args = parser.parse_args()
    print(json.dumps(run(args.size), ensure_ascii=False, indent=2))