"""Expose the bundled CUDA toolkit to the frozen application's workers."""

import os
import sys
from pathlib import Path


bundle_root = Path(sys._MEIPASS)
cuda_bin = bundle_root / "bin" / "x64"
if cuda_bin.is_dir() and (bundle_root / "include").is_dir():
    os.environ["CUDA_PATH"] = str(bundle_root)
    os.environ["PATH"] = str(cuda_bin) + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        # Retain the handle for the process lifetime, including CUDA worker JIT.
        sys._lifecalor_cuda_dll_directory = os.add_dll_directory(str(cuda_bin))
