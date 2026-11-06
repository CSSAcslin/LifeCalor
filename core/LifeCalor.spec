# -*- mode: python ; coding: utf-8 -*-

from importlib.metadata import PackageNotFoundError
from importlib.util import find_spec
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, copy_metadata


PROJECT_DIR = Path(SPECPATH).resolve()

datas = [
    (str(PROJECT_DIR / 'style.qss'), '.'),
    (str(PROJECT_DIR / 'appearance' / '*.qss'), 'appearance'),
    (str(PROJECT_DIR / 'HTML' / '*.html'), 'HTML'),
    (str(PROJECT_DIR / 'LifeCalor.ico'), '.'),
]
binaries = []
hiddenimports = [
    'compute.backends.cuda',
    # CuPy reaches this stdlib module through its frozen runtime path, which
    # PyInstaller cannot infer from the normal source import graph.
    'graphlib',
]


def collect_runtime_package(package_name):
    package_datas, package_binaries, package_hiddenimports = collect_all(package_name)
    datas.extend(package_datas)
    binaries.extend(package_binaries)
    hiddenimports.extend(package_hiddenimports)


if find_spec('cupy') is None:
    raise RuntimeError(
        'The full LifeCalor build requires CuPy. Install the wheel matching the '
        'target CUDA major version, or build LifeCalor-CPU.spec instead.'
    )

from cupy.cuda import get_cuda_path

# CuPy is loaded inside an isolated worker, so PyInstaller cannot discover it
# through the normal import graph. CUDA pathfinder locates the wheel-provided
# runtime libraries after the one-file bundle has been extracted.
for package in ('cupy', 'cupyx', 'cupy_backends', 'cuda.pathfinder'):
    collect_runtime_package(package)

nvidia_spec = find_spec('nvidia')
if nvidia_spec is not None:
    for package_root in nvidia_spec.submodule_search_locations or ():
        datas.append((str(Path(package_root)), 'nvidia'))

# CUDA may come from a local toolkit instead of NVIDIA's optional pip wheels.
# Mirror the toolkit layout so the frozen worker can find DLLs and JIT headers.
cuda_path = get_cuda_path()
collected_cuda_dlls = set()
if cuda_path:
    cuda_root = Path(cuda_path)
    for dll_directory in (cuda_root / 'bin' / 'x64', cuda_root / 'bin'):
        for pattern in (
            'cublas*.dll', 'cudart*.dll', 'cufft*.dll', 'curand*.dll',
            'cusolver64*.dll', 'cusparse*.dll', 'nvJitLink*.dll', 'nvrtc*.dll',
        ):
            for dll in sorted(dll_directory.glob(pattern)):
                if dll.name not in collected_cuda_dlls:
                    binaries.append((str(dll), 'bin/x64'))
                    collected_cuda_dlls.add(dll.name)
    if (cuda_root / 'include').is_dir():
        datas.append((str(cuda_root / 'include'), 'include'))

for distribution in (
    'cupy-cuda13x',
    'cuda-pathfinder',
    'cuda-toolkit',
    'nvidia-cuda-runtime',
    'nvidia-cuda-nvrtc',
    'nvidia-cublas',
    'nvidia-cufft',
    'nvidia-curand',
    'nvidia-cusolver',
    'nvidia-cusparse',
    'nvidia-nvjitlink',
):
    try:
        datas.extend(copy_metadata(distribution))
    except PackageNotFoundError:
        pass


a = Analysis(
    [str(PROJECT_DIR / 'MainWindow.py')],
    pathex=[str(PROJECT_DIR)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[str(PROJECT_DIR / 'runtime_hooks' / 'cuda_runtime.py')],
    excludes=[],
    noarchive=False,
    optimize=0,
)
# Automatic binary dependency scanning can also place NVRTC at the bundle root.
# CuPy derives CUDA's root from NVRTC's location, so retain only our toolkit layout.
a.binaries = [
    entry for entry in a.binaries
    if Path(entry[0]).name not in collected_cuda_dlls
    or Path(entry[0]).parent == Path('bin/x64')
]
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='LifeCalor',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=[str(PROJECT_DIR / 'LifeCalor.ico')],
)
