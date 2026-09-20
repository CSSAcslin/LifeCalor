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

# CuPy is loaded inside an isolated worker, so PyInstaller cannot discover it
# through the normal import graph. CUDA pathfinder locates the wheel-provided
# runtime libraries after the one-file bundle has been extracted.
for package in ('cupy', 'cupyx', 'cupy_backends', 'cuda.pathfinder'):
    collect_runtime_package(package)

nvidia_spec = find_spec('nvidia')
if nvidia_spec is not None:
    for package_root in nvidia_spec.submodule_search_locations or ():
        datas.append((str(Path(package_root)), 'nvidia'))

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
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
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
