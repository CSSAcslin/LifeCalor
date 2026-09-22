# CPU-only distribution derived from LifeCalor.spec.
# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path


PROJECT_DIR = Path(SPECPATH).resolve()

datas = [
    (str(PROJECT_DIR / 'style.qss'), '.'),
    (str(PROJECT_DIR / 'appearance' / '*.qss'), 'appearance'),
    (str(PROJECT_DIR / 'HTML' / '*.html'), 'HTML'),
    (str(PROJECT_DIR / 'LifeCalor.ico'), '.'),
]


a = Analysis(
    [str(PROJECT_DIR / 'MainWindow.py')],
    pathex=[str(PROJECT_DIR)],
    binaries=[],
    datas=datas,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'cupy',
        'cupyx',
        'cupy_backends',
        'compute.backends.cuda',
        'compute.backends.lifetime_cuda',
        'cuda',
        'cuda.pathfinder',
        'nvidia',
        'pynvml',
    ],
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
    name='LifeCalor-CPU',
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
