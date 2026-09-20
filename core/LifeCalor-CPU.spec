# CPU-only distribution derived from LifeCalor.spec. Keep the full spec unchanged.
# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['MainWindow.py'],
    pathex=[],
    binaries=[],
    datas=[('style.qss', '.'),
    ('HTML/*.html', 'HTML')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'cupy',
        'cupyx',
        'cupy_backends',
        'compute.backends.cuda',
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
    icon=['LifeCalor.ico'],
)
