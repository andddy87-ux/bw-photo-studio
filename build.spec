# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec-файл для BW Photo Studio
# Сборка: pyinstaller build.spec

import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# Скрытые импорты (модули, которые PyInstaller не определяет автоматически)
hidden_imports = [
    # OpenCV
    'cv2',
    # NumPy
    'numpy',
    'numpy.core._methods',
    'numpy.lib.format',
    # SciPy
    'scipy.ndimage',
    'scipy.ndimage._filters',
    'scipy.ndimage._interpolation',
    'scipy.ndimage._morphology',
    'scipy.ndimage._measurements',
    'scipy._lib.messagestream',
    # PIL / Pillow
    'PIL',
    'PIL.Image',
    'PIL.ImageFilter',
    'PIL.ImageOps',
    'PIL._imaging',
    # pillow-heif
    'pillow_heif',
    # rawpy
    'rawpy',
    # PyQt5
    'PyQt5',
    'PyQt5.QtWidgets',
    'PyQt5.QtCore',
    'PyQt5.QtGui',
    'PyQt5.sip',
    # imageio
    'imageio',
    'imageio.plugins',
    # Наши модули
    'core',
    'core.image_io',
    'core.bw_algorithms',
    'core.sepia',
    'core.editor',
    'core.infrared',
    'ui',
    'ui.styles',
    'ui.image_viewer',
    'ui.panels',
    'ui.main_window',
]

# Дополнительные файлы данных
datas = []

try:
    datas += collect_data_files('rawpy')
except Exception:
    pass

try:
    datas += collect_data_files('pillow_heif')
except Exception:
    pass

try:
    datas += collect_data_files('cv2')
except Exception:
    pass

a = Analysis(
    ['main.py'],
    pathex=['.'],
    binaries=[],
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'matplotlib',
        'IPython',
        'jupyter',
        'notebook',
        'pandas',
        'PyQt5.QtWebEngine',
        'PyQt5.QtWebEngineWidgets',
        'PyQt5.QtBluetooth',
        'PyQt5.QtNfc',
        'PyQt5.QtSensors',
        'PyQt5.QtSerialPort',
        'PyQt5.QtSql',
        'PyQt5.QtTest',
        'PyQt5.QtXml',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='BW_Photo_Studio',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,          # Без консольного окна
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # icon='icon.ico',      # Раскомментируйте и добавьте icon.ico для своей иконки
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[
        'vcruntime140.dll',
        'python3*.dll',
        'Qt5*.dll',
    ],
    name='BW_Photo_Studio',
)
