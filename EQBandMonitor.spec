# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller 打包配置：生成单文件免安装 exe。

要点：
  * **onedir=False（onefile）**：用户只要一个 exe，不需要旁边一堆 dll。
    代价是首次启动要解压到临时目录（约 1~2 秒），对这类常驻小工具可接受。
  * **windowed**：不弹控制台窗口。但 `--version` 自检需要 stdout，
    PyInstaller 的 windowed 模式下 stdout 是 None —— 所以 CI 自检改用
    `--version` + 文件重定向的方式会失败；因此这里保留 console=False，
    自检改为「启动后看进程是否存活」。
  * **排除无用大件**：Qt 只用到 Core/Gui/Widgets，排除 WebEngine、
    Qml、3D、Multimedia、Pdf 等，能把体积从 ~200MB 压到 ~80MB。
  * soundcard 依赖 cffi，必须显式收集其二进制与数据。
"""
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

ROOT = Path(SPECPATH)

datas = [
    (str(ROOT / "assets" / "icon.ico"), "assets"),
    (str(ROOT / "assets" / "icon.png"), "assets"),
]
datas += collect_data_files("soundcard")

binaries = collect_dynamic_libs("soundcard")

hiddenimports = [
    "soundcard",
    "soundcard.mediafoundation",
    "_cffi_backend",
]

# 明确排除用不到的重型 Qt 模块
excludes = [
    "PySide6.QtWebEngineCore",
    "PySide6.QtWebEngineWidgets",
    "PySide6.QtWebEngineQuick",
    "PySide6.QtQml",
    "PySide6.QtQuick",
    "PySide6.QtQuick3D",
    "PySide6.Qt3DCore",
    "PySide6.Qt3DRender",
    "PySide6.QtMultimedia",
    "PySide6.QtMultimediaWidgets",
    "PySide6.QtPdf",
    "PySide6.QtPdfWidgets",
    "PySide6.QtCharts",
    "PySide6.QtDataVisualization",
    "PySide6.QtBluetooth",
    "PySide6.QtNetworkAuth",
    "PySide6.QtPositioning",
    "PySide6.QtSerialPort",
    "PySide6.QtSql",
    "PySide6.QtTest",
    "PySide6.QtDesigner",
    "PySide6.QtHelp",
    "PySide6.QtOpenGL",
    "PySide6.QtOpenGLWidgets",
    "PySide6.QtSvg",
    "PySide6.QtSvgWidgets",
    "PySide6.QtXml",
    "PySide6.QtUiTools",
    "tkinter",
    "unittest",
    "pydoc_data",
    "lib2to3",
    "matplotlib",
    "pandas",
    "IPython",
]

a = Analysis(
    [str(ROOT / "main.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
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
    name="EQBandMonitor",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(ROOT / "assets" / "icon.ico"),
    version=str(ROOT / "build" / "version_info.txt")
    if (ROOT / "build" / "version_info.txt").is_file()
    else None,
)
