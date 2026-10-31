"""运行时资源定位：兼容源码运行与 PyInstaller 打包（onefile/onedir）。"""

from __future__ import annotations

import sys
from pathlib import Path


def resource_root() -> Path:
    """返回资源根目录。

    PyInstaller 会把 `--add-data` 的内容解到 `sys._MEIPASS`（onefile 是
    临时解压目录，onedir 是 `_internal`）。源码运行时就是项目根目录。
    """
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        return Path(meipass)
    return Path(__file__).resolve().parent.parent


def asset(*parts: str) -> Path:
    """拼接资源路径，例如 asset("icon.ico")。"""
    return resource_root() / "assets" / Path(*parts)
