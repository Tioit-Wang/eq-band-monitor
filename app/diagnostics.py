"""运行时诊断：崩溃日志 + 单实例保护。

为什么需要：
  * **windowed（无控制台）exe 里 print/traceback 无处可去**，
    出问题表现为"启动后无声消失"，完全无法排查。
    所以把 stdout/stderr、未捕获异常、Qt 自身警告统一写进日志文件。
  * **单实例**：这是一款常驻托盘 + 全局热键的工具。启动两份时第二份
    会因为热键被占用而"看起来能跑但按了没反应"，用户很难理解。
    与其让它半残运行，不如直接把已有窗口唤出来。

日志位置：%LOCALAPPDATA%\\EQBandMonitor\\app.log（超过 1MB 时轮转一次）
"""

from __future__ import annotations

import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

APP_DIR_NAME = "EQBandMonitor"
_MUTEX_NAME = "Global\\EQBandMonitor_SingleInstance_v1"
_mutex_handle = None


def app_data_dir() -> Path:
    """用户数据目录：日志、崩溃记录都放这里。"""
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    d = Path(base) / APP_DIR_NAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def log_path() -> Path:
    return app_data_dir() / "app.log"


def _rotate(max_bytes: int = 1_000_000) -> None:
    p = log_path()
    try:
        if p.is_file() and p.stat().st_size > max_bytes:
            backup = p.with_suffix(".1.log")
            if backup.is_file():
                backup.unlink()
            p.rename(backup)
    except Exception:
        pass


def log(msg: str) -> None:
    """追加一行日志。绝不抛异常 —— 日志本身不能成为故障源。"""
    try:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_path(), "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {msg}\n")
    except Exception:
        pass


def _excepthook(exc_type, exc, tb) -> None:
    try:
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        log("未捕获异常:\n" + text)
    except Exception:
        pass


def install() -> None:
    """安装日志重定向与异常钩子。应在 QApplication 之前调用。"""
    _rotate()

    # 打包成 windowed 后 sys.stdout/stderr 是 None，print 会静默丢弃，
    # 这里换成一个写日志的流，让一切输出都留下痕迹。
    class _Stream:
        def __init__(self, tag: str) -> None:
            self.tag = tag

        def write(self, s: str) -> int:
            if s and s.strip():
                log(f"{self.tag}: {s.rstrip()}")
            return len(s)

        def flush(self) -> None:
            pass

        def isatty(self) -> bool:
            return False

    if getattr(sys, "frozen", False):
        if sys.stdout is None:
            sys.stdout = _Stream("OUT")
        if sys.stderr is None:
            sys.stderr = _Stream("ERR")

    sys.excepthook = _excepthook
    log(f"=== 启动 pid={os.getpid()} frozen={getattr(sys, 'frozen', False)} ===")


def install_qt_message_handler() -> None:
    """把 Qt 自己的警告也导进日志（很多崩溃前会先有 Qt 警告）。"""
    try:
        from PySide6.QtCore import QtMsgType, qInstallMessageHandler

        levels = {
            QtMsgType.QtDebugMsg: "DEBUG",
            QtMsgType.QtInfoMsg: "INFO",
            QtMsgType.QtWarningMsg: "WARN",
            QtMsgType.QtCriticalMsg: "CRIT",
            QtMsgType.QtFatalMsg: "FATAL",
        }

        def handler(mode, context, message):  # noqa: ANN001
            lvl = levels.get(mode, str(mode))
            if lvl in ("WARN", "CRIT", "FATAL"):
                log(f"Qt[{lvl}] {message}")

        qInstallMessageHandler(handler)
    except Exception:
        pass


def acquire_single_instance() -> bool:
    """尝试成为唯一实例。返回 True=本进程是第一个，False=已有实例在跑。"""
    global _mutex_handle
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateMutexW.restype = wintypes.HANDLE
        handle = kernel32.CreateMutexW(None, True, _MUTEX_NAME)
        if not handle:
            return True  # 拿不到互斥体就不要阻止启动
        already = ctypes.get_last_error() == 183  # ERROR_ALREADY_EXISTS
        _mutex_handle = handle
        return not already
    except Exception:
        return True


def focus_existing_window(title_hint: str = "EQ Band Monitor") -> None:
    """把已运行实例的窗口调到前台（找窗口 + ShowWindow + SetForegroundWindow）。"""
    try:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.WinDLL("user32", use_last_error=True)
        found: list[int] = []

        @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
        def enum_proc(hwnd, _lparam):  # noqa: ANN001
            length = user32.GetWindowTextLengthW(hwnd)
            if length:
                buf = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(hwnd, buf, length + 1)
                if title_hint.lower() in buf.value.lower():
                    found.append(int(hwnd))
            return True

        user32.EnumWindows(enum_proc, 0)
        for hwnd in found:
            user32.ShowWindow(hwnd, 9)          # SW_RESTORE
            user32.SetForegroundWindow(hwnd)
    except Exception:
        pass
