"""全局热键：Win32 RegisterHotKey + Qt 原生事件过滤器。

为什么不用第三方库：RegisterHotKey 是系统级标准 API，不需要钩子、不注入，
反作弊不会因此判定异常。
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes

from PySide6.QtCore import QAbstractNativeEventFilter, QObject, Signal

MOD_ALT = 0x0001
MOD_CONTROL = 0x0002
MOD_SHIFT = 0x0004
MOD_NOREPEAT = 0x4000
WM_HOTKEY = 0x0312

# 虚拟键码
VK = {
    "F1": 0x70, "F2": 0x71, "F3": 0x72, "F4": 0x73, "F5": 0x74, "F6": 0x75,
    "F7": 0x76, "F8": 0x77, "F9": 0x78, "F10": 0x79, "F11": 0x7A, "F12": 0x7B,
    "H": 0x48, "Q": 0x51, "R": 0x52, "E": 0x45,
}


class HotkeyManager(QObject, QAbstractNativeEventFilter):
    """注册若干全局热键，触发时发出 `triggered(name)`。"""

    triggered = Signal(str)

    def __init__(self, parent=None) -> None:
        QObject.__init__(self, parent)
        QAbstractNativeEventFilter.__init__(self)
        self._by_id: dict[int, str] = {}
        self._next_id = 1
        self._registered: list[int] = []
        from PySide6.QtWidgets import QApplication

        self._app = QApplication.instance()
        if self._app is not None:
            self._app.installNativeEventFilter(self)

    def register(self, name: str, key: str, ctrl=False, shift=False, alt=False) -> bool:
        vk = self._parse_key(key)
        if vk is None:
            return False
        mods = MOD_NOREPEAT
        if ctrl:
            mods |= MOD_CONTROL
        if shift:
            mods |= MOD_SHIFT
        if alt:
            mods |= MOD_ALT

        hk_id = self._next_id
        self._next_id += 1
        ok = bool(ctypes.windll.user32.RegisterHotKey(None, hk_id, mods, vk))
        if ok:
            self._by_id[hk_id] = name
            self._registered.append(hk_id)
        return ok

    @staticmethod
    def _parse_key(key: str) -> int | None:
        k = key.upper().strip()
        if k in VK:
            return VK[k]
        if len(k) == 1 and k.isalnum():
            return ord(k)
        return None

    def unregister_all(self) -> None:
        for hk_id in self._registered:
            try:
                ctypes.windll.user32.UnregisterHotKey(None, hk_id)
            except Exception:
                pass
        self._registered.clear()
        self._by_id.clear()

    def nativeEventFilter(self, eventType, message):  # noqa: N802
        try:
            if eventType == b"windows_generic_MSG" or eventType == "windows_generic_MSG":
                msg = wintypes.MSG.from_address(int(message))
                if msg.message == WM_HOTKEY:
                    name = self._by_id.get(int(msg.wParam))
                    if name:
                        self.triggered.emit(name)
                        return True, 0
        except Exception:
            pass
        return False, 0
