"""用户设置的持久化：把悬浮窗位置等少量偏好存成 JSON。

为什么不用 QSettings：
  * QSettings 在 Windows 上写注册表，卸载后留残留、排查问题也不直观；
  * 我们的需求极小（几个数值），一个可读的 JSON 更透明 ——
    用户遇到问题可以直接打开看，甚至手删重置。

写入用「临时文件 + 原子替换」，避免程序被杀时留下半截文件导致读不出来。
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from . import diagnostics

FILENAME = "settings.json"


def settings_path() -> Path:
    return diagnostics.app_data_dir() / FILENAME


def load() -> dict[str, Any]:
    """读取全部设置。文件损坏时返回空字典而不是抛异常。"""
    p = settings_path()
    if not p.is_file():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        diagnostics.log(f"读取设置失败（已忽略）: {exc}")
        return {}


def save(data: dict[str, Any]) -> None:
    """原子写入。失败只记日志，绝不影响主流程。"""
    p = settings_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=".settings-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
            os.replace(tmp, p)
        finally:
            if os.path.exists(tmp):
                try:
                    os.unlink(tmp)
                except Exception:
                    pass
    except Exception as exc:
        diagnostics.log(f"写入设置失败（已忽略）: {exc}")


def get(key: str, default: Any = None) -> Any:
    return load().get(key, default)


def put(key: str, value: Any) -> None:
    """改单个键并保存（读-改-写）。"""
    data = load()
    data[key] = value
    save(data)


def forget(key: str) -> None:
    data = load()
    if key in data:
        del data[key]
        save(data)
