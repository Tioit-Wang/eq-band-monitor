"""自测：入口连通性（wiring）。

为什么需要这个文件 —— 一次真实事故：
  `main.py` 的 `_on_hotkey` 引用了未导入的 `SCENE_BY_HOTKEY`，
  于是**每一个全局热键**（F1~F11、Ctrl+Shift+*）都在回调第一行抛 NameError。
  PySide6 对槽函数异常只打 traceback 不中止进程，所以表现是
  「热键注册成功、程序正常运行、但按了完全没反应」。

  当时 156 项自测全绿、CI 全绿、Release 照常发布。
  原因是：没有任何一项测试 `import main` —— 所有测试都在验证"算得对不对"，
  没人验证"按键有没有接上"。

本文件堵这个洞：
  [1] 静态扫描：Load 上下文但未导入/未定义的名字（check_undefined.py）
  [2] 打桩后逐个调用 HotkeyManager 注册的每一个名字，不许抛异常
  [3] 每个场景热键都能被 _parse_key 解析成有效 VK
  [4] 上报的菜单/热键名单与注册名单一致（防止加了键忘了注册）
"""
from __future__ import annotations

import ast
import io
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent))

import check_undefined  # noqa: E402
from app.hotkeys import HotkeyManager  # noqa: E402
from app.recorder import SCENE_BY_KEY, SCENES  # noqa: E402

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


print("=" * 66)
print("EQ Band Monitor — 入口连通性 自测")
print("=" * 66)

ROOT = Path(__file__).resolve().parent

# ---------- 1. 静态扫描未定义名字 ----------
print("\n[1] 静态扫描：未定义的名字")
targets = [ROOT / "main.py"] + sorted((ROOT / "app").glob("*.py"))
targets += sorted(ROOT.glob("selftest_*.py")) + sorted((ROOT / "tools").glob("*.py"))
problems: list[str] = []
for p in targets:
    for line, name in check_undefined.undefined_in(p):
        problems.append(f"{p.relative_to(ROOT)}:{line} '{name}'")
check("所有模块无未定义名字（能挡住 NameError 类事故）",
      not problems, "; ".join(problems[:5]))

# ---------- 2. 打桩调用每个热键名 ----------
print("\n[2] 逐个调用热键回调，不许抛异常")


class FakeAnalyzer:
    def reset_peak_hold(self) -> None:
        pass


class FakeRawRec:
    clips: dict = {}
    current_key = None
    out_dir = "."

    def stop(self):
        return None

    def clear(self) -> None:
        pass

    def is_recording(self) -> bool:
        return False


class FakeOverlay:
    DEFAULT_POS = (24, 24)

    def is_edit_mode(self) -> bool:
        return False

    def set_edit_mode(self, *a, **k) -> None:
        pass

    def isVisible(self) -> bool:  # noqa: N802
        return True

    def setVisible(self, v) -> None:  # noqa: N802
        pass

    def move(self, *a) -> None:
        pass

    def raise_(self) -> None:
        pass


class FakeTray:
    def setToolTip(self, t) -> None:  # noqa: N802
        pass

    def showMessage(self, *a, **k) -> None:  # noqa: N802
        pass


def make_stub_app():
    """构造一个足以跑通 _on_hotkey 的假 MonitorApp。"""
    import main as M

    app = SimpleNamespace()
    app.state = SimpleNamespace(
        analyzer=FakeAnalyzer(),
        is_recording=False,
        record_scene="",
        footer_hint="",
        gait_label="正常走",
        scene_summary="",
    )
    app.raw_rec = FakeRawRec()
    app.overlay = FakeOverlay()
    app.tray = FakeTray()
    app.gait = "walk"
    app._notify = lambda t: None
    app._show_advice = lambda: None
    app._clear_recordings = lambda: None
    app._cycle_range = lambda: None
    app._toggle_overlay_edit = lambda: None
    app._reset_overlay_pos = lambda: None
    app.shutdown = lambda: None
    app._toggle_record = lambda k: None
    app._set_gait = lambda g: None
    return M, app


# 与 _register_hotkeys 里 extra 列表保持一致（新增功能键时要同步）
FUNC_HOTKEY_NAMES = [
    "gait_sneak",
    "gait_walk",
    "gait_run",
    "show_advice",
    "clear_rec",
    "edit_overlay",
    "reset_pos",
    "reset_peak",
    "toggle_range",
    "hide",
    "quit",
]

try:
    M, stub = make_stub_app()
    all_names = [s.key for s in SCENES] + FUNC_HOTKEY_NAMES
    raised: list[tuple[str, str]] = []
    for n in all_names:
        try:
            M.MonitorApp._on_hotkey(stub, n)
        except Exception as exc:  # noqa: BLE001
            raised.append((n, f"{type(exc).__name__}: {exc}"))
    check(f"{len(all_names)} 个热键回调全部无异常",
          not raised, "; ".join(f"{n}->{e}" for n, e in raised[:4]))
except Exception as exc:  # noqa: BLE001
    check("能构造 MonitorApp 的桩并调用回调", False, f"{type(exc).__name__}: {exc}")

# ---------- 3. 热键字符串都能解析成 VK ----------
print("\n[3] 热键字符串可解析")
from app.hotkeys import VK  # noqa: E402

bad_keys = []
for s in SCENES:
    if HotkeyManager._parse_key(s.hotkey) is None:
        bad_keys.append(s.hotkey)
for k in ("1", "2", "3", "F10", "F11", "R", "D", "H", "Q", "E", "X"):
    if HotkeyManager._parse_key(k) is None:
        bad_keys.append(k)
check("全部热键字符串可解析为 VK", not bad_keys, ", ".join(bad_keys))

print("\n[4] 场景表与热键查表一致")
check("SCENE_BY_KEY 覆盖全部场景（_on_hotkey 查的就是它）",
      all(SCENE_BY_KEY.get(s.key) is s for s in SCENES),
      f"{[s.key for s in SCENES if SCENE_BY_KEY.get(s.key) is not s]}")

# 注册名必须与 _on_hotkey 能识别的名字对得上
check("场景 key 都能被 _on_hotkey 识别（不会静默落到 else 分支）",
      len(SCENE_BY_KEY) == len(SCENES), f"{len(SCENE_BY_KEY)} vs {len(SCENES)}")
# ---------- 5. _register_hotkeys 的功能键名单与测试名单同步 ----------
print("\n[5] 注册名单与测试名单同步")
src = io.open(ROOT / "main.py", encoding="utf-8-sig").read()
tree = ast.parse(src)
registered: list[str] = []
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name == "_register_hotkeys":
        for sub in ast.walk(node):
            if (isinstance(sub, ast.Assign)
                    and any(getattr(t, "id", "") == "extra" for t in sub.targets)
                    and isinstance(sub.value, ast.List)):
                for elt in sub.value.elts:
                    if (isinstance(elt, ast.Tuple) and elt.elts
                            and isinstance(elt.elts[0], ast.Constant)):
                        registered.append(elt.elts[0].value)
missing = [n for n in registered if n not in FUNC_HOTKEY_NAMES]
extra = [n for n in FUNC_HOTKEY_NAMES if n not in registered]
check("注册的功能键都已纳入连通性测试",
      not missing, f"未测: {missing}")
check("测试名单里没有已删除的功能键", not extra, f"多余: {extra}")

print("\n" + "=" * 66)
if FAILS:
    print(f"结果：{len(FAILS)} 项失败 → {', '.join(FAILS)}")
    sys.exit(1)
print("结果：全部通过 ✓")
print("=" * 66)
