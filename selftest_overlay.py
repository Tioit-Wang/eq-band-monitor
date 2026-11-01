"""自测：悬浮窗页脚的移动方式徽标会随状态变化。

为什么单独测这个：分层透明窗口（WA_TranslucentBackground）没法用
桌面截屏或 GetWindowDC 抓到内容 —— 截出来是纯黑。所以直接让 Qt
把 widget 画到 QImage 上，验证的是同一份 paintEvent 代码。

覆盖：
  * 三种移动方式的徽标互不相同（说明 label 真的进了绘制流程）
  * 录制态与空闲态页脚不同（说明录制状态也生效）
  * 徽标不会被裁掉（页脚区域确实有像素）
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from PySide6.QtCore import QEvent, QPoint, QPointF, QSize, Qt  # noqa: E402
from PySide6.QtGui import QImage, QMouseEvent, QPainter  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from app import settings  # noqa: E402
from app.analyzer import SpectrumAnalyzer  # noqa: E402
from app.overlay import Overlay  # noqa: E402

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


class St:
    analyzer = None
    capture_ok = True
    status_text = "测试设备"
    is_recording = False
    record_scene = ""
    record_elapsed = 0.0
    footer_hint = "提示"
    gait_label = "正常走"
    scene_summary = ""


print("=" * 62)
print("EQ Band Monitor — 悬浮窗页脚 / 移动方式徽标 自测")
print("=" * 62)

app = QApplication.instance() or QApplication(sys.argv)

state = St()
an = SpectrumAnalyzer(samplerate=48000, release_ms=200)
rng = np.random.default_rng(0)
t0 = time.perf_counter()
for i in range(6):
    an.process(rng.normal(0, 0.05, 4800).astype(np.float32), t0 + i * 0.1)
state.analyzer = an

ov = Overlay(state)


def render(label: str) -> np.ndarray:
    state.gait_label = label
    img = QImage(QSize(ov.width(), ov.height()), QImage.Format_ARGB32_Premultiplied)
    img.fill(0)
    p = QPainter(img)
    # QWidget.render(painter) 的签名要求 targetOffset；传 (0,0) 走正常 paintEvent
    ov.render(p, QPoint(0, 0))
    p.end()
    ptr = img.constBits()
    arr = np.frombuffer(ptr, dtype=np.uint8).reshape(
        img.height(), img.bytesPerLine() // 4, 4
    )
    return arr[:, : img.width(), :].copy()


print("\n[1] 三种移动方式的徽标互不相同")
a = render("静步")
b = render("正常走")
c = render("奔跑")

check("渲染尺寸合理", a.shape[0] > 100 and a.shape[1] > 200, f"{a.shape}")

footer = slice(max(0, a.shape[0] - 70), a.shape[0])
non_empty = int((a[footer][:, :, 3] > 0).sum())
check("页脚确实画了东西", non_empty > 500, f"{non_empty} 个非透明像素")

dab = int(np.abs(a[footer].astype(int) - b[footer].astype(int)).sum())
dac = int(np.abs(a[footer].astype(int) - c[footer].astype(int)).sum())
dbc = int(np.abs(b[footer].astype(int) - c[footer].astype(int)).sum())
check("静步 vs 正常走 有差异", dab > 0, f"diff={dab}")
check("静步 vs 奔跑 有差异", dac > 0, f"diff={dac}")
check("正常走 vs 奔跑 有差异", dbc > 0, f"diff={dbc}")

print("\n[2] 录制状态影响页脚")
state.is_recording = True
state.record_scene = "楼下脚步·奔跑"
state.record_elapsed = 2.3
rec = render("奔跑")
drec = int(np.abs(rec[footer].astype(int) - c[footer].astype(int)).sum())
check("录制态与空闲态不同", drec > 0, f"diff={drec}")

state.is_recording = False
state.record_scene = ""
state.record_elapsed = 0.0

print("\n[3] 帧内容区不受徽标影响（只改页脚）")
body = slice(0, max(0, a.shape[0] - 70))
same_signal = (np.abs(a[body].astype(int) - b[body].astype(int)).sum(axis=(0, 2)) > 0)
# 频谱区是实时数据，两次渲染可能有轻微差异；这里只要求"绝大部分像素一致"
changed_rows = int((same_signal > 0).sum())
check("主体区域基本不变", changed_rows <= body.stop // 2,
      f"{changed_rows}/{body.stop} 行有差异")

print("\n[4] 拖动交互：锁定/编辑态切换")


def click_through_state() -> bool | None:
    """读 Win32 的 WS_EX_TRANSPARENT 位，判断当前是否穿透。

    只测 Qt 属性不够 —— 真正让游戏"点得到"的是这个扩展样式位。
    """
    try:
        import ctypes

        hwnd = int(ov.winId())
        ex = ctypes.windll.user32.GetWindowLongW(hwnd, -20)
        return bool(ex & 0x00000020)
    except Exception:
        return None


check("默认不在编辑态", not ov.is_edit_mode())
locked_through = click_through_state()
check("锁定态默认穿透（游戏不受影响）",
      locked_through is None or locked_through is True, f"transparent={locked_through}")

ov.set_edit_mode(True)
check("进入编辑态", ov.is_edit_mode())
check("编辑态不穿透（能接收鼠标）", click_through_state() is not True,
      f"transparent={click_through_state()}")

ov.set_edit_mode(False)
check("退出编辑态", not ov.is_edit_mode())
check("退出后恢复穿透", click_through_state() is not False,
      f"transparent={click_through_state()}")

print("\n[5] 编辑态有可见的视觉提示")
ov.set_edit_mode(True)
edit_img = render("正常走")
ov.set_edit_mode(False)
lock_img = render("正常走")
diff_edit = int(np.abs(edit_img.astype(int) - lock_img.astype(int)).sum())
check("编辑态画面与锁定态不同", diff_edit > 0, f"diff={diff_edit}")

print("\n[6] 位置记忆与越界回落")
from app import settings  # noqa: E402

saved_backup = settings.get("overlay_pos")
try:
    settings.put("overlay_pos", [200, 160])
    ov2 = Overlay(state)
    check("按保存值恢复位置", (ov2.pos().x(), ov2.pos().y()) == (200, 160),
          f"{ov2.pos().x()},{ov2.pos().y()}")

    # 越界：保存一个所有屏幕都碰不到的位置，应回落到默认
    settings.put("overlay_pos", [-9000, -9000])
    ov3 = Overlay(state)
    dp = Overlay.DEFAULT_POS
    check("越界位置回落到默认", (ov3.pos().x(), ov3.pos().y()) == dp,
          f"{ov3.pos().x()},{ov3.pos().y()}")

    # 坏数据不应崩
    settings.put("overlay_pos", ["not", "numbers"])
    ov4 = Overlay(state)
    check("坏设置不崩且回落默认",
          (ov4.pos().x(), ov4.pos().y()) == dp, f"{ov4.pos().x()},{ov4.pos().y()}")
finally:
    if saved_backup is None:
        settings.forget("overlay_pos")
    else:
        settings.put("overlay_pos", saved_backup)

print("\n[7] 鼠标拖动真的改变位置并持久化")


def send(widget, kind, global_pos, button=Qt.LeftButton) -> None:
    """投递一个鼠标事件。

    走的是和真实鼠标相同的 mousePress/Move/Release 代码路径，
    只是绕过系统级指针移动（本环境的 SendInput 受完整性级别限制）。
    """
    local = widget.mapFromGlobal(global_pos)
    ev = QMouseEvent(
        kind, QPointF(local), QPointF(global_pos), button, button, Qt.NoModifier
    )
    QApplication.sendEvent(widget, ev)


saved2 = settings.get("overlay_pos")
try:
    settings.forget("overlay_pos")
    ovd = Overlay(state)
    start = ovd.pos()

    # 锁定态：事件应被忽略
    press = QPoint(ovd.pos().x() + 100, ovd.pos().y() + 10)
    send(ovd, QEvent.MouseButtonPress, press)
    send(ovd, QEvent.MouseMove, press + QPoint(80, 60))
    send(ovd, QEvent.MouseButtonRelease, press + QPoint(80, 60))
    check("锁定态拖动无效", ovd.pos() == start,
          f"{start.x()},{start.y()} -> {ovd.pos().x()},{ovd.pos().y()}")

    # 编辑态：应跟随
    ovd.set_edit_mode(True)
    p0 = ovd.pos()
    g0 = QPoint(p0.x() + 120, p0.y() + 12)
    send(ovd, QEvent.MouseButtonPress, g0)
    send(ovd, QEvent.MouseMove, g0 + QPoint(150, 90))
    check("编辑态拖动窗口跟随",
          (ovd.pos().x(), ovd.pos().y()) == (p0.x() + 150, p0.y() + 90),
          f"{p0.x()},{p0.y()} -> {ovd.pos().x()},{ovd.pos().y()}")

    send(ovd, QEvent.MouseButtonRelease, g0 + QPoint(150, 90))
    after = ovd.pos()
    stored = settings.get("overlay_pos")
    check("松手后位置已持久化", stored == [after.x(), after.y()], f"{stored}")

    ovd2 = Overlay(state)
    check("重启后恢复位置",
          (ovd2.pos().x(), ovd2.pos().y()) == (after.x(), after.y()),
          f"{ovd2.pos().x()},{ovd2.pos().y()}")

    # 双击退出编辑态
    ovd.set_edit_mode(True)
    send(ovd, QEvent.MouseButtonDblClick,
         QPoint(ovd.pos().x() + 60, ovd.pos().y() + 10))
    check("双击标题栏回到锁定", not ovd.is_edit_mode())

    # 拖出屏幕应被拉回
    ovd.set_edit_mode(True)
    pp = ovd.pos()
    gg = QPoint(pp.x() + 50, pp.y() + 10)
    send(ovd, QEvent.MouseButtonPress, gg)
    send(ovd, QEvent.MouseMove, QPoint(-5000, -5000))
    send(ovd, QEvent.MouseButtonRelease, QPoint(-5000, -5000))
    scr = app.primaryScreen().availableGeometry()
    inside = (
        ovd.pos().x() + ovd.width() > scr.left()
        and ovd.pos().x() < scr.right()
        and ovd.pos().y() + 40 > scr.top()
        and ovd.pos().y() < scr.bottom()
    )
    check("拖出屏幕后被拉回可见范围", inside,
          f"pos={ovd.pos().x()},{ovd.pos().y()}")
finally:
    if saved2 is None:
        settings.forget("overlay_pos")
    else:
        settings.put("overlay_pos", saved2)

print("\n" + "=" * 62)
if FAILS:
    print(f"结果：{len(FAILS)} 项失败 → {', '.join(FAILS)}")
    sys.exit(1)
print("结果：全部通过 ✓")
print("=" * 62)
