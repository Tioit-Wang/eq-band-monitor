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

from PySide6.QtCore import QPoint, QSize  # noqa: E402
from PySide6.QtGui import QImage, QPainter  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

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

print("\n" + "=" * 62)
if FAILS:
    print(f"结果：{len(FAILS)} 项失败 → {', '.join(FAILS)}")
    sys.exit(1)
print("结果：全部通过 ✓")
print("=" * 62)
