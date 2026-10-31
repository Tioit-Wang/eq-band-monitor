"""生成应用图标 assets/icon.ico 与 assets/icon.png。

用 QPainter 画一个频谱柱状图标，与应用内托盘图标同一视觉语言。
输出多尺寸 ICO（16/24/32/48/64/128/256），供 exe 嵌入与窗口使用。
"""
from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPixmap
from PySide6.QtWidgets import QApplication

OUT = Path(__file__).parent / "assets"
OUT.mkdir(exist_ok=True)

ACCENT = QColor("#22d3ee")
BG_TOP = QColor("#0d1117")
BG_BOT = QColor("#161b22")

OUT = Path(__file__).resolve().parent.parent / "assets"
OUT.mkdir(parents=True, exist_ok=True)

SIZES = (256, 128, 64, 48, 32, 24)

app = QApplication.instance() or QApplication(sys.argv)


def draw(size: int) -> QPixmap:
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing, True)

    # 圆角底板
    p.setPen(Qt.NoPen)
    from PySide6.QtGui import QLinearGradient

    grad = QLinearGradient(0, 0, 0, size)
    grad.setColorAt(0.0, BG_TOP)
    grad.setColorAt(1.0, BG_BOT)
    p.setBrush(grad)
    r = size * 0.22
    p.drawRoundedRect(QRectF(0, 0, size, size), r, r)

    # 频谱柱：5 根，中间高、两端低，像实时频谱
    heights = [0.30, 0.55, 0.80, 0.62, 0.40]
    n = len(heights)
    bar_w = size * 0.085
    gap = size * 0.055
    total = n * bar_w + (n - 1) * gap
    x0 = (size - total) / 2
    base = size * 0.78

    for i, h in enumerate(heights):
        x = x0 + i * (bar_w + gap)
        bh = size * h * 0.62
        rect = QRectF(x, base - bh, bar_w, bh)
        # 高度越高越亮
        c = QColor(ACCENT)
        c.setAlphaF(0.55 + 0.45 * h)
        p.setBrush(c)
        p.drawRoundedRect(rect, bar_w * 0.3, bar_w * 0.3)

    # 底部一条基线
    p.setBrush(QColor("#30363d"))
    p.drawRoundedRect(
        QRectF(x0, base + size * 0.03, total, max(1.0, size * 0.018)),
        size * 0.01,
        size * 0.01,
    )
    p.end()
    return pm


# PNG（256，供窗口/文档使用）
draw(256).save(str(OUT / "icon.png"), "PNG")

# ICO：Qt 的 ICO 写入器支持多尺寸
pm0 = draw(256)
ok = pm0.save(str(OUT / "icon.ico"), "ICO")
print("ico written:", ok, (OUT / "icon.ico").stat().st_size if ok else 0, "bytes")

# 同时导出各尺寸 PNG，方便 README 引用
for s in SIZES:
    draw(s).save(str(OUT / f"icon-{s}.png"), "PNG")

print("png:", (OUT / "icon.png").stat().st_size, "bytes")
