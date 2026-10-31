"""置顶、穿透、不抢焦点的悬浮窗 + 频谱绘制 + 录制状态。"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontMetrics,
    QLinearGradient,
    QPainter,
    QPen,
)
from PySide6.QtWidgets import QWidget

from .analyzer import BAND_CENTERS, N_BANDS, format_hz

# ---- 配色（呼应截图 EQ 的青蓝高亮）----
BG = QColor("#0d1117")
GRID = QColor(255, 255, 255, 18)
TEXT = QColor("#e6edf3")
TEXT_DIM = QColor("#7d8590")
ACCENT = QColor("#22d3ee")
PEAK_LINE = QColor("#ffffff")
WARN = QColor("#f87171")
REC = QColor("#f85149")
OK = QColor("#3fb950")


class Overlay(QWidget):
    """点击穿透的悬浮层。所有绘制都在 paintEvent 里手动完成。"""

    def __init__(self, app_state, parent=None) -> None:
        super().__init__(parent)
        self.state = app_state
        self.setWindowTitle("EQ Band Monitor")

        self.setWindowFlags(
            Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.Tool
            | Qt.WindowTransparentForInput
            | Qt.NoDropShadowWindowHint
        )
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setFocusPolicy(Qt.NoFocus)

        self.resize(520, 268)
        self.move(24, 24)

        self._font_small = QFont("Segoe UI", 8)
        self._font_num = QFont("Consolas", 8)
        self._font_title = QFont("Segoe UI Semibold", 9)

    # ---------------- Windows 层面加固 ----------------

    def showEvent(self, ev) -> None:
        super().showEvent(ev)
        self._apply_win32_styles()

    def _apply_win32_styles(self) -> None:
        """补上 NoActivate / ToolWindow，确保不进 Alt+Tab、不抢焦点。"""
        try:
            import ctypes

            GWL_EXSTYLE = -20
            WS_EX_TOPMOST = 0x00000008
            WS_EX_TRANSPARENT = 0x00000020
            WS_EX_TOOLWINDOW = 0x00000080
            WS_EX_LAYERED = 0x00080000
            WS_EX_NOACTIVATE = 0x08000000

            hwnd = int(self.winId())
            user32 = ctypes.windll.user32
            cur = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            new = (
                cur
                | WS_EX_TOPMOST
                | WS_EX_TRANSPARENT
                | WS_EX_TOOLWINDOW
                | WS_EX_LAYERED
                | WS_EX_NOACTIVATE
            )
            user32.SetWindowLongW(hwnd, GWL_EXSTYLE, new)
        except Exception:
            pass

    # ---------------- 绘制 ----------------

    def paintEvent(self, ev) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setRenderHint(QPainter.TextAntialiasing, True)

        w, h = self.width(), self.height()
        p.setBrush(BG)
        p.setPen(QPen(QColor(255, 255, 255, 26), 1))
        p.drawRoundedRect(QRectF(0.5, 0.5, w - 1, h - 1), 8, 8)

        title_h = 24
        readout_h = 26
        foot_h = 34
        pad = 10

        spec_top = pad + title_h
        spec_bottom = h - foot_h - readout_h - pad
        spec_rect = QRectF(pad, spec_top, w - 2 * pad, max(40.0, spec_bottom - spec_top))

        self._draw_titlebar(p, pad, title_h, w)
        self._draw_spectrum(p, spec_rect)
        self._draw_readout(p, pad, spec_rect.bottom() + 4, w)
        self._draw_footer(p, pad, h - foot_h + 2, w)
        p.end()

    def _draw_titlebar(self, p: QPainter, pad: float, th: float, w: float) -> None:
        p.setFont(self._font_title)
        p.setPen(TEXT)
        p.drawText(QRectF(pad, pad - 2, w * 0.4, th - 2),
                   Qt.AlignLeft | Qt.AlignVCenter, "EQ Band Monitor")

        st = self.state
        ok = st.capture_ok
        dot = OK if ok else WARN
        cx = w - pad - 6
        cy = pad + th / 2 - 2
        p.setPen(Qt.NoPen)
        p.setBrush(dot)
        p.drawEllipse(QPointF(cx, cy), 3.5, 3.5)

        p.setFont(self._font_small)
        p.setPen(TEXT_DIM)
        msg = st.status_text
        fm = QFontMetrics(self._font_small)
        tw = fm.horizontalAdvance(msg)
        p.drawText(QRectF(cx - 12 - tw, pad - 2, tw, th - 2),
                   Qt.AlignRight | Qt.AlignVCenter, msg)

    def _draw_spectrum(self, p: QPainter, rect: QRectF) -> None:
        st = self.state
        an = st.analyzer
        if an is None:
            return

        p.setClipRect(rect)

        span = max(1e-6, an.max_db - an.min_db)
        ndiv = 5
        for i in range(ndiv + 1):
            y = rect.bottom() - rect.height() * i / ndiv
            p.setPen(QPen(GRID, 1))
            p.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
            p.setFont(self._font_small)
            p.setPen(TEXT_DIM)
            p.drawText(QRectF(rect.left() + 2, y - 12, 34, 12),
                       Qt.AlignLeft | Qt.AlignBottom, f"{an.min_db + span * i / ndiv:.0f}")

        norm = an.normalized()
        norm_peak = np.clip((an.peak_hold - an.min_db) / span, 0.0, 1.0)
        n = N_BANDS
        gap = 1.0
        bw = max(1.0, (rect.width() - gap * (n - 1)) / n)

        grad = QLinearGradient(0, rect.bottom(), 0, rect.top())
        grad.setColorAt(0.0, QColor(34, 211, 238, 60))
        grad.setColorAt(1.0, QColor(34, 211, 238, 235))

        p.setPen(Qt.NoPen)
        p.setBrush(grad)
        for i in range(n):
            bh = rect.height() * float(norm[i])
            if bh < 1.0:
                continue
            p.drawRect(QRectF(rect.left() + i * (bw + gap),
                              rect.bottom() - bh, bw, bh))

        # 峰值保持线
        p.setPen(QPen(PEAK_LINE, 1))
        for i in range(n):
            x = rect.left() + i * (bw + gap)
            y = rect.bottom() - rect.height() * float(norm_peak[i])
            p.drawLine(QPointF(x, y), QPointF(x + bw, y))

        # 录制中：在图上叠一层半透明红框，提醒"正在采样"
        if st.is_recording:
            p.setPen(QPen(QColor(248, 81, 73, 200), 2))
            p.setBrush(QColor(248, 81, 73, 18))
            p.drawRect(rect)

        p.setClipping(False)

        p.setFont(self._font_small)
        p.setPen(TEXT_DIM)
        for i in range(0, n, 6):
            x = rect.left() + i * (bw + gap)
            p.drawText(QRectF(x - 12, rect.bottom() + 1, 44, 12),
                       Qt.AlignLeft | Qt.AlignTop, format_hz(BAND_CENTERS[i]))

    def _draw_readout(self, p: QPainter, pad: float, y: float, w: float) -> None:
        st = self.state
        an = st.analyzer
        if an is None:
            return
        p.setPen(QPen(GRID, 1))
        p.drawLine(QPointF(pad, y), QPointF(w - pad, y))

        ry = y + 4
        p.setFont(self._font_small)
        p.setPen(TEXT_DIM)
        p.drawText(QRectF(pad, ry, 30, 16), Qt.AlignLeft | Qt.AlignVCenter, "PEAK")

        p.setFont(self._font_num)
        if an.has_signal and an.peak_hz > 0:
            p.setPen(ACCENT)
            p.drawText(QRectF(pad + 30, ry, 78, 16), Qt.AlignLeft | Qt.AlignVCenter,
                       f"{an.peak_hz:,.0f} Hz")
        else:
            p.setPen(TEXT_DIM)
            p.drawText(QRectF(pad + 30, ry, 78, 16), Qt.AlignLeft | Qt.AlignVCenter, "—")

        tops = an.top_bands(4) if an.has_signal else []
        p.setFont(self._font_small)
        tx = pad + 116
        for i, (fc, db) in enumerate(tops):
            p.setPen(ACCENT if i == 0 else TEXT_DIM)
            p.drawText(QRectF(tx, ry, 62, 16), Qt.AlignLeft | Qt.AlignVCenter,
                       f"{format_hz(fc)} {db:.0f}")
            tx += 62

        if not tops:
            p.setPen(TEXT_DIM)
            p.drawText(QRectF(pad + 116, ry, w - pad - 116, 16),
                       Qt.AlignLeft | Qt.AlignVCenter,
                       "无信号 — 播放点声音，或检查输出设备")

    def _draw_footer(self, p: QPainter, pad: float, y: float, w: float) -> None:
        st = self.state
        p.setPen(QPen(GRID, 1))
        p.drawLine(QPointF(pad, y - 2), QPointF(w - pad, y - 2))

        p.setFont(self._font_small)

        # 第一行左：当前移动方式徽标（位置脚步录制时会拼进场景名）
        badge = f"[{st.gait_label}]"
        fm = QFontMetrics(self._font_small)
        bw = fm.horizontalAdvance(badge) + 10
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(34, 211, 238, 40))
        p.drawRoundedRect(QRectF(pad, y + 1, bw, 13), 3, 3)
        p.setPen(ACCENT)
        p.drawText(QRectF(pad, y, bw, 15), Qt.AlignCenter, badge)

        # 第一行右：录制状态 / 操作提示
        tx = pad + bw + 6
        if st.is_recording:
            p.setPen(REC)
            p.drawText(QRectF(tx, y, w - tx - pad, 15), Qt.AlignLeft | Qt.AlignVCenter,
                       f"● 正在录制「{st.record_scene}」 {st.record_elapsed:4.1f}s"
                       "  （再按同一键结束）")
        else:
            p.setPen(TEXT_DIM)
            p.drawText(QRectF(tx, y, w - tx - pad, 15), Qt.AlignLeft | Qt.AlignVCenter,
                       st.footer_hint)

        # 第二行：已录场景概览
        p.setPen(TEXT_DIM)
        p.drawText(QRectF(pad, y + 15, w - 2 * pad, 15), Qt.AlignLeft | Qt.AlignVCenter,
                   st.scene_summary)
