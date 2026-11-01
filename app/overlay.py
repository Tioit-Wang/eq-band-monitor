"""置顶、穿透、不抢焦点的悬浮窗 + 频谱绘制 + 录制状态。"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRectF, Qt
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontMetrics,
    QLinearGradient,
    QPainter,
    QPen,
)
from PySide6.QtWidgets import QWidget

from . import diagnostics, settings
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
            | Qt.NoDropShadowWindowHint
        )
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self.setFocusPolicy(Qt.NoFocus)

        self.resize(520, 268)

        # 拖动状态
        self._edit_mode = False          # 编辑态才可以拖
        self._dragging = False
        self._drag_offset = QPoint()

        self._font_small = QFont("Segoe UI", 8)
        self._font_num = QFont("Consolas", 8)
        self._font_title = QFont("Segoe UI Semibold", 9)
        # 位置：优先用保存值；越界（换显示器/改分辨率）则回落到默认点
        self.move(self._load_position())
        self.setMouseTracking(True)

        # 立刻套用一次穿透状态，不要等 showEvent ——
        # 否则窗口在显示前是可点击的，短暂的瞬间可能吃掉一次鼠标事件；
        # 单元测试里窗口从不 show，也就永远等不到那个时机。
        self._apply_click_through(not self._edit_mode)

    # ---------------- Windows 层面加固 / 点击穿透切换 ----------------

    GWL_EXSTYLE = -20
    WS_EX_TOPMOST = 0x00000008
    WS_EX_TRANSPARENT = 0x00000020
    WS_EX_TOOLWINDOW = 0x00000080
    WS_EX_LAYERED = 0x00080000
    WS_EX_NOACTIVATE = 0x08000000

    def showEvent(self, ev) -> None:
        super().showEvent(ev)
        self._apply_win32_styles()
        self._apply_click_through(not self._edit_mode)

    def _apply_win32_styles(self) -> None:
        """补上 NoActivate / ToolWindow，确保不进 Alt+Tab、不抢焦点。

        WS_EX_TRANSPARENT 不在这里无条件加 —— 它由 _apply_click_through()
        按编辑态/锁定态决定，否则编辑态也点不到窗口。
        """
        try:
            import ctypes

            hwnd = int(self.winId())
            user32 = ctypes.windll.user32
            cur = user32.GetWindowLongW(hwnd, self.GWL_EXSTYLE)
            new = (
                cur
                | self.WS_EX_TOPMOST
                | self.WS_EX_TOOLWINDOW
                | self.WS_EX_LAYERED
                | self.WS_EX_NOACTIVATE
            )
            user32.SetWindowLongW(hwnd, self.GWL_EXSTYLE, new)
        except Exception:
            pass

    def _apply_click_through(self, enabled: bool) -> None:
        """开启/关闭鼠标穿透。

        两条路都要走：
          * Qt 的 WA_TransparentForMouseEvents —— 控制 Qt 是否把事件投给本控件；
          * Win32 的 WS_EX_TRANSPARENT —— 控制命中测试是否算上这个窗口。
        只改 Qt 属性时窗口仍会"吃掉"点击（游戏里那一块点不到），
        只改 Win32 时 Qt 内部仍可能收到事件。两者必须同步。
        """
        self.setAttribute(Qt.WA_TransparentForMouseEvents, enabled)
        try:
            import ctypes

            hwnd = int(self.winId())
            user32 = ctypes.windll.user32
            cur = user32.GetWindowLongW(hwnd, self.GWL_EXSTYLE)
            new = (cur | self.WS_EX_TRANSPARENT) if enabled else (
                cur & ~self.WS_EX_TRANSPARENT
            )
            user32.SetWindowLongW(hwnd, self.GWL_EXSTYLE, new)
        except Exception:
            pass

    # ---------------- 编辑态 / 拖动 ----------------

    DEFAULT_POS = (24, 24)

    def _load_position(self) -> QPoint:
        """读取保存的位置，并做可见性校验。

        为什么要校验：用户可能拔了副屏、改了分辨率，或上次拖到屏幕外。
        直接套用会得到"看不见的窗口"，比回到默认位置糟得多。
        """
        saved = settings.get("overlay_pos")
        dx, dy = self.DEFAULT_POS
        if isinstance(saved, (list, tuple)) and len(saved) == 2:
            try:
                sx, sy = int(saved[0]), int(saved[1])
            except (TypeError, ValueError):
                return QPoint(dx, dy)
            if self._is_position_visible(sx, sy):
                return QPoint(sx, sy)
            diagnostics.log(f"悬浮窗保存位置 {sx},{sy} 不在任何屏幕上，回落默认位置")
        return QPoint(dx, dy)

    def _is_position_visible(self, x: int, y: int) -> bool:
        """标题栏那一条至少有一部分落在某个屏幕内，才算用户还能抓到它。"""
        try:
            from PySide6.QtGui import QGuiApplication

            probe_x = x + self.width() // 2
            probe_y = y + 12
            for screen in QGuiApplication.screens():
                g = screen.availableGeometry()
                if g.left() <= probe_x <= g.right() and g.top() <= probe_y <= g.bottom():
                    return True
            return False
        except Exception:
            return True  # 判断不了就别拦着

    def is_edit_mode(self) -> bool:
        return self._edit_mode

    def set_edit_mode(self, on: bool, save: bool = True) -> None:
        """切换锁定/编辑态。编辑态不穿透、可拖动。"""
        if on == self._edit_mode:
            return
        self._edit_mode = on
        self._dragging = False
        self._apply_click_through(not on)
        if on:
            # 编辑态需要能拿到键盘/焦点相关的鼠标捕获，先抬到最前
            self.raise_()
        elif save:
            self._save_position()
        self.update()

    def _save_position(self) -> None:
        try:
            pos = self.pos()
            settings.put("overlay_pos", [int(pos.x()), int(pos.y())])
            diagnostics.log(f"悬浮窗位置已保存: {pos.x()},{pos.y()}")
        except Exception:
            pass

    def _clamp_to_screen(self, pos: QPoint) -> QPoint:
        """把窗口限制在屏幕可用区域内，至少保证标题栏可抓。"""
        try:
            from PySide6.QtGui import QGuiApplication

            screens = QGuiApplication.screens()
            if not screens:
                return pos
            # 找离窗口中心最近的屏幕
            cx = pos.x() + self.width() // 2
            cy = pos.y() + self.height() // 2
            best = min(
                screens,
                key=lambda s: (s.geometry().center().x() - cx) ** 2
                + (s.geometry().center().y() - cy) ** 2,
            )
            g = best.availableGeometry()
            margin = 12
            min_x, max_x = g.left() - self.width() + margin + 60, g.right() - margin - 60
            min_y, max_y = g.top() - 0, g.bottom() - margin - 40
            return QPoint(
                max(min(pos.x(), max_x), min_x),
                max(min(pos.y(), max_y), min_y),
            )
        except Exception:
            return pos

    # ---- 鼠标事件：只有编辑态才会收到 ----

    TITLEBAR_H = 26  # 标题栏可拖动区域高度

    def mousePressEvent(self, ev) -> None:
        if not self._edit_mode or ev.button() != Qt.LeftButton:
            return super().mousePressEvent(ev)
        # 允许在标题栏区域按下拖动；其余区域也允许，但视觉上标题栏更符合直觉
        self._dragging = True
        self._drag_offset = ev.globalPosition().toPoint() - self.pos()
        ev.accept()

    def mouseMoveEvent(self, ev) -> None:
        if not (self._edit_mode and self._dragging):
            return super().mouseMoveEvent(ev)
        target = ev.globalPosition().toPoint() - self._drag_offset
        self.move(target)
        ev.accept()

    def mouseReleaseEvent(self, ev) -> None:
        if not self._edit_mode:
            return super().mouseReleaseEvent(ev)
        if self._dragging:
            self._dragging = False
            # 松手时吸附回屏幕内，避免拖到一半跑出可见范围
            self.move(self._clamp_to_screen(self.pos()))
            self._save_position()
            self.update()
        ev.accept()

    def mouseDoubleClickEvent(self, ev) -> None:
        """编辑态下双击标题栏 = 退出编辑态（符合常见窗口习惯）。"""
        if self._edit_mode and ev.button() == Qt.LeftButton:
            self.set_edit_mode(False)
            ev.accept()
            return
        super().mouseDoubleClickEvent(ev)

    # ---------------- 绘制 ----------------

    def paintEvent(self, ev) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setRenderHint(QPainter.TextAntialiasing, True)

        w, h = self.width(), self.height()

        if self._edit_mode:
            # 编辑态：把底色稍微提亮 + 画虚线边框，一眼能看出"现在处于可拖状态"，
            # 避免用户以为还在穿透而反复点游戏。
            p.setBrush(QColor(26, 32, 42, 235))
            p.setPen(QPen(QColor(34, 211, 238, 170), 1.4, Qt.DashLine))
        else:
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
        if self._edit_mode:
            self._draw_edit_hint(p, w, h)
        p.end()

    def _draw_edit_hint(self, p: QPainter, w: float, h: float) -> None:
        """编辑态的拖动把手 + 提示条。

        把手画在标题栏右侧（三点竖排），比整块可拖更符合"这里能抓"的直觉。
        """
        pad = 10
        # 拖动把手
        hx = w - pad - 8
        cy = pad + 10
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(34, 211, 238, 200))
        for i in range(3):
            p.drawEllipse(QPointF(hx, cy - 4 + i * 4), 1.4, 1.4)

        # 底部提示条
        text = "拖动移动 · 双击回到锁定 · 位置自动保存"
        p.setFont(self._font_small)
        fm = QFontMetrics(self._font_small)
        tw = fm.horizontalAdvance(text) + 16
        bx = (w - tw) / 2
        by = h - 20
        p.setBrush(QColor(34, 211, 238, 36))
        p.setPen(QPen(QColor(34, 211, 238, 120), 1))
        p.drawRoundedRect(QRectF(bx, by, tw, 15), 4, 4)
        p.setPen(ACCENT)
        p.drawText(QRectF(bx, by, tw, 15), Qt.AlignCenter, text)

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
