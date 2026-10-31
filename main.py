"""EQ Band Monitor — 程序入口。

功能：
  1. 实时 61 段频谱悬浮窗（置顶、点击穿透、不抢焦点）
  2. 按 F1–F9 为不同声学场景录制原始音频（再按一次结束）
  3. 按 F10 生成交接包，交给 AI 判断该怎么调

场景是「位置 × 移动方式」两个维度的组合：
  位置：自己 F2 / 楼下 F3 / 旁边 F4 / 远处 F5
  方式：静步 Ctrl+Shift+1 / 正常走 Ctrl+Shift+2 / 奔跑 Ctrl+Shift+3
  固定场景：静止底噪 F1 / 跳跃 F6 / 开火 F7 / 手榴弹 F8 / 震撼弹 F9

功能键：
  F10 生成交接包  F11 清空  Ctrl+Shift+R 峰值清零  Ctrl+Shift+D 显示范围
  Ctrl+Shift+H 隐藏悬浮窗   Ctrl+Shift+Q 退出
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QColor, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QMenu,
    QMessageBox,
    QSystemTrayIcon,
)

from app import diagnostics as diag
from app import fac
from app.advice_window import AdviceWindow
from app.analyzer import SpectrumAnalyzer
from app.capture import LoopbackCapture, list_devices, pick_best_device
from app.handoff import save_bundle
from app.hotkeys import HotkeyManager
from app.overlay import Overlay
from app.rawanalyze import build_report
from app.rawrec import RawRecorder
from app.recorder import (
    FIXED_SCENES,
    GAITS,
    GAIT_BY_KEY,
    LOCATIONS,
    SCENES,
    combo_key,
    display_name,
)
from app.resources import asset
from app.version import APP_NAME, HOMEPAGE, ORG, __version__

ANALYSIS_FPS = 20
SAMPLERATE = 48000
BLOCKSIZE = 4800  # 已不再使用固定 FFT 尺寸，分析器内部按频段分三级
  # 0.1s；soundcard 的 record() 有固定开销，大块低频更省 CPU


@dataclass
class AppState:
    """悬浮窗绘制时读取的共享状态。"""

    analyzer: SpectrumAnalyzer | None = None
    capture_ok: bool = False
    status_text: str = "启动中…"
    # 录制状态
    is_recording: bool = False
    record_scene: str = ""
    record_elapsed: float = 0.0
    footer_hint: str = ""
    gait_label: str = "正常走"
    scene_summary: str = ""


class MonitorApp:
    def __init__(self, qt_app: QApplication) -> None:
        self.qt_app = qt_app
        self.state = AppState()

        self.devices = list_devices()
        self.state.analyzer = SpectrumAnalyzer(
            samplerate=SAMPLERATE, release_ms=200.0
        )
        # 原始音频落盘器（交付给 AI 的主要产物）
        self.raw_rec = RawRecorder(out_dir=self.raw_dir(), samplerate=SAMPLERATE)

        # 默认采集"系统默认输出设备"的 loopback。
        # 注意：list_devices() 为了展示把默认设备排在最前，所以必须取它的
        # 原始 index，不能硬编码 0（否则可能采到一个静音的端点）。
        # 自动挑一个真正有声音的端点：系统默认可能是虚拟增强器（FxSound），
        # 它的回环端点在无音频流时完全静音，只按默认选会采到一片静音。
        default_idx = pick_best_device(self.devices)
        self.capture = LoopbackCapture(
            device_index=default_idx,
            samplerate=SAMPLERATE,
            blocksize=BLOCKSIZE,
            channels=2,
        )
        self.capture.start()

        self.overlay = Overlay(self.state)
        self.overlay.show()

        self.advice_window: AdviceWindow | None = None
        self.base_preset = None
        self.base_preset_path = None

        # 热键
        self.hotkeys = HotkeyManager()
        self.hotkeys.triggered.connect(self._on_hotkey)
        self._register_hotkeys()

        self._build_tray()

        self._timer = QTimer()
        self._timer.setTimerType(Qt.PreciseTimer)
        self._timer.timeout.connect(self._tick)
        self._timer.start(int(1000 / ANALYSIS_FPS))

        self._last_seq = -1
        self._raw_last_seq = -1
        self._last_status = 0.0
        # 当前移动方式：影响 F2-F5 的录制内容（静步/正常走/奔跑）
        self.gait = "walk"
        self.state.gait_label = GAIT_BY_KEY["walk"]
        self.state.scene_summary = self._raw_summary()

    # ---------------- 热键注册 ----------------

    def _register_hotkeys(self) -> None:
        ok, fail = [], []
        for s in SCENES:
            (ok if self.hotkeys.register(s.key, s.hotkey) else fail).append(s.hotkey)

        # 功能键必须避开场景键：场景已用到 F1-F9，功能键从 F10 起。
        # 不用 F12：许多系统/软件把它占为"另存为"或调试键。
        # 移动方式选择器：3 个挡位，用 Ctrl+Shift+1/2/3 切换。
        # 这样「位置 × 方式」的 12 种组合不需要 12 个热键。
        extra = [
            ("gait_sneak", "1", True, True),
            ("gait_walk", "2", True, True),
            ("gait_run", "3", True, True),
            ("show_advice", "F10", False, False),
            ("clear_rec", "F11", False, False),
            ("reset_peak", "R", True, True),
            ("toggle_range", "D", True, True),
            ("hide", "H", True, True),
            ("quit", "Q", True, True),
        ]
        for name, key, ctrl, shift in extra:
            got = self.hotkeys.register(name, key, ctrl=ctrl, shift=shift)
            (ok if got else fail).append(("Ctrl+Shift+" if ctrl else "") + key)
        self._hk_ok = ok
        self._hk_fail = fail
        diag.log(f"热键注册成功 {len(ok)} 个: {', '.join(ok)}")
        if fail:
            diag.log(f"热键注册失败 {len(fail)} 个: {', '.join(fail)}")
        self._hk_fail = fail

    # ---------------- 主循环 ----------------

    def _tick(self) -> None:
        block, seq = self.capture.read_latest()
        an = self.state.analyzer
        now = time.perf_counter()

        if seq != self._last_seq:
            self._last_seq = seq
            if an is not None and block is not None:
                an.process(block, now)
        else:
            if an is not None:
                an.process(np.zeros(0, dtype=np.float32), now)

        # 录制：原始 PCM 落盘（这是要交给 AI 的主要产物）。
        # 必须用独立的序号判断"这块是否已经喂过"。
        # 早先用 `raw_seq == seq` 是错的：没有新数据时两者都等于 _last_seq，
        # 同一个块会被重复喂入，实测导致录制时长精确翻倍（2.0s → 4.1s）。
        if self.raw_rec.is_recording:
            raw, raw_seq = self.capture.read_latest_raw()
            if raw is not None and raw_seq != self._raw_last_seq:
                self._raw_last_seq = raw_seq
                self.raw_rec.feed(raw)

        # 状态刷新（限流 5Hz）
        if now - self._last_status > 0.2:
            self._last_status = now
            status = self.capture.status
            ok = status == "采集中"
            self.state.capture_ok = ok
            if ok:
                name = self.capture.device_name or "未知设备"
                self.state.status_text = name if len(name) <= 30 else name[:29] + "…"
            else:
                self.state.status_text = status

            self.state.is_recording = self.raw_rec.is_recording
            if self.raw_rec.is_recording:
                self.state.record_scene = self.raw_rec.current_name
                self.state.record_elapsed = self.raw_rec.elapsed()
            self.state.scene_summary = self._raw_summary()

        self.overlay.update()

    # ---------------- 交互 ----------------

    def _on_hotkey(self, name: str) -> None:
        # 场景录制键
        scene = SCENE_BY_HOTKEY.get(name.upper()) if name in SCENE_BY_HOTKEY else None
        if scene is None:
            scene = next((s for s in SCENES if s.key == name), None)

        if scene is not None:
            self._toggle_record(scene.key)
        if name.startswith("gait_"):
            self._set_gait(name.split("_", 1)[1])
        elif name == "show_advice":
            self._show_advice()
        elif name == "clear_rec":
            self._clear_recordings()
        elif name == "reset_peak":
            if self.state.analyzer:
                self.state.analyzer.reset_peak_hold()
        elif name == "toggle_range":
            self._cycle_range()
        elif name == "hide":
            self.overlay.setVisible(not self.overlay.isVisible())
        elif name == "quit":
            self.shutdown()

    def _toggle_record(self, key: str) -> None:
        """开始/结束录制。位置型场景会自动拼上当前移动方式。"""
        rec = self.raw_rec
        base = next((s for s in SCENES if s.key == key), None)
        if base is None:
            return

        is_location = base in LOCATIONS
        actual_key = combo_key(key, self.gait) if is_location else key
        name = display_name(actual_key)

        if rec.current_key == actual_key:
            clip = rec.stop()
            if clip:
                self._notify(f"「{name}」已保存 {clip.seconds:.1f}s")
            else:
                self._notify(f"「{name}」没采到数据")
        else:
            if rec.is_recording:
                prev = rec.stop()
                if prev:
                    self._notify(f"「{prev.scene_name}」已保存 {prev.seconds:.1f}s")
            rec.start(actual_key, name)
            hint = f"（当前方式：{GAIT_BY_KEY.get(self.gait, self.gait)}）" if is_location else ""
            self._notify(f"开始录制「{name}」{hint}，再按 {base.hotkey} 结束")

        self.state.is_recording = rec.is_recording
        self.state.record_scene = name if rec.is_recording else ""
        self.state.scene_summary = self._raw_summary()

    def _set_gait(self, gait: str) -> None:
        """切换当前移动方式（影响 F2-F5 的录制内容）。"""
        self.gait = gait
        label = GAIT_BY_KEY.get(gait, gait)
        self.state.gait_label = label
        self.state.footer_hint = (
            f"当前移动方式：{label}（Ctrl+Shift+1/2/3 切换）· "
            "F2-F5 录位置脚步 · F10 生成交接包"
        )
        self._notify(f"移动方式切换为「{label}」")

    def _raw_summary(self) -> str:
        clips = self.raw_rec.clips
        if not clips:
            return ("还没录制 — Ctrl+Shift+1/2/3 选移动方式，"
                    "F2 自己 / F3 楼下 / F4 旁边 / F5 远处，F10 生成交接包")
        done = [f"{c.scene_name}{c.seconds:.0f}s" for c in clips.values()]
        cov = self.raw_rec.coverage()
        tot = sum(len(v) for v in cov.values())
        return "已录：" + "  ".join(done) + f"    （位置×方式 {tot}/12）    按 F10 生成交接包"

    def _clear_recordings(self) -> None:
        self.raw_rec.stop()
        self.raw_rec.clear()
        self.state.is_recording = False
        self.state.scene_summary = self._raw_summary()
        self._notify("已清空录制记录（磁盘上的 WAV 文件仍保留）")

    def _cycle_range(self) -> None:
        an = self.state.analyzer
        if an is None:
            return
        spans = [60.0, 45.0, 90.0]
        cur = an.max_db - an.min_db
        idx = int(np.argmin([abs(cur - s) for s in spans]))
        nxt = spans[(idx + 1) % len(spans)]
        an.min_db = an.max_db - nxt
        self._notify(f"显示范围：{nxt:.0f} dB")

    def _show_advice(self) -> None:
        """F10：分析已录制的原始音频，生成交接包。"""
        clips = self.raw_rec.clips
        if not clips:
            self._notify("还没有录制的音频。先按 F2 / F5 录几个场景")
            return

        paths = {k: (c.path, c.scene_name) for k, c in clips.items()}
        try:
            rep = build_report(paths)
        except Exception as exc:
            self._notify(f"分析失败：{exc}")
            return

        base_info = None
        if self.base_preset is not None:
            rows = ["```"]
            for i, b in enumerate(self.base_preset.trimmed(), 1):
                rows.append(f"Band {i:<2} {b.cf:>9.2f} Hz  {b.gain:+6.2f} dB")
            rows.append("```")
            base_info = f"预设名：**{self.base_preset.name}**\n\n" + "\n".join(rows)

        out_dir = os.path.join(
            self.raw_rec.out_dir,
            "handoff_" + datetime.now().strftime("%m%d_%H%M%S"),
        )
        try:
            save_bundle(out_dir, rep, {k: c.path for k, c in clips.items()}, base_info)
        except Exception as exc:
            self._notify(f"写交接包失败：{exc}")
            return

        if self.advice_window is not None:
            self.advice_window.close()
        self.advice_window = AdviceWindow(rep, out_dir, base_preset_info=base_info)
        self.advice_window.show()
        self.advice_window.raise_()
        self.advice_window.activateWindow()
    # ---------------- 预设 ----------------

    def _load_preset(self, path: str) -> None:
        """加载 FxSound .fac 作为"听感基线"。

        建议会以增量的形式叠在它上面，而不是整个覆盖 ——
        那个预设承载了用户的主观听感偏好，我们只补游戏里的缺口。
        """
        try:
            p = fac.load(path)
        except Exception as exc:
            self._notify(f"读取预设失败：{exc}")
            return
        if not p.bands:
            self._notify("这个文件里没有 EQ 频段数据")
            return
        self.base_preset = p
        self.base_preset_path = path
        self._notify(f"已加载基线预设「{p.name}」（{len(p.bands)} 段）")

    def _choose_preset(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            None, "选择 FxSound 预设", self._preset_dir(), "FxSound 预设 (*.fac)"
        )
        if path:
            self._load_preset(path)

    def _clear_preset(self) -> None:
        self.base_preset = None
        self.base_preset_path = None
        self._notify("已清除基线预设，建议将从零开始算")

    @staticmethod
    def _preset_dir() -> str:
        d = os.path.join(os.path.expanduser("~"), "Documents", "FxSound", "Presets")
        return d if os.path.isdir(d) else os.path.expanduser("~")

    @staticmethod
    def raw_dir() -> str:
        """原始音频存放目录（按会话日期分组，便于回看）。"""
        base = os.path.join(os.path.expanduser("~"), "Documents", "EQBandMonitor", "recordings")
        os.makedirs(base, exist_ok=True)
        return base

    # ---------------- 托盘 ----------------

    def _build_tray(self) -> None:
        # 优先用打包/仓库里的图标；找不到才回退到代码画的频谱柱。
        icon_path = asset("icon.png")
        if icon_path.is_file():
            icon = QIcon(str(icon_path))
        else:
            pix = QPixmap(32, 32)
            pix.fill(QColor(0, 0, 0, 0))
            p = QPainter(pix)
            p.setRenderHint(QPainter.Antialiasing, True)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor("#22d3ee"))
            for i, hgt in enumerate([8, 16, 26, 18, 10]):
                p.drawRect(4 + i * 5, 30 - hgt, 3, hgt)
            p.end()
            icon = QIcon(pix)

        self.tray = QSystemTrayIcon(icon)
        self.tray.setToolTip(f"{APP_NAME} v{__version__}")
        self.tray.activated.connect(self._onTrayActivated)

        menu = QMenu()
        menu.addAction(QAction("显示 / 隐藏悬浮窗", menu,
                               triggered=lambda: self.overlay.setVisible(not self.overlay.isVisible())))
        menu.addAction(QAction("生成交接包 (F10)", menu, triggered=self._show_advice))
        menu.addAction(QAction("清空录制数据 (F11)", menu, triggered=self._clear_recordings))
        menu.addSeparator()
        pm = menu.addMenu("基线预设 (FxSound .fac)")
        pm.addAction(QAction("选择预设文件…", pm, triggered=self._choose_preset))
        pm.addAction(QAction("清除基线预设", pm, triggered=self._clear_preset))
        menu.addSeparator()

        rec_menu = menu.addMenu("开始录制场景")
        for s in SCENES:
            rec_menu.addAction(
                QAction(f"{s.hotkey}  {s.name} — {s.desc}", rec_menu,
                        triggered=lambda _=False, k=s.key: self._toggle_record(k))
            )

        menu.addSeparator()
        dm = menu.addMenu("输出设备")
        self._dev_actions = []
        for d in self.devices:
            act = QAction(d.label(), dm, checkable=True)
            act.setChecked(d.index == self.capture.device_index)
            act.triggered.connect(lambda _=False, idx=d.index: self._switch_device(idx))
            dm.addAction(act)
            self._dev_actions.append(act)
        menu.addAction(QAction("重新扫描设备", menu, triggered=self._rescan_devices))

        menu.addSeparator()
        if self._hk_fail:
            menu.addAction(QAction(f"⚠ 热键冲突：{', '.join(self._hk_fail)}", menu)).setEnabled(False)
        menu.addAction(QAction(f"关于 {APP_NAME} v{__version__}", menu, triggered=self._about))
        menu.addAction(QAction("退出", menu, triggered=self.shutdown))

        self.tray.setContextMenu(menu)
        self.tray.show()

    def _onTrayActivated(self, reason) -> None:
        if reason == QSystemTrayIcon.Trigger:
            self._show_advice()

    def _about(self) -> None:
        """关于对话框：版本与仓库地址。打包成 exe 后便于核对版本。"""
        QMessageBox.about(
            None,
            f"关于 {APP_NAME}",
            f"<b>{APP_NAME}</b> v{__version__}<br><br>"
            "游戏音频频谱监视 + 场景原始音频录制 + EQ 判断交接包。<br>"
            "不注入进程、不 hook 游戏、不读内存。<br><br>"
            f'<a href="{HOMEPAGE}">{HOMEPAGE}</a>',
        )

    def _switch_device(self, index: int) -> None:
        self.capture.device_name = ""
        self.capture.set_device(index)
        for i, act in enumerate(self._dev_actions):
            act.setChecked(i == index)
        self._notify("已切换采集设备")

    def _rescan_devices(self) -> None:
        self.devices = list_devices()
        self._build_tray()
        self._notify(f"已找到 {len(self.devices)} 个可采集端点")

    def _notify(self, text: str) -> None:
        try:
            self.tray.showMessage("EQ Band Monitor", text, QSystemTrayIcon.Information, 2200)
        except Exception:
            pass

    def shutdown(self) -> None:
        self._timer.stop()
        self.hotkeys.unregister_all()
        self.capture.stop()
        try:
            self.tray.hide()
        except Exception:
            pass
        self.qt_app.quit()


def main() -> int:
    # --version 供 CI / 打包自检使用：不启动 GUI，直接打印版本并退出。
    if "--version" in sys.argv or "-V" in sys.argv:
        print(f"{APP_NAME} {__version__}")
        return 0

    # 打包成 windowed 后没有控制台，必须先把日志与异常钩子装好，
    # 否则任何启动期异常都会让程序"无声消失"、完全无法排查。
    # 注意顺序：日志要在 QApplication 之前（Qt 自身警告也走这里）。
    diag.install()
    diag.install_qt_message_handler()

    if not diag.acquire_single_instance():
        # 已有实例：把它唤到前台，而不是起第二个抢不到热键的残废实例
        diag.log("检测到已有实例，唤醒现有窗口后退出")
        diag.focus_existing_window(APP_NAME)
        return 0

    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    app.setApplicationName(APP_NAME)
    app.setApplicationDisplayName(APP_NAME)
    app.setApplicationVersion(__version__)
    app.setOrganizationName(ORG)

    # 应用图标：窗口与任务栏都用它（打包后从 _MEIPASS/assets 取）
    icon_file = asset("icon.ico")
    if not icon_file.is_file():
        icon_file = asset("icon.png")
    if icon_file.is_file():
        app.setWindowIcon(QIcon(str(icon_file)))
    if not QSystemTrayIcon.isSystemTrayAvailable():
        print("警告：系统托盘不可用，热键仍工作。", file=sys.stderr)

    mon = MonitorApp(app)
    app.aboutToQuit.connect(mon.shutdown)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
