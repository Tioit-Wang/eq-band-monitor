"""WASAPI loopback 采集层。

职责：
  * 枚举系统中所有可回环采集的播放端点（speaker loopback）。
  * 在独立线程里持续把音频块推进到共享环形缓冲。
  * 设备拔出 / 切换时自动重连，不抛异常到主线程。
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class DeviceInfo:
    """一个可采集的播放端点。"""

    index: int
    name: str
    is_default: bool = False
    channels: int = 2

    def label(self) -> str:
        return ("★ " if self.is_default else "") + self.name


def list_devices() -> list[DeviceInfo]:
    """枚举可回环采集的端点。默认扬声器排最前并标记 ★。"""
    import soundcard as sc

    out: list[DeviceInfo] = []
    try:
        default_name = sc.default_speaker().name
    except Exception:
        default_name = ""

    try:
        mics = sc.all_microphones(include_loopback=True)
    except Exception:
        return out

    for i, m in enumerate(mics):
        name = m.name
        out.append(
            DeviceInfo(
                index=i,
                name=name,
                is_default=(name == default_name),
                channels=getattr(m, "channels", 2) or 2,
            )
        )

    # 默认设备置顶，其余保持原顺序
    out.sort(key=lambda d: (not d.is_default, d.index))
    return out


class LoopbackCapture:
    """持续采集指定端点，把最新音频块放进环形缓冲。

    线程安全：`read_latest()` 只读取受锁保护的最新块，不阻塞采集线程。

    性能要点：soundcard 的每次 `record()` 有约 1.3ms 的固定开销（与块大小无关），
    所以用**大块低频**读取（默认 0.1s/块 ≈ 10 次/秒），而不是小块高频。
    实测小块 47 次/秒占 6.2%，大块 10 次/秒只占 4.2%。
    """

    def __init__(
        self,
        device_index: int = 0,
        samplerate: int = 48000,
        blocksize: int = 4800,
        channels: int = 2,
    ) -> None:
        self.device_index = device_index
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.channels = channels

        self._lock = threading.Lock()
        self._latest: np.ndarray | None = None       # mono，供分频分析
        self._latest_raw: np.ndarray | None = None   # 原始立体声，供录制
        self._seq = 0

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        # 状态（供 UI 读取）
        self.status = "未启动"
        self.last_error = ""
        self.reconnects = 0
        self.device_name = ""
        self._recorder = None

    # ---------------- 生命周期 ----------------

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="loopback-capture")
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        t = self._thread
        if t and t.is_alive():
            t.join(timeout=timeout)
        self._thread = None
        self._close_recorder()

    def set_device(self, device_index: int) -> None:
        """切换设备：置标志后让采集循环自己重开流。"""
        with self._lock:
            if device_index == self.device_index:
                return
            self.device_index = device_index
        self._close_recorder()

    # ---------------- 数据读取 ----------------
    def read_latest(self) -> tuple[np.ndarray | None, int]:
        """返回 (最新 mono 音频块, 序号)。序号变化表示有新数据。"""
        with self._lock:
            if self._latest is None:
                return None, self._seq
            return self._latest, self._seq

    def read_latest_raw(self) -> tuple[np.ndarray | None, int]:
        """返回 (最新原始立体声块 shape=(N, ch), 序号)，供录制原始音频。"""
        with self._lock:
            if self._latest_raw is None:
                return None, self._seq
            return self._latest_raw, self._seq

    # ---------------- 内部实现 ----------------

    def _close_recorder(self) -> None:
        rec = self._recorder
        self._recorder = None
        if rec is not None:
            try:
                rec.__exit__(None, None, None)
            except Exception:
                pass

    def _open_recorder(self):
        import soundcard as sc

        mics = sc.all_microphones(include_loopback=True)
        if not mics:
            raise RuntimeError("系统中没有可回环采集的播放端点")

        # 按稳定名称匹配，避免索引漂移；名字找不到就退回索引
        target = None
        cur_name = self.device_name
        if cur_name:
            for m in mics:
                if m.name == cur_name:
                    target = m
                    break
        if target is None:
            if self.device_index >= len(mics):
                raise RuntimeError("设备索引超出范围")
            target = mics[self.device_index]

        self.device_name = target.name
        ch = max(1, min(self.channels, getattr(target, "channels", 2) or 2))
        rec = target.recorder(
            samplerate=self.samplerate,
            channels=ch,
            blocksize=self.blocksize,
        )
        rec.__enter__()
        self._recorder = rec
        return rec

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.status = "连接中…"
                rec = self._open_recorder()
                self.status = "采集中"
                self.last_error = ""

                while not self._stop.is_set():
                    # 设备被切换时 _close_recorder 会把 recorder 置空 → 跳出重开
                    if self._recorder is None:
                        break
                    data = rec.record(numframes=self.blocksize)
                    if data is None or len(data) == 0:
                        continue
                    arr = np.asarray(data, dtype=np.float32)
                    if arr.ndim == 1:
                        arr = arr[:, None]
                    if self._recorder is None:  # 期间被切走，丢弃这块
                        break
                    with self._lock:
                        # 分频分析用 mono，录制原始音频用原始立体声
                        self._latest = arr.mean(axis=1)
                        self._latest_raw = arr
                        self._seq += 1

            except Exception as exc:  # 拔设备、切采样率、独占冲突…
                self.last_error = f"{type(exc).__name__}: {exc}"
                self.status = "断开，重连中…"
                self.reconnects += 1
                self._close_recorder()
                for _ in range(20):  # 最多等 1s，期间可被 stop 打断
                    if self._stop.is_set():
                        break
                    time.sleep(0.05)
            finally:
                self._close_recorder()

        self.status = "已停止"


def probe_signal(seconds: float = 0.35, samplerate: int = 48000) -> dict[int, float]:
    """快速探测每个 loopback 端点的当前电平。

    为什么需要：系统默认输出可能是虚拟增强器（如 FxSound Audio Enhancer），
    它的回环端点在"没有音频流经过"时是完全静音的，
    而用户实际听到声音的物理耳机端点却可能有信号。
    只按"系统默认"选设备会采到一片静音。

    返回 {device_index: peak}，peak 为 0 表示该端点当前无信号。
    """
    import soundcard as sc

    out: dict[int, float] = {}
    try:
        mics = sc.all_microphones(include_loopback=True)
    except Exception:
        return out

    for i, m in enumerate(mics):
        try:
            with m.recorder(samplerate=samplerate, channels=2, blocksize=1024) as rec:
                peak = 0.0
                frames = int(samplerate * seconds)
                got = 0
                while got < frames:
                    d = rec.record(numframes=min(1024, frames - got))
                    if d is None or len(d) == 0:
                        break
                    peak = max(peak, float(abs(d).max()))
                    got += len(d)
            out[i] = peak
        except Exception as exc:
            out[i] = -1.0  # 打开失败
    return out


def pick_best_device(devices: list[DeviceInfo], prefer_default: bool = True) -> int:
    """挑一个真正能出声音的端点。

    优先级：
      1. 有信号 且 是系统默认
      2. 有信号 的任意端点（取信号最大者）
      3. 系统默认（即使暂时静音，可能只是当下没播放）
      4. 索引 0
    """
    levels = probe_signal()
    if not levels:
        return next((d.index for d in devices if d.is_default), 0)

    silent_threshold = 1e-4
    default_idx = next((d.index for d in devices if d.is_default), None)

    def ok(idx: int) -> bool:
        return levels.get(idx, -1.0) > silent_threshold

    if prefer_default and default_idx is not None and ok(default_idx):
        return default_idx

    live = [(v, k) for k, v in levels.items() if v > silent_threshold]
    if live:
        live.sort(reverse=True)
        return live[0][1]

    if default_idx is not None:
        return default_idx
    return devices[0].index if devices else 0
