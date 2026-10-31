"""原始音频录制：把 loopback 采集到的 PCM 直接落盘成 WAV。

设计决定：
  * **24-bit PCM**。16-bit 在 -60dB 以下量化噪声会吃掉底噪信息，
    而我们恰恰要分析底噪；32-bit float 兼容性差（部分工具不认）。
    24-bit 兼顾动态范围与通用性。
  * 保留**原始声道数**，不做 mono 混合。左右声道信息对定位有价值，
    而且是否混音可以留给分析阶段决定。
  * 每个场景一个独立文件，文件名带时间戳，避免覆盖。
  * 用 `wave` 标准库手写 24-bit（标准库只原生支持 8/16-bit，
    但 wave 模块允许任意 sampwidth，写 3 字节样本即可）。
"""

from __future__ import annotations

import os
import wave
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np


def _f32_to_s24(x: np.ndarray) -> bytes:
    """float32 [-1, 1] → 24-bit 小端 PCM 字节流。"""
    a = np.clip(np.asarray(x, dtype=np.float32), -1.0, 1.0)
    # 左移 8 位映射到 24-bit 有符号整数范围
    ints = np.round(a * 8388607.0).astype(np.int32)
    ints = np.clip(ints, -8388608, 8388607)
    # 取低 24 位，小端 3 字节
    b = ints.astype("<i4").view(np.uint8).reshape(-1, 4)[:, :3]
    return b.tobytes()

def safe_filename_part(key: str) -> str:
    """把场景 key 变成可安全用作文件名的片段。

    key 可能是组合键 'near:run'。**冒号绝不能留在文件名里**：
    NTFS 会把 `xxx_near:run.wav` 解释成「文件 xxx_near 的备用数据流 run.wav」，
    结果目录里出现一个 0 字节的 `xxx_near`，真实数据藏在 ADS 里，
    复制到 audio/ 目录时也会同样坏掉（实测确认）。
    """
    return key.replace(":", "-").replace("/", "-").replace("\\", "-")

@dataclass
class RecordedClip:
    """一次录制的结果。"""

    scene_key: str
    scene_name: str
    path: str
    seconds: float
    samplerate: int
    channels: int
    peak_db: float
    rms_db: float
    clipped: bool          # 是否出现削波（接近满刻度）
    silent: bool           # 整段几乎无信号

    def summary_line(self) -> str:
        flags = []
        if self.clipped:
            flags.append("⚠削波")
        if self.silent:
            flags.append("⚠几乎静音")
        tail = ("  " + " ".join(flags)) if flags else ""
        return (
            f"{self.scene_name:<8} {self.seconds:5.1f}s  "
            f"峰值 {self.peak_db:6.1f}dB  RMS {self.rms_db:6.1f}dB{tail}"
        )


@dataclass
class RawRecorder:
    """把采集到的原始 PCM 累积成 WAV。"""

    out_dir: str
    samplerate: int = 48000
    clip_threshold: float = 0.999       # 峰值超过此值视为削波
    silent_threshold: float = 1e-4      # 峰值低于此值视为几乎静音

    current_key: str | None = None
    current_name: str = ""
    _chunks: list[np.ndarray] = field(default_factory=list)
    _channels: int = 2
    _started: float = 0.0
    _frames: int = 0

    clips: dict[str, RecordedClip] = field(default_factory=dict)

    MAX_SECONDS = 30.0

    # ---------------- 控制 ----------------

    def start(self, key: str, name: str) -> None:
        self.current_key = key
        self.current_name = name
        self._chunks = []
        self._frames = 0
        self._started = datetime.now().timestamp()

    @property
    def is_recording(self) -> bool:
        return self.current_key is not None

    def elapsed(self) -> float:
        if self.current_key is None:
            return 0.0
        return self._frames / max(1, self.samplerate)

    def feed(self, block: np.ndarray) -> None:
        """喂入一块原始音频 shape=(N, ch)。"""
        if self.current_key is None or block is None:
            return
        arr = np.asarray(block, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[:, None]
        if arr.size == 0:
            return
        self._channels = arr.shape[1]
        self._chunks.append(arr.copy())
        self._frames += arr.shape[0]
        # 超过上限就丢弃最早的块，避免长时间录制吃内存
        if self._frames > self.MAX_SECONDS * self.samplerate:
            drop = self._frames - int(self.MAX_SECONDS * self.samplerate)
            while drop > 0 and self._chunks:
                head = self._chunks[0]
                if head.shape[0] <= drop:
                    drop -= head.shape[0]
                    self._chunks.pop(0)
                else:
                    self._chunks[0] = head[drop:]
                    drop = 0
            self._frames = int(self.MAX_SECONDS * self.samplerate)

    def stop(self) -> RecordedClip | None:
        """结束录制并写盘。"""
        key, name = self.current_key, self.current_name
        self.current_key = None
        if key is None or not self._chunks:
            return None

        data = np.concatenate(self._chunks, axis=0)
        self._chunks = []

        peak = float(np.max(np.abs(data)))
        rms = float(np.sqrt(np.mean(data.astype(np.float64) ** 2)))
        peak_db = 20.0 * np.log10(max(peak, 1e-9))
        rms_db = 20.0 * np.log10(max(rms, 1e-9))

        os.makedirs(self.out_dir, exist_ok=True)
        stamp = datetime.now().strftime("%m%d_%H%M%S")
        fname = f"{stamp}_{safe_filename_part(key)}.wav"
        path = os.path.join(self.out_dir, fname)

        with wave.open(path, "wb") as w:
            w.setnchannels(data.shape[1])
            w.setsampwidth(3)          # 24-bit
            w.setframerate(self.samplerate)
            w.writeframes(_f32_to_s24(data))

        clip = RecordedClip(
            scene_key=key,
            scene_name=name,
            path=path,
            seconds=data.shape[0] / self.samplerate,
            samplerate=self.samplerate,
            channels=data.shape[1],
            peak_db=peak_db,
            rms_db=rms_db,
            clipped=peak >= self.clip_threshold,
            silent=peak < self.silent_threshold,
        )
        self.clips[key] = clip
        return clip

    def clear(self, key: str | None = None) -> None:
        if key is None:
            self.clips.clear()
        else:
            self.clips.pop(key, None)

    def manifest_lines(self) -> list[str]:
        if not self.clips:
            return ["（还没有录制的音频）"]
        return [c.summary_line() for c in self.clips.values()]

    def coverage(self) -> dict[str, list[str]]:
        """位置 → 已录到的移动方式列表（用于显示 位置×方式 N/12 的进度）。

        只看 key 里的冒号，不依赖 recorder.py，避免循环导入。
        """
        order = ("sneak", "walk", "run")
        raw: dict[str, set[str]] = {}
        for key in self.clips:
            if ":" not in key:
                continue
            loc, gait = key.split(":", 1)
            raw.setdefault(loc, set()).add(gait)
        return {loc: [g for g in order if g in gs] for loc, gs in raw.items()}
