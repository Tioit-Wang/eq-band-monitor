"""原始音频分析：把录制的 WAV 变成 AI 可读的结构化事实。

为什么需要它：
  录下来的是原始 PCM，直接"丢给 AI"没有意义 ——
  AI 需要一个**可核对的中间层**：文件里到底有什么、能量分布如何、
  瞬态特征是什么、有没有削波/静音这类数据质量问题。
  有了这层，判断就是基于可验证的数字，而不是凭感觉。

产出：
  1. 每个文件的元信息（时长/声道/采样率/峰值/RMS/削波/底噪）
  2. 分场景时间平均频谱（功率域，1/6 倍频程 61 段）
  3. 瞬态特征：起音时间分布、事件计数（区分"连续"vs"脉冲"声）
  4. 音调性指标：谱峰锐度（区分"乐音/脚步"vs"宽带噪声/枪声"）
  5. 场景间的形状差（tilt 域），可直接转成 EQ 增益需求
"""

from __future__ import annotations

import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .analyzer import BAND_CENTERS, BAND_HI, BAND_LO, N_BANDS


# ---------------- WAV 读写 ----------------

def read_wav(path: str) -> tuple[np.ndarray, int]:
    """读 WAV，返回 (float32 数组 shape=(N, ch) 归一化到 [-1,1], 采样率)。

    支持 8/16/24/32-bit 整数与 32-bit float。
    """
    with wave.open(path, "rb") as w:
        nch = w.getnchannels()
        width = w.getsampwidth()
        rate = w.getframerate()
        raw = w.readframes(w.getnframes())

    if width == 1:      # 8-bit 无符号
        a = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        a = (a - 128.0) / 128.0
    elif width == 2:    # 16-bit
        a = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 3:    # 24-bit
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        # 符号扩展
        v = np.where(v & 0x800000, v - 0x1000000, v)
        a = v.astype(np.float32) / 8388608.0
    elif width == 4:    # 可能是 32-bit int 或 float；按 float 处理（we write s24）
        a = np.frombuffer(raw, dtype="<f4").astype(np.float32)
    else:
        raise ValueError(f"不支持的位深: {width*8}-bit")

    if nch > 1:
        a = a.reshape(-1, nch)
    else:
        a = a[:, None]
    return a, rate


# ---------------- 分频（复用 analyzer 的三级多分辨率思路）----------------

def band_power_spectrum(mono: np.ndarray, rate: int) -> np.ndarray | None:
    """整段音频的分场景频谱：Welch 平均，返回 61 段带内总能量（线性）。

    为什么用 Welch 平均而不是单个大 FFT：
    单次大 FFT 会把瞬态事件"摊平"在整个时间窗里，得到的是平均谱，
    而且方差不收敛。分段加窗再平均是功率谱估计的标准做法。
    """
    x = np.asarray(mono, dtype=np.float64).ravel()
    if x.size < 4096:
        return None

    # 三级窗口，与实时分析器一致。
    # 最后一层的上限必须超过最高中心频率（20158.7Hz），否则最高段会被漏掉 ——
    # 这正是之前 2k–20kHz 跨度出现 157dB 的原因（那一整段能量是 0）。
    tiers = ((120.0, 65536), (500.0, 16384), (float("inf"), 4096))
    out = np.zeros(N_BANDS)

    for upper, size in tiers:
        if x.size < size:
            # 数据不够长就用可用的最大 2 的幂
            size = 1 << int(np.floor(np.log2(x.size)))
            size = max(2048, size)
        win = np.hanning(size)
        wnorm = float(np.sum(win**2))
        freqs = np.fft.rfftfreq(size, d=1.0 / rate)
        bin_hz = rate / size

        # 50% 重叠分段
        step = size // 2
        acc = np.zeros(len(freqs))
        nseg = 0
        for start in range(0, x.size - size + 1, step):
            seg = x[start : start + size] * win
            spec = np.fft.rfft(seg)
            p = (spec.real**2 + spec.imag**2) / (wnorm * size)
            if len(p) > 2:
                p[1:-1] *= 2.0
            acc += p
            nseg += 1
        if nseg == 0:
            continue
        acc /= nseg

        # 只取该级负责的频段
        lo_lim = 19.0 if upper == 120.0 else (120.0 if upper == 500.0 else 500.0)
        for i in range(N_BANDS):
            fc = BAND_CENTERS[i]
            if not (lo_lim <= fc < upper):
                continue
            i0 = int(np.searchsorted(freqs, BAND_LO[i], side="left"))
            i1 = int(np.searchsorted(freqs, BAND_HI[i], side="right"))
            i0 = max(0, min(i0, len(freqs) - 1))
            i1 = max(i0 + 1, min(i1, len(freqs)))
            out[i] += float(np.sum(acc[i0:i1]))

    return out


# ---------------- 瞬态 / 音调性 ----------------

@dataclass
class TransientStats:
    """瞬态的统计特征，用来区分"连续声"和"脉冲声"。"""

    event_count: int
    onsets_per_sec: float
    steady_ratio: float      # 时间上"高于门限"的比例
    attack_ms_median: float


def transient_stats(mono: np.ndarray, rate: int) -> TransientStats:
    """用短时能量包络做简单的起音检测。"""
    x = np.asarray(mono, dtype=np.float64).ravel()
    hop = max(1, int(rate * 0.01))     # 10ms
    n = len(x) // hop
    if n < 4:
        return TransientStats(0, 0.0, 0.0, 0.0)

    frames = x[: n * hop].reshape(n, hop)
    rms = np.sqrt(np.mean(frames**2, axis=1))
    db = 20.0 * np.log10(np.maximum(rms, 1e-9))

    peak = float(np.max(db))
    floor = float(np.percentile(db, 10))

    # steady_ratio 的语义：有多少比例的时间"接近满活动"。
    # 不能简单地用 floor+门限来判定 —— 对恒幅信号（持续音），
    # floor 和 peak 几乎相等，`db > floor+6` 永远为假，
    # 会得出"活跃比例 0%"的错误结论，而它其实是 100% 活跃。
    # 所以门限固定取"距离峰值 20dB 以内"，并单独处理动态范围极小的情形。
    dynamic_range = peak - floor
    if dynamic_range < 6.0:
        # 动态范围太小说明这是一段稳态信号
        steady_ratio = 1.0 if peak > -80.0 else 0.0
        thresh = floor
    else:
        thresh = peak - 20.0
        steady_ratio = float(np.mean(db > thresh))

    above = db > thresh

    # 起音：从 false→true 的跳变
    rising = np.flatnonzero((~above[:-1]) & (above[1:])) if n > 1 else np.array([], int)
    onsets = int(len(rising))
    dur = len(x) / rate
    onsets_per_sec = onsets / dur if dur > 0 else 0.0

    # 起音时间：从跳变到达到局部峰值所需的帧数（10ms/帧）
    attacks = []
    for r in rising[:200]:
        j = r
        while j < n - 1 and db[j + 1] >= db[j]:
            j += 1
        attacks.append((j - r) * 10.0)
    attack_med = float(np.median(attacks)) if attacks else 0.0

    return TransientStats(
        event_count=onsets,
        onsets_per_sec=round(onsets_per_sec, 2),
        steady_ratio=round(steady_ratio, 3),
        attack_ms_median=round(attack_med, 1),
    )


def tonality_index(mono: np.ndarray, rate: int, size: int = 8192) -> float:
    """谱峰锐度：0 = 纯宽带噪声，越大越接近"有音高"的离散峰。

    用谱的峰度（kurtosis）做代理：宽带噪声接近高斯（峰度≈3），
    音调性信号在少数 bin 上能量集中，峰度显著更高。
    """
    x = np.asarray(mono, dtype=np.float64).ravel()
    if x.size < size:
        return 0.0
    win = np.hanning(size)
    seg = x[x.size // 2 - size // 2 : x.size // 2 + size // 2] * win
    p = np.abs(np.fft.rfft(seg)) ** 2
    p = p[1:]
    if p.sum() <= 0:
        return 0.0
    p = p / p.sum()
    m = float(np.mean(p))
    if m <= 0:
        return 0.0
    k = float(np.mean((p - m) ** 4) / (m**4))
    # 映射到便于阅读的量级
    return round(float(np.log10(max(k, 1e-9))), 2)


# ---------------- 单文件 / 多文件汇总 ----------------

@dataclass
class ClipAnalysis:
    key: str
    name: str
    path: str
    seconds: float
    rate: int
    channels: int
    peak_db: float
    rms_db: float
    noise_floor_db: float
    clipped: bool
    silent: bool
    peak_hz: float
    spectrum_db: np.ndarray | None
    transients: TransientStats | None
    tonality: float
    stereo_corr: float | None
    digital_silence_ratio: float = 0.0   # 数字静音帧占比（loopback 无流时为 0 值）

    def header(self) -> str:
        return (
            f"{self.name}  ({Path(self.path).name})\n"
            f"  时长 {self.seconds:.2f}s · {self.rate}Hz · {self.channels}ch · "
            f"峰值 {self.peak_db:.1f}dB · RMS {self.rms_db:.1f}dB · "
            f"噪声底 {self.noise_floor_db:.1f}dB"
        )


def analyze_file(path: str, key: str, name: str) -> ClipAnalysis:
    data, rate = read_wav(path)
    mono = data.mean(axis=1)
    dur = len(mono) / rate

    peak = float(np.max(np.abs(data)))
    rms = float(np.sqrt(np.mean(mono.astype(np.float64) ** 2)))
    peak_db = 20 * np.log10(max(peak, 1e-9))
    rms_db = 20 * np.log10(max(rms, 1e-9))

    # 噪声底：短时 RMS 的 10 分位，但**必须排除数字静音帧**。
    # WASAPI loopback 在没有音频流经过时会输出精确的 0.0，
    # 直接取分位会得到 -180dB 这种"比量化噪声还低"的假值，
    # 让人误以为底噪极低。只统计"有内容"的帧才有物理意义。
    hop = max(1, int(rate * 0.02))
    n = max(1, len(mono) // hop)
    frames = mono[: n * hop].reshape(n, hop)
    fr = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    live = fr[fr > 1e-6]           # 排除数字静音
    if live.size >= 3:
        noise_floor_db = float(20 * np.log10(max(float(np.percentile(live, 10)), 1e-9)))
        digital_silence_ratio = float(1.0 - live.size / fr.size)
    else:
        noise_floor_db = -120.0
        digital_silence_ratio = 1.0

    spec = band_power_spectrum(mono, rate)
    spec_db = 10 * np.log10(np.maximum(spec, 1e-20)) if spec is not None else None

    peak_hz = 0.0
    if spec_db is not None and np.any(np.isfinite(spec_db)):
        peak_hz = float(BAND_CENTERS[int(np.argmax(spec_db))])

    stereo_corr = None
    if data.shape[1] >= 2:
        l, r = data[:, 0].astype(np.float64), data[:, 1].astype(np.float64)
        if l.std() > 1e-9 and r.std() > 1e-9:
            stereo_corr = round(float(np.corrcoef(l, r)[0, 1]), 3)

    return ClipAnalysis(
        key=key,
        name=name,
        path=path,
        seconds=dur,
        rate=rate,
        channels=data.shape[1],
        peak_db=peak_db,
        rms_db=rms_db,
        noise_floor_db=noise_floor_db,
        clipped=peak >= 0.999,
        silent=peak < 1e-4,
        peak_hz=peak_hz,
        spectrum_db=spec_db,
        transients=transient_stats(mono, rate),
        tonality=tonality_index(mono, rate),
        stereo_corr=stereo_corr,
        digital_silence_ratio=digital_silence_ratio,
    )


@dataclass
class RawReport:
    """汇总报告：交给 AI 判断的全部事实。"""

    clips: list[ClipAnalysis] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def by_key(self, key: str) -> ClipAnalysis | None:
        return next((c for c in self.clips if c.key == key), None)

    # ---- 场景分组访问 ----
    @staticmethod
    def _loc(key: str) -> str:
        """从 key 取位置部分：'near:run' → 'near'，'fire' → 'fire'。"""
        return key.split(":", 1)[0]

    @staticmethod
    def _gait(key: str) -> str | None:
        """从 key 取移动方式：'near:run' → 'run'，'fire' → None。"""
        return key.split(":", 1)[1] if ":" in key else None

    def _mean_of(self, locations: tuple[str, ...],
                 gaits: tuple[str, ...] | None = None) -> np.ndarray | None:
        """按位置（可选再按移动方式）筛选并功率域平均。"""
        arrs = []
        for c in self.clips:
            if c.spectrum_db is None:
                continue
            if self._loc(c.key) not in locations:
                continue
            if gaits is not None and self._gait(c.key) not in gaits:
                continue
            arrs.append(c.spectrum_db)
        return _power_mean(arrs)

    def enemy_step_spectrum(self, gaits: tuple[str, ...] | None = None) -> np.ndarray | None:
        """敌人脚步（旁边/楼下/远处）的合并频谱。"""
        return self._mean_of(("near", "below", "far"), gaits)

    def own_step_spectrum(self, gaits: tuple[str, ...] | None = None) -> np.ndarray | None:
        """自己脚步的频谱。"""
        return self._mean_of(("own",), gaits)

    def gunshot_spectrum(self) -> np.ndarray | None:
        """枪声类干扰的合并频谱。"""
        return self._mean_of(("fire", "grenade", "flash"))

    # 兼容旧名字
    def keep_spectrum(self) -> np.ndarray | None:
        return self.enemy_step_spectrum()

    def reduce_spectrum(self) -> np.ndarray | None:
        return self.gunshot_spectrum()

    def idle_spectrum(self) -> np.ndarray | None:
        c = self.by_key("idle")
        return c.spectrum_db if c else None

    def distance_groups(self, gaits: tuple[str, ...] | None = None) -> dict[str, np.ndarray]:
        """按位置分别取频谱（同一位置若有多种移动方式则合并）。"""
        out: dict[str, np.ndarray] = {}
        for loc in ("near", "below", "far"):
            arr = self._mean_of((loc,), gaits)
            if arr is not None:
                out[loc] = arr
        return out

    def gait_groups(self) -> dict[str, dict[str, np.ndarray]]:
        """按位置再按移动方式拆分，用于对比「静步/正常/奔跑」的频谱差异。

        同一个 (位置, 方式) 若录了多段（比如补录过），必须**功率域合并**，
        不能后一段直接覆盖前一段 —— 否则补录时音量稍小就会把整组数据拉低，
        距离衰减和步态对比的结论都会跟着错。
        """
        out: dict[str, dict[str, np.ndarray]] = {}
        for loc in ("own", "near", "below", "far"):
            buckets: dict[str, list[np.ndarray]] = {}
            for c in self.clips:
                g = self._gait(c.key)
                if g is None or self._loc(c.key) != loc or c.spectrum_db is None:
                    continue
                buckets.setdefault(g, []).append(c.spectrum_db)
            sub: dict[str, np.ndarray] = {}
            for g, arrs in buckets.items():
                merged = _power_mean(arrs)
                if merged is not None:
                    sub[g] = merged
            if sub:
                out[loc] = sub
        return out

    def gait_coverage(self) -> dict[str, list[str]]:
        """位置 → 该位置已录到的移动方式。报告开头用它提示还缺哪些组合。"""
        out: dict[str, list[str]] = {}
        order = ("sneak", "walk", "run")
        for loc, sub in self.gait_groups().items():
            out[loc] = [g for g in order if g in sub]
        return out


def _power_mean(arrs: list[np.ndarray]) -> np.ndarray | None:
    if not arrs:
        return None
    p = np.mean([10.0 ** (a / 10.0) for a in arrs], axis=0)
    return 10.0 * np.log10(np.maximum(p, 1e-20))


def build_report(clips: dict[str, tuple[str, str]]) -> RawReport:
    """clips: {key: (path, name)}"""
    rep = RawReport()
    for key, (path, name) in clips.items():
        try:
            rep.clips.append(analyze_file(path, key, name))
        except Exception as exc:
            rep.notes.append(f"{name}: 分析失败 {type(exc).__name__}: {exc}")
    return rep
