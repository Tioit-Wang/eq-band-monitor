"""频谱分析层：多分辨率 FFT → 1/6 倍频程聚合 → 时间平滑 → 峰值插值。

===== 为什么这样设计 =====

1) 范围 20Hz–20kHz、分辨率 1/6 倍频程（61 段）
   旧的 1/3 倍频程只有 29 段、25Hz–16kHz，太粗：FxSound 的 10 个频段
   可以自由落在任意频率（实测用户预设里有 540Hz 和 642Hz 两个相邻点），
   29 段的粒度根本无法支撑"自由频段拟合"。

2) 三级多分辨率 FFT —— 这是必须付出的代价
   1/6 倍频程带宽 = fc·(2^(1/12) − 2^(−1/12)) ≈ 0.115·fc
     20Hz  → 带宽 2.3Hz
     100Hz → 带宽 11.5Hz
     1kHz  → 带宽 115Hz
   48kHz 采样下 bin 宽 = 48000/N：
     N=4096   → 11.7Hz   （20Hz 段只有 0.2 个 bin，完全不可用）
     N=16384  → 2.93Hz   （20Hz 段 0.8 bin，仍不可用）
     N=65536  → 0.73Hz   （20Hz 段 3.1 bin，勉强够）
   所以按频段分三级窗口。低频用长窗的代价是时间分辨率差，
   但低频声本来就接近稳态（脚步的低频"砰"、枪声的低频"轰"），可以接受。

3) 低频级降频更新
   65536 点 rfft 约 1.5ms，每帧都算会把 CPU 翻倍。
   低频内容变化慢，所以每 3 帧更新一次（约 150ms），误差不可感知。

4) 不做粉噪补偿
   1/6 倍频程带内粉噪的能量每段本来就相等；而带能量与 EQ 增益是 1:1 关系
   （EQ 抬 6dB，该段读数就 +6dB，已由 selftest 验证偏差 0.02dB），
   这正好让"读数差"可以直接翻译成"EQ 增益"。
"""

from __future__ import annotations

import math

import numpy as np

# ---- 1/6 倍频程中心频率，20Hz–20kHz（以 1000Hz 为锚点，标准做法）----
# 1000·2^(n/6)，n = -34..26 → 19.69Hz .. 20142Hz，共 61 段
BAND_CENTERS = 1000.0 * 2.0 ** (np.arange(-34, 27) / 6.0)
N_BANDS = len(BAND_CENTERS)

# 频段边界（1/6 倍频程）：fc·2^(±1/12)
BAND_LO = BAND_CENTERS * 2.0 ** (-1.0 / 12.0)
BAND_HI = BAND_CENTERS * 2.0 ** (1.0 / 12.0)

# 三级窗口：(该级覆盖到的最高中心频率, FFT 长度)
# 注意顺序：低频在前，匹配时取第一个 fc < 上限的级
TIERS: tuple[tuple[float, int], ...] = (
    (120.0, 65536),    # 极低频：0.73Hz bin
    (500.0, 16384),    # 低中频：2.93Hz bin
    (20001.0, 4096),   # 中高频：11.7Hz bin
)

# 低频级每 N 帧才更新一次
LOW_TIER_EVERY = 3


def format_hz(f: float) -> str:
    """1000 → '1k'，1250 → '1.25k'，412 → '412'，19.7 → '19.7'。"""
    if f >= 1000:
        s = f"{f / 1000.0:.2f}".rstrip("0").rstrip(".")
        return s + "k"
    if f >= 100:
        return f"{f:.0f}"
    return f"{f:.1f}".rstrip("0").rstrip(".")


def _db(power) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(power, 1e-14))


def _band_slice(freqs: np.ndarray, lo: float, hi: float) -> slice:
    i0 = int(np.searchsorted(freqs, lo, side="left"))
    i1 = int(np.searchsorted(freqs, hi, side="right"))
    i0 = max(0, min(i0, len(freqs) - 1))
    i1 = max(i0 + 1, min(i1, len(freqs)))
    return slice(i0, i1)


class _Tier:
    """一级 FFT：自己的窗口、频谱栅格、历史缓冲、求和矩阵。"""

    def __init__(self, size: int, samplerate: int, band_idx: np.ndarray) -> None:
        self.size = size
        self.samplerate = samplerate
        self.band_idx = band_idx

        self.win = np.hanning(size)
        self.win_norm = float(np.sum(self.win**2))
        self.freqs = np.fft.rfftfreq(size, d=1.0 / samplerate)
        self.bin_hz = samplerate / size

        self.hist = np.zeros(size)

        # 该级负责的频段的 bin 区间与 0/1 求和矩阵
        self.slices = [
            _band_slice(self.freqs, BAND_LO[i], BAND_HI[i]) for i in band_idx
        ]
        m = np.zeros((len(self.slices), len(self.freqs)))
        for row, sl in enumerate(self.slices):
            m[row, sl] = 1.0
        self.summation = m
        self.bins_per_band = np.array([s.stop - s.start for s in self.slices])

    def push(self, blk: np.ndarray) -> None:
        n = len(blk)
        if n >= self.size:
            self.hist = blk[-self.size :].astype(np.float64, copy=True)
        else:
            self.hist = np.concatenate([self.hist[n:], blk])

    def power(self) -> np.ndarray:
        spec = np.fft.rfft(self.hist * self.win)
        p = (spec.real**2 + spec.imag**2) / (self.win_norm * self.size)
        if len(p) > 2:
            p[1:-1] *= 2.0  # 单边谱
        return p


class SpectrumAnalyzer:
    """把时域块转换成 61 段 1/6 倍频程 dB 值。"""

    def __init__(
        self,
        samplerate: int = 48000,
        release_ms: float = 200.0,
        min_db: float = -90.0,
        max_db: float = 0.0,
    ) -> None:
        self.samplerate = samplerate
        self.release_ms = release_ms
        self.min_db = min_db
        self.max_db = max_db

        # 按中心频率把频段分配到各级
        assign = np.full(N_BANDS, -1, dtype=int)
        for tier_i, (upper_hz, _size) in enumerate(TIERS):
            mask = (assign < 0) & (BAND_CENTERS < upper_hz)
            assign[mask] = tier_i
        assign[assign < 0] = len(TIERS) - 1
        self._assign = assign

        self._tiers = [
            _Tier(size, samplerate, np.flatnonzero(assign == i))
            for i, (_u, size) in enumerate(TIERS)
        ]

        # 状态
        self.bands_db = np.full(N_BANDS, min_db)
        self.inst_bands_db = np.full(N_BANDS, min_db)
        self.peak_hold = np.full(N_BANDS, min_db)
        self.sample_peak_lin = 0.0
        self.loudness_db = min_db
        self.peak_hz = 0.0
        self.peak_db = min_db
        self.has_signal = False

        self._signal_until = 0.0
        self._last_t = 0.0
        self._frame = 0
        self._low_power: np.ndarray | None = None

    # ---------------- 主流程 ----------------

    def process(self, block: np.ndarray, now: float | None = None) -> None:
        import time

        t = time.perf_counter() if now is None else now

        if block is None or len(block) == 0:
            self._decay(t)
            return

        blk = np.asarray(block, dtype=np.float64).ravel()
        self.sample_peak_lin = float(np.max(np.abs(blk)))

        self._frame += 1
        for i, tier in enumerate(self._tiers):
            tier.push(blk)

        # 各级功率谱。最低级降频更新。
        powers: list[np.ndarray | None] = []
        for i, tier in enumerate(self._tiers):
            if i == 0 and self._low_power is not None and (self._frame % LOW_TIER_EVERY):
                powers.append(self._low_power)
            else:
                p = tier.power()
                if i == 0:
                    self._low_power = p
                powers.append(p)

        band_power = np.empty(N_BANDS)
        for i, tier in enumerate(self._tiers):
            idx = tier.band_idx
            if idx.size == 0:
                continue
            p = powers[i]
            # 带内总能量。
            # 注意：**不能**再除以 bin 数。P(f)·df 才是物理能量，而各级的
            # df 不同（0.73 / 2.93 / 11.7 Hz），除以 bin 数会让不同级被不同地
            # 缩放，结果就是整条曲线呈系统性倾斜（实测粉噪跨度达 19.8dB）。
            # 由于各段带宽都是 1/6 倍频程，带内总能量本身就具有可比性。
            band_power[idx] = tier.summation @ p

        inst_db = _db(band_power)
        self.inst_bands_db = inst_db.copy()
        self.loudness_db = float(_db(np.array([float(np.mean(band_power))]))[0])

        # ---- 时间包络：快 attack / 慢 release ----
        dt = max(1e-3, t - self._last_t) if self._last_t else 1.0 / 20.0
        self._last_t = t
        tau = max(1e-3, self.release_ms / 1000.0)
        decay = math.exp(-dt / tau)

        prev = self.bands_db
        rise = inst_db > prev
        fall = inst_db + (prev - inst_db) * decay
        self.bands_db = np.where(rise, inst_db, fall)
        np.clip(self.bands_db, self.min_db, self.max_db, out=self.bands_db)
        np.maximum(self.peak_hold, self.bands_db, out=self.peak_hold)

        # 信号判定带 300ms 保持：脉冲音之间有静音隙，逐帧判定会让读数闪跳
        if self.sample_peak_lin > 1e-4:
            self._signal_until = t + 0.3
        self.has_signal = t < self._signal_until

        self._update_peak(powers)

    def _decay(self, t: float) -> None:
        dt = max(1e-3, t - self._last_t) if self._last_t else 0.05
        self._last_t = t
        np.maximum(self.bands_db - 60.0 * dt, self.min_db, out=self.bands_db)
        np.minimum(self.inst_bands_db, self.bands_db, out=self.inst_bands_db)
        self.sample_peak_lin *= 0.5
        self.has_signal = t < self._signal_until

    def _update_peak(self, powers: list[np.ndarray | None]) -> None:
        """在 20Hz–20kHz 内找最强谱峰，用所在级的分辨率做抛物线插值。"""
        best_val = -1e30
        best_hz = 0.0

        for i, tier in enumerate(self._tiers):
            p = powers[i]
            if p is None or len(p) < 8:
                continue
            lo_hz = 20.0 if i == 0 else TIERS[i - 1][0]
            hi_hz = TIERS[i][0]
            j0 = max(1, int(np.searchsorted(tier.freqs, lo_hz)))
            j1 = min(len(p) - 1, int(np.searchsorted(tier.freqs, hi_hz)))
            if j1 <= j0 + 1:
                continue
            k = int(np.argmax(p[j0:j1])) + j0
            val = float(p[k])
            if val > best_val:
                best_val = val
                best_hz = self._parabolic(p, k, tier.bin_hz)

        if best_val > 0:
            self.peak_hz = best_hz
            self.peak_db = float(_db(np.array([best_val]))[0])

    @staticmethod
    def _parabolic(p: np.ndarray, k: int, bin_hz: float) -> float:
        if k <= 0 or k >= len(p) - 1:
            return k * bin_hz
        a, b, c = float(p[k - 1]), float(p[k]), float(p[k + 1])
        denom = a - 2.0 * b + c
        delta = 0.5 * (a - c) / denom if abs(denom) > 1e-20 else 0.0
        return (k + float(np.clip(delta, -0.5, 0.5))) * bin_hz

    # ---------------- 便捷访问 ----------------

    def reset_peak_hold(self) -> None:
        self.peak_hold[:] = self.min_db
        self.sample_peak_lin = 0.0

    def top_bands(self, n: int = 5) -> list[tuple[float, float]]:
        order = np.argsort(self.bands_db)[::-1][:n]
        return [
            (float(BAND_CENTERS[i]), float(self.bands_db[i]))
            for i in order
            if self.bands_db[i] > self.min_db + 1e-6
        ]

    def normalized(self) -> np.ndarray:
        span = max(1e-6, self.max_db - self.min_db)
        return np.clip((self.bands_db - self.min_db) / span, 0.0, 1.0)

    def level_db(self) -> float:
        p = float(np.mean(10.0 ** (self.bands_db / 10.0)))
        return float(10.0 * math.log10(max(p, 1e-14)))
