"""自测：分析器正确性（61 段 / 20Hz–20kHz / 三级多分辨率）。"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from app.analyzer import BAND_CENTERS, N_BANDS, SpectrumAnalyzer, format_hz  # noqa: E402

SR = 48000
FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def feed(freq: float, amp: float = 0.3, seconds: float = 3.0) -> SpectrumAnalyzer:
    an = SpectrumAnalyzer(samplerate=SR, release_ms=30)
    n = int(SR * seconds)
    t = np.arange(n) / SR
    sig = (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    t0 = time.perf_counter()
    for i in range(0, len(sig) - 4800, 4800):
        an.process(sig[i : i + 4800], t0 + i / SR)
    return an


print("=" * 66)
print("EQ Band Monitor — 分析器自测（61 段 / 1/6 倍频程）")
print("=" * 66)

# ---------- 1. 频段结构 ----------
print("\n[1] 频段结构")
check("共 61 段", N_BANDS == 61, f"实际 {N_BANDS}")
check("严格递增", bool(np.all(np.diff(BAND_CENTERS) > 0)))
check("下限 ≈ 20Hz", 19.0 <= BAND_CENTERS[0] <= 20.5, f"{BAND_CENTERS[0]:.1f}")
check("上限 ≈ 20kHz", 19900 <= BAND_CENTERS[-1] <= 20300, f"{BAND_CENTERS[-1]:.1f}")
ratios = np.diff(np.log2(BAND_CENTERS))
check("相邻段间隔均匀（1/6 倍频程）", float(np.max(ratios) - np.min(ratios)) < 1e-9,
      f"间隔 {ratios[0]*6:.3f}/6 倍频程")

# ---------- 2. 多分辨率级 ----------
print("\n[2] 三级多分辨率")
an = SpectrumAnalyzer(samplerate=SR)
tiers = an._tiers
check("共 3 级", len(tiers) == 3)
check("最低级用 65536 窗", tiers[0].size == 65536, f"{tiers[0].size}")
check("中频级用 16384 窗", tiers[1].size == 16384, f"{tiers[1].size}")
check("高频级用 4096 窗", tiers[2].size == 4096, f"{tiers[2].size}")
covered = sum(len(t.band_idx) for t in tiers)
check("所有频段都被分配到某一级", covered == N_BANDS, f"{covered}/{N_BANDS}")
check("最低级 bin 宽 < 1Hz", tiers[0].bin_hz < 1.0, f"{tiers[0].bin_hz:.2f}Hz")
check("20Hz 段至少 2 个 bin",
      int(tiers[0].bins_per_band[0]) >= 2, f"{int(tiers[0].bins_per_band[0])} bin")

# ---------- 3. 峰值精度（含极低频）----------
print("\n[3] 峰值频点精度（目标 < 1%）")
for f in (25, 30, 40, 63, 120, 440, 1000, 2500, 8000, 15000):
    a = feed(float(f))
    err = abs(a.peak_hz - f) / f * 100
    check(f"{f:>6} Hz → {a.peak_hz:8.1f} Hz", err < 1.0, f"误差 {err:.2f}%")

# ---------- 4. 频段定位 ----------
print("\n[4] 能量落在正确频段")
for f in (25, 63, 250, 1000, 4000, 12000):
    a = feed(float(f))
    got = int(np.argmax(a.bands_db))
    want = int(np.argmin(np.abs(BAND_CENTERS - f)))
    check(f"{f:>5} Hz → {format_hz(BAND_CENTERS[got]):>7}",
          abs(got - want) <= 1, f"期望 {format_hz(BAND_CENTERS[want])}")

# ---------- 5. 粉噪应平直（跨级一致性，最关键）----------
print("\n[5] 粉噪在中高频应平直（验证各级归一化一致）")
a = SpectrumAnalyzer(samplerate=SR, release_ms=50)
N = SR * 6
rng = np.random.default_rng(7)
spec = np.fft.rfft(rng.normal(0, 1, N))
fr = np.fft.rfftfreq(N, 1 / SR)
spec[1:] /= np.sqrt(np.maximum(fr[1:], 1.0))
pink = np.fft.irfft(spec, N)
pink = (pink / np.max(np.abs(pink)) * 0.3).astype(np.float32)
t0 = time.perf_counter()
acc = []
for i in range(0, len(pink) - 4800, 4800):
    a.process(pink[i : i + 4800], t0 + i / SR)
    if i > 4800 * 30:
        acc.append(a.bands_db.copy())
mean_db = 10 * np.log10(np.mean(10 ** (np.array(acc) / 10), axis=0))
hi = mean_db[BAND_CENTERS >= 2000]
span_hi = float(hi.max() - hi.min())
check("2k–20kHz 跨度 < 6dB", span_hi < 6.0, f"实测 {span_hi:.1f} dB")
mid = mean_db[(BAND_CENTERS >= 200) & (BAND_CENTERS < 2000)]
span_mid = float(mid.max() - mid.min())
check("200Hz–2kHz 跨度 < 8dB", span_mid < 8.0, f"实测 {span_mid:.1f} dB")

# ---------- 6. 电平线性（EQ 增益可 1:1 映射的前提）----------
print("\n[6] 幅度加倍 → +6dB")
a1 = feed(1000.0, amp=0.1)
a2 = feed(1000.0, amp=0.2)
bi = int(np.argmin(np.abs(BAND_CENTERS - 1000.0)))
d = a2.bands_db[bi] - a1.bands_db[bi]
check(f"差值 {d:.2f} dB ≈ 6", abs(d - 6.0) < 0.5, f"偏差 {abs(d-6.0):.2f}")

# ---------- 7. 静音衰减 ----------
print("\n[7] 静音后衰减")
a = feed(1000.0)
loud = float(np.max(a.bands_db))
t0 = time.perf_counter()
for i in range(300):
    a.process(np.zeros(4800, dtype=np.float32), t0 + i * 0.05 + 2.0)
check(f"{loud:.0f}dB → {float(np.max(a.bands_db)):.0f}dB",
      float(np.max(a.bands_db)) < a.min_db + 5.0)
check("has_signal=False", not a.has_signal)

# ---------- 8. 性能 ----------
print("\n[8] 性能")
a = SpectrumAnalyzer(samplerate=SR)
blk = (0.2 * np.random.default_rng(1).normal(size=4800)).astype(np.float32)
for _ in range(5):
    a.process(blk)
NIT = 300
t0 = time.perf_counter()
for _ in range(NIT):
    a.process(blk)
per = (time.perf_counter() - t0) / NIT * 1000
check(f"{per:.2f} ms/帧", per < 6.0, f"20fps 占单核 {per/50*100:.1f}%")

print("\n" + "=" * 66)
if FAILS:
    print(f"结果：{len(FAILS)} 项失败 → {', '.join(FAILS)}")
    sys.exit(1)
print("结果：全部通过 ✓")
print("=" * 66)
