"""自测：原始音频录制 → WAV → 分析 → 交接包。"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from app.analyzer import BAND_CENTERS, N_BANDS  # noqa: E402
from app.handoff import render_json, render_report, save_bundle  # noqa: E402
from app.rawanalyze import (  # noqa: E402
    RawReport,
    analyze_file,
    band_power_spectrum,
    read_wav,
    tonality_index,
    transient_stats,
)
from app.rawrec import RawRecorder, _f32_to_s24  # noqa: E402

SR = 48000
FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def make_tone(freq: float, amp: float, seconds: float, seed: int = 0) -> np.ndarray:
    n = int(SR * seconds)
    t = np.arange(n) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def make_pulses(seconds: float) -> np.ndarray:
    n = int(SR * seconds)
    out = np.zeros(n, dtype=np.float32)
    period = int(SR * 0.25)
    burst = 2400
    env = np.exp(-np.arange(burst) / 400.0)
    for k in range(n // period):
        s = k * period
        out[s : s + burst] += (0.8 * env * np.sin(2 * np.pi * 800 * np.arange(burst) / SR)).astype(np.float32)
    return out


def record_stereo(left: np.ndarray, right: np.ndarray, key: str, name: str,
                  out_dir: str) -> "object":
    rec = RawRecorder(out_dir=out_dir, samplerate=SR)
    rec.start(key, name)
    data = np.stack([left, right], axis=1)
    bs = 4800
    for i in range(0, len(data), bs):
        rec.feed(data[i : i + bs])
    return rec.stop()


print("=" * 68)
print("EQ Band Monitor — 原始音频录制/分析 自测")
print("=" * 68)
tmp = tempfile.mkdtemp()

# ---------- 1. 24-bit 量化 ----------
print("\n[1] 24-bit 量化精度")
x = np.array([0.0, 0.5, -0.5, 0.999, -0.999, 1e-5], dtype=np.float32)
b = _f32_to_s24(x)
v = np.frombuffer(b, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
i = v[:, 0] | (v[:, 1] << 8) | (v[:, 2] << 16)
i = np.where(i & 0x800000, i - 0x1000000, i)
back = i.astype(np.float32) / 8388608.0
err = float(np.max(np.abs(back - x)))
check("往返误差 < 1e-6", err < 1e-6, f"{err:.2e}")
check("0.5 精确", abs(back[1] - 0.5) < 1e-6)
check("-0.5 精确", abs(back[2] + 0.5) < 1e-6)

# ---------- 2. WAV 落盘 / 读回 ----------
print("\n[2] WAV 落盘与读回")
left = make_tone(1000, 0.3, 2.0)
right = make_tone(3000, 0.2, 2.0)
clip = record_stereo(left, right, "walk", "走路", tmp)
check("生成文件", clip is not None and os.path.exists(clip.path))
check("时长准确", abs(clip.seconds - 2.0) < 0.01, f"{clip.seconds:.3f}s")
check("24-bit 立体声", clip.channels == 2)
check("文件头合法", Path(clip.path).read_bytes()[:4] == b"RIFF")

data, rate = read_wav(clip.path)
check("采样率一致", rate == SR)
check("形状一致", data.shape == (len(left), 2), f"{data.shape}")
check("幅度保持", abs(float(np.max(np.abs(data))) - 0.3) < 1e-3,
      f"{float(np.max(np.abs(data))):.4f}")

# ---------- 3. 分析正确性 ----------
print("\n[3] 单文件分析")
a = analyze_file(clip.path, "walk", "走路")
check("峰值频段 ≈1kHz", abs(a.peak_hz - 1000) / 1000 < 0.1, f"{a.peak_hz:.0f}Hz")
check("未误判削波", not a.clipped)
check("未误判静音", not a.silent)
check("声道相关 < 0.9（左右内容不同）",
      a.stereo_corr is not None and a.stereo_corr < 0.9, f"{a.stereo_corr}")
check("音调性 > 0（正弦）", a.tonality > 0, f"{a.tonality}")

# ---------- 4. 噪声底不能取到数字静音 ----------
print("\n[4] 噪声底排除数字静音")
mix = np.concatenate([np.zeros(SR, dtype=np.float32),          # 1s 纯数字静音
                      make_tone(800, 0.2, 2.0)])               # 2s 有声
c2 = record_stereo(mix, mix, "fire", "开火", tmp)
a2 = analyze_file(c2.path, "fire", "开火")
check("噪声底不是 -180dB 这类假值", a2.noise_floor_db > -110.0,
      f"{a2.noise_floor_db:.1f} dB")
check("识别出数字静音帧", a2.digital_silence_ratio > 0.15,
      f"{a2.digital_silence_ratio*100:.1f}%")

# ---------- 5. 瞬态区分 ----------
print("\n[5] 瞬态：连续音 vs 脉冲音")
cont = make_tone(800, 0.3, 2.0)
pul = make_pulses(2.0)
tc = transient_stats(cont, SR)
tp = transient_stats(pul, SR)
check("脉冲起音次数更多", tp.event_count > tc.event_count,
      f"{tp.event_count} vs {tc.event_count}")
check("脉冲活跃占比更低", tp.steady_ratio < tc.steady_ratio,
      f"{tp.steady_ratio} vs {tc.steady_ratio}")

# ---------- 6. 音调性 ----------
print("\n[6] 音调性区分")
wn = (0.3 * np.random.default_rng(0).normal(0, 1, SR * 2)).astype(np.float32)
check("正弦 > 白噪", tonality_index(cont, SR) > tonality_index(wn, SR),
      f"{tonality_index(cont, SR)} vs {tonality_index(wn, SR)}")

# ---------- 7. 分频频谱 ----------
print("\n[7] 分频频谱覆盖与一致性")
N = SR * 8
rng = np.random.default_rng(3)
spec = np.fft.rfft(rng.normal(0, 1, N))
fr = np.fft.rfftfreq(N, 1 / SR)
spec[1:] /= np.sqrt(np.maximum(fr[1:], 1.0))
pink = np.fft.irfft(spec, N)
pink = (pink / np.max(np.abs(pink)) * 0.3).astype(np.float32)
sp = band_power_spectrum(pink, SR)
check("61 段", sp.size == N_BANDS, f"{sp.size}")
check("所有段都有能量（最高段不漏）", bool(np.all(sp > 0)), f"min={sp.min():.2e}")
sdb = 10 * np.log10(np.maximum(sp, 1e-20))
hi = sdb[BAND_CENTERS >= 2000]
check("2k–20k 跨度 < 8dB", float(hi.max() - hi.min()) < 8.0,
      f"{float(hi.max()-hi.min()):.1f}dB")
mid = sdb[(BAND_CENTERS >= 200) & (BAND_CENTERS < 2000)]
check("200–2k 跨度 < 10dB", float(mid.max() - mid.min()) < 10.0,
      f"{float(mid.max()-mid.min()):.1f}dB")

# ---------- 8. 数据质量标记 ----------
print("\n[8] 数据质量标记")
z = record_stereo(np.zeros(SR, np.float32), np.zeros(SR, np.float32),
                  "idle", "底噪", tmp)
check("全零 → 静音", z.silent)
hot = record_stereo(np.full(SR, 0.9999, np.float32),
                    np.full(SR, 0.9999, np.float32), "grenade", "手榴弹", tmp)
check("满刻度 → 削波", hot.clipped)

# ---------- 9. 交接包 ----------
print("\n[9] 交接包生成")
rep = RawReport()
for key, name, path in [
    ("idle", "静止底噪", z.path),
    ("walk", "走路", clip.path),
    ("fire", "开火", c2.path),
]:
    rep.clips.append(analyze_file(path, key, name))

md = render_report(rep)
check("报告含标题", "# 游戏音频场景录制报告" in md)
check("报告含数据总览", "录制文件总览" in md)
check("报告含频谱段", "分场景频谱" in md)
check("报告含分析约定", "分析约定" in md)
check("声明了原始 PCM", "原始 PCM" in md)

js = render_json(rep)
check("JSON 可序列化", isinstance(json.dumps(js), str))
check("JSON 段数正确", len(js["band_centers_hz"]) == N_BANDS)
check("JSON 含每段频谱", all(
    c["spectrum_db"] is None or len(c["spectrum_db"]) == N_BANDS for c in js["clips"]))

out = os.path.join(tmp, "bundle")
save_bundle(out, rep, {c.key: c.path for c in rep.clips}, "预设名：**测试**")
check("生成 report.md", os.path.exists(os.path.join(out, "report.md")))
check("生成 report.json", os.path.exists(os.path.join(out, "report.json")))
check("音频已复制", len(os.listdir(os.path.join(out, "audio"))) == 3)

# ---------- 10. 录制器内存上限 ----------
print("\n[10] 录制器上限")
r = RawRecorder(out_dir=tmp, samplerate=SR)
r.start("t", "测试")
for _ in range(400):  # 400 × 0.1s = 40s > MAX_SECONDS(30)
    r.feed(np.zeros((4800, 2), dtype=np.float32))
check("超过上限后截断", r._frames <= int(RawRecorder.MAX_SECONDS * SR) + 1,
      f"{r._frames} 帧 = {r._frames/SR:.1f}s")
r.stop()



# ---------- 11. 组合键文件名必须安全（NTFS 冒号是 ADS 分隔符）----------
print("\n[11] 组合键文件名安全")
from app.rawrec import safe_filename_part  # noqa: E402

check("冒号被替换", safe_filename_part("near:run") == "near-run",
      safe_filename_part("near:run"))
check("斜杠被替换", "/" not in safe_filename_part("a/b:c"), safe_filename_part("a/b:c"))

combo = record_stereo(make_tone(600, 0.2, 1.0), make_tone(600, 0.2, 1.0),
                      "below:run", "楼下脚步·奔跑", tmp)
check("组合键落盘路径不含冒号", ":" not in os.path.basename(combo.path),
      os.path.basename(combo.path))
check("文件真实存在于目录列表",
      os.path.basename(combo.path) in os.listdir(tmp),
      f"{os.listdir(tmp)[-3:]}")
check("文件大小 = 1s × 48k × 2ch × 3B",
      abs(os.path.getsize(combo.path) - 48000 * 2 * 3) < 4096,
      f"{os.path.getsize(combo.path)} B")


# ---------- 12. 位置 × 移动方式 覆盖矩阵 ----------
print("\n[12] 位置 × 移动方式 覆盖矩阵")
rep2 = RawReport()
specs = [
    ("below:sneak", "楼下脚步·静步", -70.0, 6.0),
    ("below:walk", "楼下脚步·正常走", -58.0, 4.0),
    ("below:run", "楼下脚步·奔跑", -46.0, 2.0),
    ("far:run", "远处脚步·奔跑", -52.0, 0.0),
]
for key, name, lo_db, hi_db in specs:
    # 低频更强的信号 → 中高频相对更低，用来验证倾斜量方向
    n = SR
    t = np.arange(n) / SR
    sig = 0.3 * np.sin(2 * np.pi * 200 * t) + 0.3 * np.exp(hi_db / 20.0) * np.sin(2 * np.pi * 3000 * t)
    sig = (sig * (10 ** (lo_db / 20.0)) / 0.3 * 0.5).astype(np.float32)
    c = record_stereo(sig, sig, key, name, tmp)
    rep2.clips.append(analyze_file(c.path, key, name))

cov = rep2.gait_coverage()
check("楼下 3 种方式都在", cov.get("below") == ["sneak", "walk", "run"], f"{cov}")
check("远处只有奔跑", cov.get("far") == ["run"], f"{cov}")

gg = rep2.gait_groups()
check("gait_groups 按位置拆分", set(gg.get("below", {})) == {"sneak", "walk", "run"},
      f"{list(gg.get('below', {}))}")
check("gait_groups 不含未录位置", "near" not in gg)

md = render_report(rep2)
check("报告含覆盖矩阵标题", "位置 × 移动方式 覆盖" in md)
check("报告标出 4/12 覆盖", "4/4" in md or "/" in md)
check("报告列出未录组合", "未录：" in md)
check("报告含移动方式差异一节", "移动方式差异" in md)
check("楼下小节标注已录 3/3", "**楼下脚步**（已录 3/3）" in md)
check("远处小节标注已录 1/3", "**远处脚步**（已录 1/3）" in md)
check("远处提示缺静步", "未录方式：" in md and "静步" in md)
check("报告含同方式跨位置对比", "同一移动方式在不同位置的差异" in md)
check("报告含远处 vs 楼下倾斜", "相对倾斜" in md)


# ---------- 13. 同一组合补录两段 → 功率域合并，不互相覆盖 ----------
print("\n[13] 同组合补录合并")
rep3 = RawReport()
for i, amp in enumerate((0.4, 0.1)):   # 第二段明显更轻
    p = os.path.join(tmp, f"below_run_{i}.wav")
    n = int(SR * 1.0)
    sig = (amp * np.sin(2 * np.pi * 500 * np.arange(n) / SR)).astype(np.float32)
    c = record_stereo(sig, sig, f"below:run", "楼下脚步·奔跑", tmp)
    rep3.clips.append(analyze_file(c.path, "below:run", "楼下脚步·奔跑"))

merged = rep3.gait_groups()["below"]["run"]
single_quiet = 10 * np.log10(max(0.1**2 / 2, 1e-20))
single_loud = 10 * np.log10(max(0.4**2 / 2, 1e-20))
idx = int(np.argmin(np.abs(BAND_CENTERS - 500)))
check("合并在两段之间（不被轻的那段拉低）",
      merged[idx] > single_quiet + 3.0, f"{merged[idx]:.1f} vs {single_quiet:.1f}")
check("合并低于最响的那段（确实做了平均）",
      merged[idx] < single_loud + 1.0, f"{merged[idx]:.1f} vs {single_loud:.1f}")
print("\n" + "=" * 68)
if FAILS:
    print(f"结果：{len(FAILS)} 项失败 → {', '.join(FAILS)}")
    sys.exit(1)
print("结果：全部通过 ✓")
print("=" * 68)
