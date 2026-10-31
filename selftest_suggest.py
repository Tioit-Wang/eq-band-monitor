"""自测：EQ 建议引擎 + 自由拟合 + .fac 读写。"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from app import fac  # noqa: E402
from app.analyzer import BAND_CENTERS, N_BANDS, SpectrumAnalyzer  # noqa: E402
from app.eqsuggest import build_advice, preset_response  # noqa: E402
from app.fit import fit_eq, fit_to_preset, peaking_response  # noqa: E402
from app.recorder import (  # noqa: E402
    FIXED_SCENES,
    GAITS,
    LOCATIONS,
    SCENES,
    Recorder,
    combo_key,
    display_name,
)

SR = 48000
FAILS: list[str] = []

# .fac 测试夹具随仓库分发（tests/fixtures/sample.fac）。
# 早期版本写死了本机的 FxSound 导出路径，CI 上没有那个文件就会失败 ——
# 而现在自测是发布门禁，绝不能依赖某台机器上恰好存在的文件。
USER_FAC = Path(__file__).parent / "tests" / "fixtures" / "sample.fac"


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def make_scene(peaks: dict, amp: float, seconds: float, noise: float, seed: int) -> np.ndarray:
    n = int(SR * seconds)
    t = np.arange(n) / SR
    sig = np.random.default_rng(seed).normal(0, noise, n)
    for f, a in peaks.items():
        sig += a * np.sin(2 * np.pi * f * t)
    return (sig * amp / max(1e-9, np.max(np.abs(sig)))).astype(np.float32)


def record_into(rec: Recorder, key: str, sig: np.ndarray) -> None:
    an = SpectrumAnalyzer(samplerate=SR, release_ms=30)
    rec.start(key)
    t0 = time.perf_counter()
    for i in range(0, len(sig) - 4800, 4800):
        an.process(sig[i : i + 4800], t0 + i / SR)
        rec.feed(an.inst_bands_db, an.loudness_db)
    rec.stop()


print("=" * 68)
print("EQ Band Monitor — 自由拟合 / .fac / 建议引擎 自测")
print("=" * 68)

# ---------- 1. RBJ peaking 数学正确性 ----------
print("\n[1] RBJ peaking 滤波器数学")
r0 = peaking_response(BAND_CENTERS, 1000, 0.0, 1.4)
check("0dB 时响应恒为 0", float(np.max(np.abs(r0))) < 1e-9)
for fc, g in [(1000, 6), (540, -9), (120, 9), (6420, 4)]:
    v = float(peaking_response(np.array([float(fc)]), fc, g, 1.4)[0])
    check(f"fc={fc} gain={g}dB → {v:+.3f}dB", abs(v - g) < 0.05)
a1 = peaking_response(BAND_CENTERS, 1000, 1.0, 1.4)
a2 = peaking_response(BAND_CENTERS, 1000, 2.0, 1.4)
m = np.abs(a1) > 0.1
ratio = float(np.median(np.abs(a2[m] / a1[m])))
check(f"小增益线性度 {ratio:.3f} ≈ 2", 1.95 <= ratio <= 2.05)

# ---------- 2. .fac 字节级往返（导出能否被 FxSound 接受的关键）----------
print("\n[2] .fac 读写字节级往返")
if USER_FAC.exists():
    raw = USER_FAC.read_bytes()
    text = raw.decode("utf-8")
    p = fac.parse(text)
    check("解析出 10 段", len(p.bands) == 10, f"{len(p.bands)}")
    check("预设名正确", p.name == "暗区突围", p.name)
    check(f"Band2 = 120.08Hz +9dB",
          abs(p.bands[1].cf - 120.08) < 0.01 and abs(p.bands[1].gain - 9) < 0.01)
    out = fac.render(p)
    check("重新渲染后字节完全一致", out == text,
          f"{len(raw)} vs {len(out.encode('utf-8'))} bytes")
    check("夹具字节数 976", len(raw) == 976, f"{len(raw)} bytes")
else:
    check("找到 .fac 测试夹具", False, f"缺少 {USER_FAC}")

# ---------- 3. 自由拟合能还原任意频点 ----------
print("\n[3] 自由拟合：频点由数据决定，不套网格")
target = peaking_response(BAND_CENTERS, 137.0, 6.0, 1.2)
bands = fit_eq(target)
ok = any(abs(b.cf - 137) / 137 < 0.15 and b.gain > 4.0 for b in bands)
check(f"137Hz 的峰被放在 {bands[0].cf if bands else 0:.0f}Hz", ok)

target = peaking_response(BAND_CENTERS, 540.0, 5.0, 3.0) + peaking_response(BAND_CENTERS, 642.0, -5.0, 3.0)
bands = fit_eq(target)
model = np.zeros(N_BANDS)
for b in bands:
    model += peaking_response(BAND_CENTERS, b.cf, b.gain, b.q)
rms = float(np.sqrt(np.mean((target - model) ** 2)))
check(f"540/642Hz 紧邻双峰 RMS {rms:.2f}dB < 1.0", rms < 1.0)

# ---------- 4. 拟合输出约束 ----------
print("\n[4] 拟合输出约束")
t = (peaking_response(BAND_CENTERS, 120.0, 9.0, 0.7)
     + peaking_response(BAND_CENTERS, 1000.0, 5.0, 1.0)
     + peaking_response(BAND_CENTERS, 4000.0, -2.0, 1.0)
     + peaking_response(BAND_CENTERS, 8000.0, -4.0, 0.8))
full = fit_to_preset(t, n_bands=10)
check("正好 10 段", len(full) == 10, f"{len(full)}")
check("严格升序", all(full[i].cf < full[i + 1].cf for i in range(len(full) - 1)))
check("全部在 ±12dB 内", all(-12.5 <= b.gain <= 12.5 for b in full))
check("频率在 20Hz–20kHz", all(19 <= b.cf <= 20050 for b in full))
check("全零目标不硬塞段", len(fit_eq(np.zeros(N_BANDS))) == 0)

# ---------- 5. 场景 → 目标曲线 → 建议 ----------
print("\n[5] 端到端：脚步 500Hz / 枪声 4kHz")
rec = Recorder()
record_into(rec, "idle", make_scene({}, 0.02, 2.0, 0.03, 1))
record_into(rec, "near", make_scene({500: 1.0, 800: 0.5, 250: 0.3}, 0.25, 3.0, 0.02, 2))
record_into(rec, "far", make_scene({500: 1.0, 800: 0.5, 250: 0.3}, 0.32, 3.0, 0.02, 3))
record_into(rec, "fire", make_scene({4000: 1.0, 5000: 0.6, 2000: 0.4}, 0.85, 3.0, 0.08, 4))

adv = build_advice(rec)
print("     建议频段：")
for b in adv.bands:
    print(f"       {b.cf:>8.1f} Hz  {b.gain:+6.2f} dB  Q={b.q:.1f}")
print(f"     拟合 RMS {adv.fit_rms:.2f}dB / 最大 {adv.fit_peak:.2f}dB")

b500 = [b for b in adv.bands if 300 < b.cf < 900 and b.gain > 0]
b4k = [b for b in adv.bands if b.cf > 2500 and b.gain < 0]
check("500Hz 附近有提升段", len(b500) > 0)
check("4kHz 附近有衰减段", len(b4k) > 0)
check("最终预设 10 段", len(adv.preset.trimmed()) == 10)

# ---------- 6. 基线预设合成 ----------
print("\n[6] 在用户现有预设上做增量")
if USER_FAC.exists():
    base = fac.load(str(USER_FAC))
    base_resp = preset_response(base)
    adv2 = build_advice(rec, base_preset=base)
    check("保留原预设名", adv2.preset.name == base.name, adv2.preset.name)
    check("保留原预设的频率布置",
          all(abs(adv2.preset.bands[i].cf - base.bands[i].cf) < 1.0 for i in range(10)))
    # 合成后应当更接近目标，而不是更远
    err_before = float(np.sqrt(np.mean((adv2.target_db - base_resp) ** 2)))
    err_after = float(np.sqrt(np.mean((adv2.target_db - adv2.model_db) ** 2)))
    check(f"合成后残差 {err_after:.2f} < 原来 {err_before:.2f}", err_after < err_before)
    # 导出这个预设要能往返
    txt = fac.render(adv2.preset)
    p2 = fac.parse(txt)
    check("合成后的预设可被重新解析", len(p2.bands) == 10)

# ---------- 7. 无数据 / 单场景兜底 ----------
print("\n[7] 兜底")
adv3 = build_advice(Recorder())
check("无数据不崩", len(adv3.preset.trimmed()) == 10)
check("无数据有警告", len(adv3.warnings) > 0)

rec2 = Recorder()
record_into(rec2, "near", make_scene({500: 1.0}, 0.25, 2.0, 0.02, 5))
adv4 = build_advice(rec2)
check("只有脚步时有'缺枪声'警告", any("枪声" in w for w in adv4.warnings))

# ---------- 8. 场景表 + 录制器 ----------
print("\n[8] 场景与录制器")
check("9 个场景（4 位置 + 5 固定）", len(SCENES) == 9, f"{len(SCENES)}")
check("快捷键唯一", len({s.hotkey for s in SCENES}) == 9)
check("位置型 4 个", len(LOCATIONS) == 4, f"{[s.key for s in LOCATIONS]}")
check("固定型 5 个", len(FIXED_SCENES) == 5, f"{[s.key for s in FIXED_SCENES]}")
check("楼下/远处都在位置型里",
      {"below", "far"} <= {s.key for s in LOCATIONS})
check("3 种移动方式", [g for g, _ in GAITS] == ["sneak", "walk", "run"])
check("组合键格式", combo_key("far", "run") == "far:run")
check("组合键显示名", display_name("far:run") == "远处脚步·奔跑", display_name("far:run"))
check("裸键显示名不变", display_name("fire") == "开火", display_name("fire"))
rec5 = Recorder()
rec5.start("near")
for _ in range(2000):
    rec5.feed(np.full(N_BANDS, -40.0), -30.0)
check("帧数上限生效", len(rec5._buf["near"]) == Recorder.MAX_FRAMES)
rec5.stop()

# ---------- 9. 组合键（位置 × 移动方式） ----------
print("\n[9] 组合键：位置 × 移动方式")
rec6 = Recorder()
for key in ("below:sneak", "below:walk", "below:run", "far:sneak", "fire"):
    record_into(rec6, key, make_scene({500: 1.0, 3000: 0.4}, 0.2, 3.0, 0.02, 7))

check("recorded_keys 含组合键",
      rec6.recorded_keys("below") == ["below:run", "below:sneak", "below:walk"],
      f"{rec6.recorded_keys('below')}")
cov = rec6.coverage()
check("coverage 认出楼下三种方式",
      cov.get("below") == ["sneak", "walk", "run"], f"{cov}")
check("coverage 不漏远处", cov.get("far") == ["sneak"], f"{cov}")
check("固定场景不进 coverage", "fire" not in cov)
check("组合键的 profile 显示名不崩",
      rec6.profile("below:run") is not None
      and rec6.profile("below:run").name == "楼下脚步·奔跑",
      rec6.profile("below:run").name)

adv6 = build_advice(rec6)
check("组合键被当作脚步数据（旧版会漏掉）",
      any("楼下" in s or "远处" in s for s in adv6.used_scenes),
      f"{adv6.used_scenes}")
check("多种移动方式时给出等权说明",
      any("移动方式" in f for f in adv6.findings), f"{adv6.findings}")
check("有枪声时不再警告缺枪声",
      not any("缺少枪声" in w for w in adv6.warnings), f"{adv6.warnings}")

# 只录一种方式时应给出补录提示
rec7 = Recorder()
record_into(rec7, "far:run", make_scene({500: 1.0}, 0.2, 3.0, 0.02, 8))
adv7 = build_advice(rec7)
check("只录一种移动方式 → 提示补录",
      any("一种移动方式" in w for w in adv7.warnings), f"{adv7.warnings}")

print("\n" + "=" * 68)
if FAILS:
    print(f"结果：{len(FAILS)} 项失败 → {', '.join(FAILS)}")
    sys.exit(1)
print("结果：全部通过 ✓")
print("=" * 68)
