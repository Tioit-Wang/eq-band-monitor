"""交接包生成：把录制的 WAV + 结构化事实打包成 AI 可读的文本。

产物结构：
    <导出目录>/
        audio/           08xx_walk.wav 等原始音频（24-bit）
        report.md        人/AI 都能直接读的完整报告
        report.json      机器可解析的同一份数据

为什么需要 report.md：
  WAV 本身是二进制，任何 AI 都必须先"解码 + 分析"才能用。
  但这些分析步骤是确定性的（见 rawanalyze.py），重复做没有价值，
  而且不同的分析实现可能得出不同数字。
  所以这里把**已算好的事实**写清楚，AI 只需要做判断——
  判断是它擅长的，计算不是。
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime

import numpy as np

from .analyzer import BAND_CENTERS, format_hz
from .rawanalyze import ClipAnalysis, RawReport
from .textfmt import pad


def _fmt_spec_row(values: np.ndarray, step: int = 4) -> list[str]:
    """把 61 段按行排成便于阅读的表格。"""
    out = []
    idx = list(range(0, len(BAND_CENTERS), step))
    for i in idx:
        cell = f"{format_hz(BAND_CENTERS[i]):>7} {values[i]:+7.1f}"
        out.append(cell)
    # 每 4 个一段列，拼成多列
    rows = []
    per_line = 4
    for start in range(0, len(out), per_line):
        rows.append("   ".join(out[start : start + per_line]))
    return rows


def _clip_facts(c: ClipAnalysis) -> str:
    t = c.transients
    lines = [
        f"### {c.name}  `{os.path.basename(c.path)}`",
        "",
        f"- 时长 **{c.seconds:.2f}s** · {c.rate} Hz · {c.channels} 声道",
        f"- 峰值 **{c.peak_db:.1f} dBFS** · RMS **{c.rms_db:.1f} dBFS**",
        f"- 噪声底（10 分位）**{c.noise_floor_db:.1f} dBFS**",
        f"- 峰值所在频段 **{format_hz(c.peak_hz)} Hz**",
    ]
    if t:
        lines += [
            f"- 起音事件 **{t.event_count}** 次（{t.onsets_per_sec:.2f}/秒）",
            f"- 活跃时间占比 **{t.steady_ratio*100:.1f}%**"
            "（越低说明是「脉冲声」，越高说明是「持续声」）",
            f"- 中位起音时间 **{t.attack_ms_median:.0f} ms**",
        ]
    lines.append(f"- 音调性指数 **{c.tonality}**（越大越接近离散音高/乐音；接近 0 是宽带噪声）")
    if c.stereo_corr is not None:
        lines.append(f"- 左右声道相关系数 **{c.stereo_corr}**")
    flags = []
    if c.clipped:
        flags.append("⚠ **削波**（峰值触顶，数据不可用于电平判断）")
    if c.silent:
        flags.append("⚠ **几乎静音**（这段没采到声音，检查输出设备）")
    if flags:
        lines.append("")
        lines += flags
    return "\n".join(lines)


def _coverage_matrix(rep: RawReport) -> str:
    """位置 × 移动方式 的录制覆盖矩阵。

    放在报告最前面，是因为 AI（和用户）最容易犯的错是：
    看到"楼下脚步"有一行数据，就以为它代表楼下的全部情况。
    实际上楼下可能是静步下楼，也可能是奔跑下楼，频谱差别很大。
    这里把"缺哪些组合"显式写出来。
    """
    cov = rep.gait_coverage()
    if not cov:
        return "（没有按移动方式记录的场景）"
    loc_names = {"own": "自己", "near": "旁边", "below": "楼下", "far": "远处"}
    gait_names = {"sneak": "静步", "walk": "正常", "run": "奔跑"}
    order = [k for k in ("own", "below", "near", "far") if k in cov]
    got = sum(len(v) for v in cov.values())
    need = len(order) * 3
    lines = [
        f"**位置 × 移动方式 覆盖：{got}/{need} 个组合已录**",
        "",
        "| 位置 | " + " | ".join(gait_names[g] for g in ("sneak", "walk", "run")) + " |",
        "|---|" + "---|" * 3,
    ]
    for loc in order:
        cells = " | ".join("✔" if g in cov[loc] else "✗" for g in ("sneak", "walk", "run"))
        lines.append(f"| {loc_names.get(loc, loc)} | {cells} |")
    missing = [f"{loc_names.get(loc, loc)}·{gait_names[g]}"
               for loc in order for g in ("sneak", "walk", "run") if g not in cov[loc]]
    if missing:
        lines += [
            "",
            "未录：" + "、".join(missing[:12])
            + ("…" if len(missing) > 12 else ""),
            "",
            "> 同一个位置，静步 / 正常 / 奔跑的频谱分布差别很大："
            "奔跑的低频冲击强，静步几乎只有中高频摩擦声。"
            "只录一种方式不能代表该位置的全部情况。",
        ]
    return "\n".join(lines)


def _clip_detail_list(rep: RawReport) -> str:
    """各场景明细；名称重复时自动加序号，避免同名小节互相覆盖理解。"""
    seen: dict[str, int] = {}
    for c in rep.clips:
        seen[c.name] = seen.get(c.name, 0) + 1
    dup: dict[str, int] = {}
    out: list[str] = []
    for c in rep.clips:
        label = c.name
        if seen[c.name] > 1:
            dup[c.name] = dup.get(c.name, 0) + 1
            label = f"{c.name}（第 {dup[c.name]} 段）"
        out.append(_clip_facts(c, label))
        out.append("")
    return "\n".join(out)


def _clip_facts(c: ClipAnalysis, label: str | None = None) -> str:
    t = c.transients
    lines = [
        f"### {label or c.name}  `{os.path.basename(c.path)}`",
        "",
        f"- 时长 **{c.seconds:.2f}s** · {c.rate} Hz · {c.channels} 声道",
        f"- 峰值 **{c.peak_db:.1f} dBFS** · RMS **{c.rms_db:.1f} dBFS**",
        f"- 噪声底（10 分位）**{c.noise_floor_db:.1f} dBFS**",
        f"- 峰值所在频段 **{format_hz(c.peak_hz)} Hz**",
    ]
    if t:
        lines += [
            f"- 起音事件 **{t.event_count}** 次（{t.onsets_per_sec:.2f}/秒）",
            f"- 活跃时间占比 **{t.steady_ratio*100:.1f}%**"
            "（越低说明是「脉冲声」，越高说明是「持续声」）",
            f"- 中位起音时间 **{t.attack_ms_median:.0f} ms**",
        ]
    lines.append(f"- 音调性指数 **{c.tonality}**（越大越接近离散音高/乐音；接近 0 是宽带噪声）")
    if c.stereo_corr is not None:
        lines.append(f"- 左右声道相关系数 **{c.stereo_corr}**")
    flags = []
    if c.clipped:
        flags.append("⚠ **削波**（峰值触顶，数据不可用于电平判断）")
    if c.silent:
        flags.append("⚠ **几乎静音**（这段没采到声音，检查输出设备）")
    if flags:
        lines.append("")
        lines += flags
    return "\n".join(lines)


def render_report(rep: RawReport, base_preset_info: str | None = None) -> str:
    L: list[str] = []
    L.append("# 游戏音频场景录制报告")
    L.append("")
    L.append(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    L.append("")
    L.append("> 本报告由 EQ Band Monitor 自动生成。音频是系统回环采集的**原始 PCM**")
    L.append("> （未经任何 EQ/压缩处理），分析结果为确定性计算，可复核。")
    L.append("")

    # ---- 数据总览 ----
    L.append("## 1. 录制文件总览")
    L.append("")
    L.append("| 场景 | 文件 | 时长 | 峰值 | RMS | 噪声底 | 标记 |")
    L.append("|---|---|---|---|---|---|---|")
    for c in rep.clips:
        flags = []
        if c.clipped:
            flags.append("削波")
        if c.silent:
            flags.append("静音")
        L.append(
            f"| {c.name} | `{os.path.basename(c.path)}` | {c.seconds:.2f}s | "
            f"{c.peak_db:.1f} | {c.rms_db:.1f} | {c.noise_floor_db:.1f} | "
            f"{' '.join(flags) or '—'} |"
        )
    L.append("")
    L.append("")
    L.append(_coverage_matrix(rep))
    L.append("")
    # ---- 数据质量问题（放前面，避免 AI 基于坏数据下结论）----
    problems = []
    for c in rep.clips:
        if c.clipped:
            problems.append(f"- **{c.name}** 出现削波（峰值 {c.peak_db:.1f} dBFS）。"
                            "该段电平信息不可靠，降低游戏音量后重录。")
        if c.silent:
            problems.append(f"- **{c.name}** 几乎静音（峰值 {c.peak_db:.1f} dBFS）。"
                            "录制时可能没有声音经过该端点。")
        if c.seconds < 1.5:
            problems.append(f"- **{c.name}** 只有 {c.seconds:.2f}s，样本偏少（建议 ≥3s）。")
    if problems:
        L.append("## 2. 数据质量警告")
        L.append("")
        L += problems
        L.append("")
        sec = 3
    else:
        L.append("## 2. 数据质量")
        L.append("")
        L.append("未发现削波、静音或样本不足的问题。")
        L.append("")
        sec = 3

    # ---- 每个文件的明细 ----
    L.append(f"## {sec}. 各场景明细")
    L.append("")
    L.append(_clip_detail_list(rep))
    L.append("")

    # ---- 频谱 ----
    n = sec + 1
    L.append(f"## {n}. 分场景频谱（1/6 倍频程，61 段，dB）")
    L.append("")
    L.append("数值是**时间平均后的带内能量**，可直接与 EQ 增益比较：")
    L.append("EQ 在某段抬 +6dB，该段读数就 +6dB。")
    L.append("")
    # 纯音/极窄带信号会让相邻频段读到 -150dB 以下，这是分频器的正常下溢，
    # 不是"游戏里该频段没声音"。必须写清楚，否则 AI 会拿它当"该段可任意砍"。
    L.append("> ⚠ 若某段读数低于 **-100 dBFS**，说明这段只有分频器的数值下溢、")
    L.append("> 没有实际内容（多见于合成测试信号）。**不要据此判断该频段可以砍**，")
    L.append("> 真实游戏录音里几乎不会出现这种情况。")
    L.append("")
    for c in rep.clips:
        if c.spectrum_db is None:
            continue
        L.append(f"**{c.name}**")
        L.append("")
        L.append("```")
        for row in _fmt_spec_row(c.spectrum_db):
            L.append(row)
        L.append("```")
        L.append("")

    # ---- 场景间差异 ----
    n += 1
    L.append(f"## {n}. 场景间形状差异（已去除整体音量）")
    L.append("")
    keep = rep.keep_spectrum()
    red = rep.reduce_spectrum()
    idle = rep.idle_spectrum()

    if keep is not None and red is not None:
        sep = (keep - np.mean(keep)) - (red - np.mean(red))
        L.append("`want − reduce`，正值 = 该频段「想听清的声音」相对更多 → 建议提升；")
        L.append("负值 = 该频段「想压掉的声音」相对更多 → 建议衰减。")
        L.append("")
        L.append("```")
        for row in _fmt_spec_row(sep):
            L.append(row)
        L.append("```")
        L.append("")
        # 自动指出最显著的几处
        order = np.argsort(sep)[::-1]
        L.append("最值得提升的频段：" + "、".join(
            f"{format_hz(BAND_CENTERS[i])}({sep[i]:+.0f})" for i in order[:5]))
        L.append("")
        L.append("最值得衰减的频段：" + "、".join(
            f"{format_hz(BAND_CENTERS[i])}({sep[i]:+.0f})" for i in order[-5:][::-1]))
        L.append("")
    else:
        L.append("（需要至少一个敌人脚步场景和一个枪声场景才能算差异）")
        L.append("")

    # ---- 距离衰减分析（本版新增，核心）----
    groups = rep.distance_groups()
    if len(groups) >= 2:
        n += 1
        L.append(f"## {n}. 距离衰减分析")
        L.append("")
        L.append("不同距离的敌人脚步，频谱差异主要来自**空气与材质的距离衰减**：")
        L.append("距离越远，高频衰减越多。这决定了「抬高频」对远距离敌人是否有效。")
        L.append("")
        names = {"near": "旁边", "below": "楼下", "far": "远处"}
        # 以最近的为基准，算每个距离的高频损失
        ref_key = "near" if "near" in groups else next(iter(groups))
        ref = groups[ref_key]
        L.append(f"基准：**{names.get(ref_key, ref_key)}**")
        L.append("")
        L.append("| 频段 | " + " | ".join(names.get(k, k) for k in groups) + " | 相对基准差 |")
        L.append("|---|" + "---|" * len(groups) + "---|")
        probe = [i for i, f in enumerate(BAND_CENTERS) if 500 <= f <= 16000][::6]
        for i in probe:
            cells = " | ".join(f"{groups[k][i]:.1f}" for k in groups)
            diffs = [f"{groups[k][i] - ref[i]:+.1f}" for k in groups if k != ref_key]
            L.append(
                f"| {format_hz(BAND_CENTERS[i])}Hz | {cells} | {' / '.join(diffs) or '—'} |"
            )
        L.append("")
        # 高频倾斜：用两组的**去均值形状差**，而不是绝对电平差。
        # 绝对差会把"远处本来就更轻"混进来 —— 测出的是"整体衰减 + 频谱倾斜"之和，
        # 而 EQ 决策需要的只是后者。去均值（减掉各自的全频段均值）正好抵消整体衰减。
        LOWBAND = [i for i, f in enumerate(BAND_CENTERS) if 200 <= f <= 800]
        HF = [i for i, f in enumerate(BAND_CENTERS) if f >= 4000]
        for k in groups:
            if k == ref_key:
                continue
            ak, ar = groups[k], ref
            # 只在"两者都不是哑段"的位置比较：用全谱峰值-40dB 作为很宽松的门限
            def active(arr):
                return arr > (float(np.max(arr)) - 40.0)
            hf = [i for i in HF if active(ak)[i] and active(ar)[i]]
            if len(hf) >= 2:
                # 各自去均值后再相减，得到纯粹的"形状差"
                shape_delta = (ak - np.mean(ak)) - (ar - np.mean(ar))
                tilt_delta = float(np.mean(shape_delta[hf]))
                L.append(
                    f"- **{names.get(k, k)}** 相对{names.get(ref_key, ref_key)}，"
                    f"4kHz 以上的**相对倾斜** **{tilt_delta:+.1f} dB**"
                    f"（{len(hf)} 段有效；负值 = 高频相对更少）"
                )
            else:
                L.append(f"- **{names.get(k, k)}**：4kHz 以上有效频段不足，无法比较")
        L.append("")
        L.append("> 「距离越远高频越少」是物理必然。若数据证实这一点，说明")
        L.append("> 抬高频只帮近距离敌人；对远处敌人应当依赖 200–800Hz 的中低频。")
        L.append("")

    # ---- 自己脚步 vs 敌人脚步（可分性）----
    own = rep.own_step_spectrum()
    enemy = rep.enemy_step_spectrum()
    if own is not None and enemy is not None:
        n += 1
        L.append(f"## {n}. 自己脚步 vs 敌人脚步：能否分开")
        L.append("")
        overlap = enemy - own
        best = np.argsort(overlap)[::-1]
        sep_span = float(np.max(overlap) - np.min(overlap))
        L.append(f"两者频谱的最大差异 **{sep_span:.1f} dB**。")
        L.append("")
        L.append("```")
        for row in _fmt_spec_row(overlap):
            L.append(row)
        L.append("```")
        L.append("")
        L.append("敌人脚步相对更明显的频段："
                 + "、".join(f"{format_hz(BAND_CENTERS[i])}({overlap[i]:+.1f})" for i in best[:4]))
        L.append("")
        if sep_span < 6.0:
            L.append("⚠ **差异不足 6dB：这两种声音在频谱上几乎完全重叠。**")
            L.append("EQ 无法把它们分开 —— 提升任何脚步频段，自己脚步也会同样被放大。")
            L.append("这种情况下唯一有效的方向是降低自己脚步的响度（游戏内设置），"
                     "或调整听音习惯（静止听声）。")
        else:
            L.append("差异足够，可以在两者差距最大的频段做定向提升。")
        L.append("")

    if keep is not None and idle is not None:
        snr = keep - idle
        L.append("**信噪比**（敌人脚步相对静止底噪的富余）：")
        L.append("")
        L.append(f"- 最大 {float(np.max(snr)):.1f} dB")
        L.append(f"- 中位 {float(np.median(snr)):.1f} dB")
        if float(np.max(snr)) < 6.0:
            L.append("- ⚠ 最大富余不足 6dB：此时任何 EQ 提升都会连底噪一起放大，"
                     "应优先降低游戏总音量。")
        L.append("")

    # ---- 移动方式差异（静步 / 正常 / 奔跑），按位置逐个展开 ----
    gg = rep.gait_groups()
    if gg:
        n += 1
        L.append(f"## {n}. 移动方式差异（静步 / 正常 / 奔跑）")
        L.append("")
        L.append("同一位置不同移动方式的频谱差异。移动越快，脚步冲击越强、低频越多；")
        L.append("静步几乎没有低频冲击，能量集中在鞋底摩擦的中高频。")
        L.append("")
        L.append("> 这一节是**每个位置各自**的三种方式对比。楼下和远处的静步是最难听见、")
        L.append("> 也是最有价值的信息声 —— 如果它们的中高频在 EQ 里被压掉了，")
        L.append("> 无论抬多少低频都补不回来。")
        L.append("")
        gait_names = {"sneak": "静步", "walk": "正常", "run": "奔跑"}
        loc_names = {"own": "自己", "near": "旁边", "below": "楼下", "far": "远处"}
        GAIT_ORDER = ("sneak", "walk", "run")
        lo_idx = [i for i, f in enumerate(BAND_CENTERS) if f <= 300]
        hi_idx = [i for i, f in enumerate(BAND_CENTERS) if 1000 <= f <= 6000]

        for loc in ("below", "far", "near", "own"):
            if loc not in gg:
                continue
            sub = gg[loc]
            present = [g for g in GAIT_ORDER if g in sub]
            missing = [g for g in GAIT_ORDER if g not in sub]
            L.append(f"**{loc_names.get(loc, loc)}脚步**"
                     f"（已录 {len(present)}/3）")
            L.append("")
            if missing:
                L.append("未录方式：" + "、".join(gait_names[g] for g in missing)
                         + " —— 该位置在这一方式下的表现没有数据支撑。")
                L.append("")
            if len(present) < 2:
                L.append("（只有一种方式，无法做方式间对比）")
                L.append("")
                continue

            L.append("| 频段 | " + " | ".join(gait_names[g] for g in present) + " |")
            L.append("|---|" + "---|" * len(present))
            probe = [i for i, f in enumerate(BAND_CENTERS) if 50 <= f <= 8000][::5]
            for i in probe:
                cells = " | ".join(f"{sub[g][i]:.1f}" for g in present)
                L.append(f"| {format_hz(BAND_CENTERS[i])}Hz | {cells} |")
            L.append("")

            # 量化：低速 vs 高速 的低频/中高频占比变化
            if "sneak" in sub and "run" in sub:
                d_lo = float(np.mean(sub["run"][lo_idx]) - np.mean(sub["sneak"][lo_idx]))
                d_hi = float(np.mean(sub["run"][hi_idx]) - np.mean(sub["sneak"][hi_idx]))
                L.append(f"- 奔跑相比静步：低频(<300Hz) **{d_lo:+.1f} dB**，"
                         f"中高频(1–6kHz) **{d_hi:+.1f} dB**")
                if d_lo - d_hi > 4:
                    L.append(f"- 低频多出 {d_lo-d_hi:.1f} dB：说明奔跑的脚步"
                             "主要靠**低频冲击**被察觉，静步则依赖中高频摩擦声。")
                    L.append("  若要同时听清两者，低频和中高频都得兼顾，"
                             "只抬低频会让静步更难听见。")
                L.append("")

        # 不同位置的同一种移动方式对比：同样"静步"，楼下和远处的差别
        L.append("**同一移动方式在不同位置的差异**")
        L.append("")
        L.append("同样是静步，楼下（隔着地板）和远处（隔着空气距离）的频谱不同。")
        L.append("这决定了「抬中高频」对哪种场景更有效。")
        L.append("")
        cross_locs = [k for k in ("near", "below", "far") if k in gg]
        for g in GAIT_ORDER:
            have = [k for k in cross_locs if g in gg[k]]
            if len(have) < 2:
                continue
            L.append(f"- {gait_names[g]}：")
            for k in have:
                arr = gg[k][g]
                hi_mean = float(np.mean(arr[hi_idx]))
                lo_mean = float(np.mean(arr[lo_idx]))
                L.append(f"  - {loc_names.get(k, k)}：低频 {lo_mean:.1f} dBFS，"
                         f"中高频 {hi_mean:.1f} dBFS，"
                         f"中高频相对低频 **{hi_mean - lo_mean:+.1f} dB**")
            if "below" in have and "far" in have:
                b = gg["below"][g] - np.mean(gg["below"][g])
                f_ = gg["far"][g] - np.mean(gg["far"][g])
                tilt = float(np.mean((f_ - b)[hi_idx]))
                L.append(f"  - 远处相对楼下的中高频**相对倾斜** **{tilt:+.1f} dB**"
                         f"（{'远处中高频更少，抬高频对楼下更划算' if tilt < -2 else '两者接近，同一套增益可以兼顾'}）")
            L.append("")

        L.append("> 结论用法：若楼下的静步在中高频明显强于远处，说明楼下的静步更依赖")
        L.append("> 中高频摩擦声被察觉，抬 1–6kHz 对「听楼下静步」收益最大；")
        L.append("> 远处无论哪种方式都偏低频，只能靠 200–800Hz 的中低频。")
        L.append("")

    # ---- 基线预设 ----
    if base_preset_info:
        n += 1
        L.append(f"## {n}. 当前基线预设")
        L.append("")
        L.append(base_preset_info)
        L.append("")
        L.append("> 若在此预设上调整，应只做增量修改，不要覆盖原有听感。")
        L.append("")

    # ---- 给 AI 的说明 ----
    n += 1
    L.append(f"## {n}. 分析约定")
    L.append("")
    L.append("- 所有 dB 都是 dBFS（0 = 满刻度）。")
    L.append("- 频谱值是 1/6 倍频程带内总能量，与 EQ 增益 1:1 对应。")
    L.append("- 「想听清」= 敌人脚步（旁边/楼下/远处，含静步/正常/奔跑）+ 跳跃；")
    L.append("  「想压掉」= 开火/手榴弹/震撼弹；「自己脚步」是**干扰源**不是目标；")
    L.append("  「静止底噪」仅作参考基准。")
    L.append("- 位置型场景的 key 是 `位置:移动方式`（如 `far:run`），")
    L.append("  `report.json` 里的 `scene` 字段是显示名，文件名里冒号已换成 `-`。")
    L.append("- 可调 EQ 为 10 段参数均衡，中心频率**可任意设定**")
    L.append("  （不是固定的 25/63/125 网格），单段增益范围 ±12dB。")
    L.append("- 原始音频在 `audio/` 目录，24-bit PCM，未做任何处理。")
    L.append("")
    if rep.notes:
        L.append("### 分析过程中的问题")
        L.append("")
        L += [f"- {x}" for x in rep.notes]
        L.append("")

    return "\n".join(L)


def render_json(rep: RawReport) -> dict:
    """同一份数据的机器可读版本。"""
    def clip_obj(c: ClipAnalysis) -> dict:
        return {
            "scene": c.name,
            "key": c.key,
            "location": c.key.split(":", 1)[0],
            "gait": c.key.split(":", 1)[1] if ":" in c.key else None,
            "file": os.path.basename(c.path),
            "seconds": round(c.seconds, 3),
            "rate": c.rate,
            "channels": c.channels,
            "peak_dbfs": round(c.peak_db, 2),
            "rms_dbfs": round(c.rms_db, 2),
            "noise_floor_dbfs": round(c.noise_floor_db, 2),
            "peak_band_hz": round(c.peak_hz, 1),
            "clipped": c.clipped,
            "silent": c.silent,
            "tonality": c.tonality,
            "stereo_corr": c.stereo_corr,
            "transients": None if c.transients is None else {
                "events": c.transients.event_count,
                "onsets_per_sec": c.transients.onsets_per_sec,
                "steady_ratio": c.transients.steady_ratio,
                "attack_ms_median": c.transients.attack_ms_median,
            },
            "spectrum_db": None if c.spectrum_db is None
            else [round(float(v), 2) for v in c.spectrum_db],
        }

    return {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "band_centers_hz": [round(float(v), 2) for v in BAND_CENTERS],
        "band_resolution": "1/6 octave, 61 bands, 19.7Hz-20158.7Hz",
        "clips": [clip_obj(c) for c in rep.clips],
        "gait_coverage": rep.gait_coverage(),
        "notes": rep.notes,
    }


def save_bundle(
    out_dir: str,
    rep: RawReport,
    copies: dict[str, str],
    base_preset_info: str | None = None,
) -> str:
    """把音频 + 报告写进一个目录。返回目录路径。

    copies: {key: 源 wav 路径}，会被复制到 out_dir/audio/
    """
    os.makedirs(out_dir, exist_ok=True)
    audio_dir = os.path.join(out_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    for src in copies.values():
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(audio_dir, os.path.basename(src)))

    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write(render_report(rep, base_preset_info))

    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(render_json(rep), f, ensure_ascii=False, indent=2)

    return out_dir
