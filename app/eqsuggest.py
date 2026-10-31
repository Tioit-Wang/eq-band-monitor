"""EQ 建议引擎：把场景录制统计 → 目标曲线 → 自由频点拟合 → 可导入的 FxSound 预设。

===== 推导链 =====

[1] 每个场景的能量形状
      录制时累积瞬时 61 段频谱，功率域平均 → mean_db（与音量无关的形状）

[2] 目标曲线 target(f)
      先去电平（减均值），再算"想留 − 想砍"的可分离度：
          sep = tilt(keep) − tilt(reduce)
      然后用信噪比门控：某频段脚步声只比底噪高 2dB 时，抬它只会放大底噪。

[3] 自由拟合
      交给 fit.py：贪心 + 坐标下降，求 cf / gain / Q 都由数据决定，
      不套任何固定频率网格（FxSound 的 10 个 band 中心频率是任意值）。

[4] 与现有预设合成
      如果用户已经有一个"听着很舒服"的预设，我们的建议应当作为**增量**叠上去，
      而不是把它整个覆盖掉 —— 那个预设承载了用户的听感偏好，
      我们只负责"补上游戏里缺的、压掉多余的"。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .analyzer import BAND_CENTERS, N_BANDS, format_hz
from .fac import EqBand, FacPreset
from .fit import fit_on_fixed_freqs, fit_to_preset, peaking_response

from .recorder import (
    ENEMY_STEP_LOCATIONS,
    GAIT_BY_KEY,
    GUNSHOT_KEYS,
    SCENE_BY_KEY,
    Recorder,
    SceneProfile,
    display_name,
    parse_key,
)

# ---- 输出约束 ----
LOW_GUARD_HZ = 50.0    # 以下不深砍：人耳听阈高、游戏几乎无内容
LOW_GUARD_CUT = 6.0
MAX_TOTAL_BOOST = 30.0  # 10 段正向增益总和上限
MIN_MEANINGFUL_DB = 1.0  # EQ 上 ±1dB 以下听不出差别，直接归零

# 已有基线预设时，单个频段允许的最大改动量。
# 用户那个预设承载主观听感，我们只做保守修正，不重写它。
MAX_DELTA_DB = 4.0


@dataclass
class Advice:
    """完整的 EQ 建议结果。"""

    bands: list[EqBand]              # 自由拟合出的频段（cf/gain/Q 都由数据定）
    preset: FacPreset                # 可直接导出的完整预设
    target_db: np.ndarray            # 61 段目标曲线
    band_sep: np.ndarray             # 61 段可分离度
    band_snr: np.ndarray             # 61 段信噪比
    model_db: np.ndarray             # 拟合模型的幅频响应
    summary: str
    findings: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    used_scenes: list[str] = field(default_factory=list)
    fit_rms: float = 0.0
    fit_peak: float = 0.0

    def table_lines(self) -> list[str]:
        """对齐的文本表（中英混排按显示宽度补齐）。"""
        from .textfmt import pad

        lines = [
            pad("频点", 10, "right") + "  " + pad("增益", 8, "right") + "  "
            + pad("Q值", 6, "right") + "   说明",
            "-" * 72,
        ]
        for b in self.bands:
            g = f"{b.gain:+.1f}" if abs(b.gain) >= 0.05 else "0.0"
            if abs(b.gain) < 0.05:
                note = "保持不动"
            elif b.gain > 0:
                note = "提升（该频段脚步声占优）"
            else:
                note = "衰减（该频段枪声占优）"
            lines.append(
                pad(f"{b.cf:.1f}Hz" if b.cf < 1000 else format_hz(b.cf) + "Hz", 10, "right")
                + "  " + pad(g, 8, "right") + "  " + pad(f"{b.q:.2f}", 6, "right")
                + "   " + note
            )
        return lines


def tilt(x: np.ndarray) -> np.ndarray:
    return x - float(np.mean(x))


def _avg(profiles: list[SceneProfile]) -> np.ndarray | None:
    if not profiles:
        return None
    p = np.mean([10.0 ** (pr.mean_db / 10.0) for pr in profiles], axis=0)
    return 10.0 * np.log10(np.maximum(p, 1e-20))


def _avg_gait_equal(profiles: list[SceneProfile]) -> np.ndarray | None:
    """先按移动方式分组各自平均，再在方式之间等权平均。

    为什么要等权：静步/正常/奔跑是三种**并列**的听觉场景，
    玩家关心的权重是"能不能都听见"，而不是"每种录了几秒"。
    直接对所有帧做功率平均会让录得长的那个方式主导结果 ——
    实测中奔跑录 3 段、静步只录 1 段时，静步的特征会被完全淹掉，
    于是建议只针对奔跑优化，静步更难听见。
    """
    if not profiles:
        return None
    buckets: dict[str | None, list[SceneProfile]] = {}
    for pr in profiles:
        _, gait = parse_key(pr.key)
        buckets.setdefault(gait, []).append(pr)
    per_gait = [_avg(v) for v in buckets.values()]
    per_gait = [a for a in per_gait if a is not None]
    if not per_gait:
        return None
    p = np.mean([10.0 ** (a / 10.0) for a in per_gait], axis=0)
    return 10.0 * np.log10(np.maximum(p, 1e-20))


def _collect(recorder: Recorder, locations: tuple[str, ...]) -> list[SceneProfile]:
    """收集一批位置下的全部已录画像（组合键会被正确展开）。"""
    keys: list[str] = []
    for loc in locations:
        keys += recorder.location_keys(loc)
    out: list[SceneProfile] = []
    for k in keys:
        pr = recorder.profile(k)
        if pr is not None:
            out.append(pr)
    return out


def preset_response(preset: FacPreset) -> np.ndarray:
    """现有预设的幅频响应（61 段）。Q 未知，用保守的 1.4。"""
    out = np.zeros(N_BANDS)
    for b in preset.trimmed():
        out = out + peaking_response(BAND_CENTERS, b.cf, b.gain, getattr(b, "q", 1.4))
    return out


def build_advice(
    recorder: Recorder,
    base_preset: FacPreset | None = None,
    max_gain: float = 12.0,
    target_snr_db: float = 12.0,
    strength: float = 1.0,
    max_boost_db: float = 12.0,
) -> Advice:
    """根据已录制的场景计算自由频点 EQ 建议。"""
    # 场景 key 可能是组合键（"near:run"），因此必须按**位置前缀**扫已录 key，
    # 不能写死裸键名：只录了 near:run 时 has_data("near") 是 False，
    # 会导致整套推导静默退化成"没有脚步声数据"。
    keep_keys = (
        [k for loc in ENEMY_STEP_LOCATIONS for k in recorder.location_keys(loc)]
        + recorder.recorded_keys("jump")
    )
    reduce_keys = [k for k in GUNSHOT_KEYS if recorder.has_data(k)]
    has_idle = recorder.has_data("idle")

    keep_profiles = [p for p in (recorder.profile(k) for k in keep_keys) if p]
    reduce_profiles = [p for p in (recorder.profile(k) for k in reduce_keys) if p]
    idle_profile = recorder.profile("idle") if has_idle else None

    findings: list[str] = []
    warnings: list[str] = []
    sep = np.zeros(N_BANDS)
    snr = np.full(N_BANDS, 99.0)

    empty_preset = FacPreset(name="EQ Band Monitor")
    empty_preset.bands = [EqBand(1000.0, 0.0) for _ in range(10)]

    if not keep_profiles and not reduce_profiles:
        return Advice(
            bands=[],
            preset=empty_preset,
            target_db=np.zeros(N_BANDS),
            band_sep=sep,
            band_snr=snr,
            model_db=np.zeros(N_BANDS),
            summary="还没有任何录制数据。先按快捷键录几个场景再来看建议。",
            warnings=["未检测到任何有效录制"],
        )

    # 步态等权：静步/正常/奔跑是三种并列场景，不能让录得久的那个独大。
    # 否则「只抬奔跑的低频」这类建议会让静步更难听见。
    keep_band = _avg_gait_equal(keep_profiles)
    reduce_band = _avg_gait_equal(reduce_profiles)

    # 顺带说明：录了几种移动方式，避免用户以为一种代表全部
    _gaits_seen = sorted({g for g in (parse_key(k)[1] for k in keep_keys) if g})
    if len(_gaits_seen) >= 2:
        findings.append(
            f"敌人脚步覆盖 {len(_gaits_seen)} 种移动方式"
            f"（{'/'.join(GAIT_BY_KEY.get(g, g) for g in _gaits_seen)}），"
            "已在方式之间等权平均，不会让某一种独大。"
        )
    elif len(_gaits_seen) == 1:
        warnings.append(
            f"敌人脚步只录了「{GAIT_BY_KEY.get(_gaits_seen[0], _gaits_seen[0])}」一种移动方式。"
            "静步/正常/奔跑的低频与中高频分布不同，建议把三种都录一遍"
            "（Ctrl+Shift+1/2/3 切换后按同一位置键）。"
        )

    # ---- 可分离度 ----
    if keep_band is not None and reduce_band is not None:
        sep = tilt(keep_band) - tilt(reduce_band)
        findings.append("两条曲线已归一化到同一电平，比较的是频谱形状而非音量。")
    elif keep_band is not None:
        sep = tilt(keep_band) * 0.5
        warnings.append("缺少枪声类场景，无法定位需要衰减的频段")
    else:
        sep = -tilt(reduce_band) * 0.5
        warnings.append("缺少脚步声类场景，无法定位需要增强的频段")

    # ---- 信噪比门控（抬/砍分别算）----
    w_up = np.ones(N_BANDS)
    w_dn = np.ones(N_BANDS)
    if idle_profile is not None:
        idle = idle_profile.mean_db
        if keep_band is not None:
            snr = keep_band - idle
            w_up = np.clip(snr / target_snr_db, 0.0, 1.0)
            findings.append(f"脚步声相对底噪的最大富余 {float(np.max(snr)):.1f} dB。")
            if float(np.max(snr)) < 6.0:
                warnings.append(
                    "脚步声相对底噪的余量不足 6dB，任何提升都会连底噪一起放大。"
                )
        if reduce_band is not None:
            w_dn = np.clip((reduce_band - idle) / target_snr_db, 0.0, 1.0)
    else:
        warnings.append("未录制'静止底噪'，无法做信噪比门控（建议补录）")

    # ---- 归一化到 EQ 可用范围 ----
    scale = float(np.percentile(np.abs(sep), 90))
    if scale < 1e-6:
        scale = 1.0
    norm = np.clip(sep / scale, -1.0, 1.0)
    w = np.where(norm > 0, w_up, w_dn)
    target = norm * np.where(norm > 0, max_boost_db, 12.0) * strength * w

    # 低频保护
    low = BAND_CENTERS < LOW_GUARD_HZ
    if low.any() and (target[low] < -LOW_GUARD_CUT).any():
        target = np.where(low, np.maximum(target, -LOW_GUARD_CUT), target)
        findings.append(f"{LOW_GUARD_HZ:.0f}Hz 以下的衰减限制在 {LOW_GUARD_CUT:.0f}dB 内。")

    # ---- 自由拟合 ----
    base_resp = preset_response(base_preset) if base_preset is not None else np.zeros(N_BANDS)
    # 若给了现有预设，把它当作已生效的基线：我们的建议只补差值
    target_resid = target - base_resp if base_preset is not None else target

    fitted = fit_to_preset(target_resid, n_bands=10, max_gain=max_gain)

    # 低于阈值归零
    for b in fitted:
        if abs(b.gain) < MIN_MEANINGFUL_DB:
            b.gain = 0.0

    # 总量限制
    pos_sum = sum(b.gain for b in fitted if b.gain > 0)
    if pos_sum > MAX_TOTAL_BOOST:
        k = MAX_TOTAL_BOOST / pos_sum
        for b in fitted:
            if b.gain > 0:
                b.gain = round(b.gain * k, 2)
        findings.append(f"提升总量超过 {MAX_TOTAL_BOOST:.0f}dB，已按比例缩放。")

    # ---- 合成最终预设 ----
    if base_preset is not None:
        # 在用户现有预设的频点上重新求增益（不移动频点）。
        # 关键：增量必须限幅。用户那个预设是"听着舒服"的主观调音，
        # 我们的职责只是补游戏里的缺口，不是借拟合之力重写它。
        # 实测不限幅时会把 +9dB 的段改成 -3dB（变化 12dB），听感彻底变了。
        base_freqs = [b.cf for b in base_preset.trimmed()]
        blended = fit_on_fixed_freqs(target - base_resp, base_freqs, max_gain=max_gain)
        capped: list[EqBand] = []
        clamped = 0
        for b, d in zip(base_preset.trimmed(), blended):
            adj = float(np.clip(d.gain, -MAX_DELTA_DB, MAX_DELTA_DB))
            if abs(adj - d.gain) > 1e-6:
                clamped += 1
            capped.append(EqBand(cf=b.cf, gain=round(b.gain + adj, 2), q=b.q))
        final = FacPreset(
            name=base_preset.name,
            version=base_preset.version,
            raw_lines=list(base_preset.raw_lines),
        )
        final.bands = capped
        findings.append(
            f"已在现有预设上做增量调整，单段最大改动限制在 ±{MAX_DELTA_DB:.0f}dB。"
        )
        if clamped:
            findings.append(f"有 {clamped} 个段触及限幅，说明该处缺口超过保守调整范围。")
    else:
        final = FacPreset(name="EQ Band Monitor")
        final.bands = list(fitted)

    model = np.zeros(N_BANDS)
    for b in final.trimmed():
        model = model + peaking_response(BAND_CENTERS, b.cf, b.gain, b.q)

    resid = target - model
    fit_rms = float(np.sqrt(np.mean(resid**2)))
    fit_peak = float(np.max(np.abs(resid)))

    # ---- 摘要 ----
    boosted = [b for b in fitted if b.gain >= MIN_MEANINGFUL_DB]
    cutted = [b for b in fitted if b.gain <= -MIN_MEANINGFUL_DB]
    parts = []
    if boosted:
        parts.append("增强 " + "、".join(f"{format_hz(b.cf)}({b.gain:+.0f})" for b in boosted))
    if cutted:
        parts.append("衰减 " + "、".join(f"{format_hz(b.cf)}({b.gain:+.0f})" for b in cutted))
    summary = "；".join(parts) if parts else "各频段差异不明显，建议保持当前 EQ。"

    used = keep_keys + reduce_keys + (["idle"] if has_idle else [])
    total_frames = sum(
        (recorder.profile(k).frames for k in used if recorder.profile(k)), start=0
    )
    if total_frames < 60:
        warnings.append(
            f"样本量偏少（{total_frames} 帧 ≈ {total_frames/20:.1f} 秒），建议每场景录 3 秒以上。"
        )

    return Advice(
        bands=fitted,
        preset=final,
        target_db=target,
        band_sep=sep,
        band_snr=snr,
        model_db=model,
        summary=summary,
        findings=findings,
        warnings=warnings,
        used_scenes=[display_name(k) for k in used],
        fit_rms=fit_rms,
        fit_peak=fit_peak,
    )


def _combine(base: FacPreset, delta: list[EqBand]) -> FacPreset:
    """把增量建议叠加到现有预设上。

    做法：在 base 的每个 band 频率上算 delta 的总响应，加到该 band 的 gain 上。
    这样保留 base 的频率布置（用户已经调好的位置），只改变增益量。
    """
    out = FacPreset(name=base.name, version=base.version, raw_lines=list(base.raw_lines))
    blended: list[EqBand] = []
    for b in base.trimmed():
        add = 0.0
        for d in delta:
            add += float(
                peaking_response(np.array([b.cf]), d.cf, d.gain, d.q)[0]
            )
        blended.append(EqBand(cf=b.cf, gain=round(b.gain + add, 2), q=getattr(b, "q", 1.4)))
    out.bands = blended
    return out
