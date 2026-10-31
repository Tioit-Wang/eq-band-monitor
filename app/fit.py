"""自由频点拟合：从测量出的频谱差异推导最优 EQ 频段，而不是套固定网格。

===== 为什么不能固定网格 =====

FxSound 的 10 个频段中心频率是**任意值**（实测用户预设：
62.5 / 120.08 / 199.54 / 540 / 641.98 / 1010 / 2003 / 6000 / 7493 / 12995）。
注意 540 和 642 只差 0.25 倍频程 —— 说明它允许同区域放多个窄段。

所以「把结果对齐到 25/63/125/…」是错的：
  * 那些频率上你根本不一定有 band；
  * 真正需要调的地方可能是 140Hz，而最近的固定点 125Hz 差了 0.16 倍频程，
    用 Q=1.4 的钟形曲线去凑，形状就是不对。

===== 算法：贪心残差拟合 =====

目标：找一个 10 段 EQ（每段有 cf / gain / Q），使其幅频响应**最接近**目标曲线。

  目标曲线  target(f) = 由场景分析得出的 29/61 段增益需求
  模型      H(f)      = Σ_i peaking(f; cf_i, gain_i, Q_i)
  目标      min ‖H − target‖²  ，且段数 ≤ 10

这是非线性最小二乘（对 cf_i、Q_i 非线性）。用**贪心 + 局部精修**求解：

  1. 残差 r(f) = target(f) − H(f)
  2. 找一个最能降低残差的候选 (cf, Q)：在残差绝对值最大处放一个段，
     扫一遍 Q 值，取投影增益最小二乘解
  3. 加入该段，更新 H
  4. 重复直到用满 10 段，或残差已足够小

好处：
  * 频点由数据决定 —— 差异在哪就放哪，不受制于预设网格
  * 天然支持"两个窄段挨在一起"（如用户预设的 540 + 642）
"""

from __future__ import annotations

import math

import numpy as np

from .analyzer import BAND_CENTERS, N_BANDS, format_hz
from .fac import EqBand

# 频率栅格：用 1/6 倍频程中心频率（与测量分辨率一致），另加密到 1/24 倍频程
_FINE_FREQS = 1000.0 * 2.0 ** (np.arange(-48, 49) / 24.0)   # 250Hz 附近很密
_LOG_FC = np.log2(BAND_CENTERS)


def peaking_response(
    freqs: np.ndarray, fc: float, gain_db: float, q: float
) -> np.ndarray:
    """标准 RBJ peaking EQ 的幅频响应（dB）。
    用正确的 RBJ Audio EQ Cookbook peaking 双二阶：
        A  = 10^(gain_db / 40)
        w0 = 2π·fc / fs
        α  = sin(w0) / (2Q)
        b0 = 1 + α·A,  b1 = −2cos(w0),  b2 = 1 − α·A
        a0 = 1 + α/A,  a1 = −2cos(w0),  a2 = 1 − α/A

    注意 α 只由 Q 决定，绝不要把 A 乘进 α —— 那样增益会改变带宽，
    是常见错误（增益越大峰越窄），与 FxSound 的观感不符。
    """
    if abs(gain_db) < 1e-9:
        return np.zeros_like(freqs, dtype=float)

    fc = float(np.clip(fc, 20.0, 20000.0))
    q = float(np.clip(q, 0.3, 12.0))
    fs = 48000.0  # FxSound 内部处理采样率；对本机 48k 输出是精确的

    a_lin = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * fc / fs
    alpha = math.sin(w0) / (2.0 * q)
    cos_w0 = math.cos(w0)

    b0 = 1.0 + alpha * a_lin
    b1 = -2.0 * cos_w0
    b2 = 1.0 - alpha * a_lin
    a0 = 1.0 + alpha / a_lin
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha / a_lin

    # 在 z = e^{jw} 上求值
    w = 2.0 * np.pi * np.asarray(freqs, dtype=float) / fs
    z1 = np.exp(-1j * w)
    z2 = z1 * z1
    numer = b0 + b1 * z1 + b2 * z2
    denom = a0 + a1 * z1 + a2 * z2
    h = numer / denom
    return 20.0 * np.log10(np.maximum(np.abs(h), 1e-12))


def fit_eq(
    target_db: np.ndarray,
    max_bands: int = 10,
    q_candidates: tuple[float, ...] = (0.5, 0.7, 1.0, 1.4, 2.0),
    max_gain: float = 12.0,
    min_peak_residual_db: float = 1.5,
    min_sep_oct: float = 1.0 / 3.0,
    n_refine_passes: int = 4,
) -> list[EqBand]:
    """贪心 + 坐标下降精修。

    返回按中心频率升序排列的 EqBand 列表（长度 ≤ max_bands）。

    停止条件用**最大残差**而不是 RMS：
    两个紧邻的峰（如 540Hz +5dB / 642Hz -5dB）RMS 可能只有 0.6dB，
    但局部误差有 4dB 以上 —— 只看 RMS 会误判为"已经够好"而直接跳过。
    """
    target = np.asarray(target_db, dtype=float)
    if target.size != N_BANDS:
        raise ValueError(f"target 需要 {N_BANDS} 段，收到 {target.size}")

    freqs = BAND_CENTERS.copy()
    model = np.zeros(N_BANDS)
    bands: list[EqBand] = []

    # ---- 阶段 1：贪心加入 ----
    for _ in range(max_bands):
        resid = target - model
        if float(np.max(np.abs(resid))) < min_peak_residual_db:
            break

        peak_i = int(np.argmax(np.abs(resid)))
        center = BAND_CENTERS[peak_i]

        best = None
        # 候选频点：峰顶 ±1/3 倍频程，步长 1/12
        for off in np.arange(-4, 5):
            fc = float(center * 2.0 ** (off / 12.0))
            if not (20.0 <= fc <= 20000.0):
                continue
            # 与已有段保持最小间距，避免两个几乎同频的段互相拉扯
            if any(abs(math.log2(fc / b.cf)) < min_sep_oct for b in bands):
                continue
            for q in q_candidates:
                shape = peaking_response(freqs, fc, 1.0, q)
                denom = float(shape @ shape)
                if denom < 1e-9:
                    continue
                gain = float(np.clip(shape @ resid / denom, -max_gain, max_gain))
                if abs(gain) < 0.3:
                    continue
                err = float(np.sqrt(np.mean((target - model - shape * gain) ** 2)))
                if best is None or err < best[0]:
                    best = (err, fc, q, gain)

        if best is None:
            break
        _e, fc, q, gain = best
        bands.append(EqBand(cf=fc, gain=gain))
        model = model + peaking_response(freqs, fc, gain, q)
        bands[-1].q = q

    if not bands:
        return []

    # ---- 阶段 2：坐标下降精修（逐个段重新最优）----
    # 贪心只保证每一步局部最优；精修能把"补丁式"的叠加收敛成更贴合的曲线。
    for _ in range(n_refine_passes):
        improved = False
        for i in range(len(bands)):
            others = np.zeros(N_BANDS)
            for j, b in enumerate(bands):
                if j == i:
                    continue
                others = others + peaking_response(
                    freqs, b.cf, b.gain, b.q
                )
            resid = target - others
            cur_err = float(np.sqrt(np.mean((target - others
                                            - peaking_response(freqs, bands[i].cf,
                                                               bands[i].gain,
                                                               bands[i].q)) ** 2)))
            best = (cur_err, bands[i].cf, bands[i].q, bands[i].gain)
            # 在当前 fc 附近细化，并重扫 Q
            for off in np.arange(-2, 3):
                fc = float(bands[i].cf * 2.0 ** (off / 12.0))
                if not (20.0 <= fc <= 20000.0):
                    continue
                for q in q_candidates:
                    shape = peaking_response(freqs, fc, 1.0, q)
                    denom = float(shape @ shape)
                    if denom < 1e-9:
                        continue
                    gain = float(np.clip(shape @ resid / denom, -max_gain, max_gain))
                    err = float(np.sqrt(np.mean((resid - shape * gain) ** 2)))
                    if err < best[0] - 1e-6:
                        best = (err, fc, q, gain)
            if best[0] < cur_err - 1e-6:
                bands[i].cf = best[1]
                bands[i].gain = best[3]
                bands[i].q = best[2]
                improved = True
        if not improved:
            break

    # 丢弃可以忽略的段
    bands = [b for b in bands if abs(b.gain) >= 0.3]

    # 合并过于接近的段。
    # 精修阶段可能出现两个几乎同频的段（如 1000Hz +5.4 和 1000Hz -1.3）：
    # 数学上它们叠加后能凑出目标曲线，但 FxSound 里画两条同频钟形曲线很奇怪，
    # 而且 FxSound 的 10 个 band 在同一位置也表达不出独立的两个峰。
    bands = _merge_close_bands(bands)

    for b in bands:
        b.cf = round(b.cf, 1)
        b.gain = round(b.gain, 2)
    bands.sort(key=lambda b: b.cf)
    return bands


def _merge_close_bands(bands: list[EqBand], min_oct: float = 1.0 / 12.0) -> list[EqBand]:
    """把中心频率相距小于 min_oct 个倍频程的段合并成一个。"""
    if len(bands) <= 1:
        return bands
    ordered = sorted(bands, key=lambda b: b.cf)
    out: list[EqBand] = [EqBand(cf=ordered[0].cf, gain=ordered[0].gain, q=ordered[0].q)]
    for b in ordered[1:]:
        prev = out[-1]
        if math.log2(b.cf / prev.cf) < min_oct:
            # 同一位置合并：增益相加，中心按增益加权（必须先算频率再更新 gain）
            total = prev.gain + b.gain
            if abs(total) > 1e-6:
                prev.cf = (prev.cf * prev.gain + b.cf * b.gain) / total
            else:
                prev.cf = (prev.cf + b.cf) / 2.0
            prev.gain = total
            prev.q = min(prev.q, b.q)
        else:
            out.append(EqBand(cf=b.cf, gain=b.gain, q=b.q))
    return [b for b in out if abs(b.gain) >= 0.3]


def fit_to_preset(
    target_db: np.ndarray,
    n_bands: int = 10,
    max_gain: float = 12.0,
) -> list[EqBand]:
    """拟合，并补齐到 FxSound 要求的固定段数。

    补齐时在没被占用的对数频率区间均匀铺 0dB 段，
    这样 FxSound 里看到的是完整 10 个点，不会破坏预设结构。
    """
    bands = fit_eq(target_db, max_bands=n_bands, max_gain=max_gain)

    # 补齐：在最大空隙里插 0dB
    while len(bands) < n_bands:
        used = sorted(b.cf for b in bands)
        lo, hi = 20.0, 20000.0
        gaps = []
        prev = lo
        for f in used:
            if f > prev * 1.05:
                gaps.append((prev, f))
            prev = f
        if hi > prev * 1.05:
            gaps.append((prev, hi))
        if not gaps:
            break
        # 在最大的对数频率空隙里插一个 0dB 段
        a, b = max(gaps, key=lambda g: math.log(g[1] / g[0]))
        mid = float(np.sqrt(a * b))
        bands.append(EqBand(cf=round(mid, 1), gain=0.0))
        bands.sort(key=lambda x: x.cf)

    return bands


def describe_fit(target_db: np.ndarray, bands: list[EqBand]) -> str:
    """拟合质量描述。"""
    freqs = BAND_CENTERS
    model = np.zeros(N_BANDS)
    for b in bands:
        model = model + peaking_response(freqs, b.cf, b.gain, 1.4)
    err = float(np.sqrt(np.mean((target_db - model) ** 2)))
    peak = float(np.max(np.abs(target_db - model)))
    return f"拟合残差 RMS {err:.2f}dB / 最大 {peak:.2f}dB，共 {len(bands)} 段"



def fit_on_fixed_freqs(
    target_db: np.ndarray,
    freqs_hz: list[float],
    max_gain: float = 12.0,
    q_candidates: tuple[float, ...] = (0.5, 0.7, 1.0, 1.4, 2.0),
) -> list[EqBand]:
    """在**给定的固定频点**上求最优增益（不移动频点）。

    用途：用户已有一个听着舒服的预设，我们只想在其频率布置上补差值。
    这是线性最小二乘问题：
        min_g ‖target − Σ g_i · shape(cf_i, Q_i)‖²
    g=0 永远是一个可行解，所以残差不会比"完全不动"更差。
    """
    target = np.asarray(target_db, dtype=float)
    f = BAND_CENTERS.copy()
    n = len(freqs_hz)
    if n == 0:
        return []

    # 对每个频点选一个最优 Q，然后联合求解增益
    shapes: list[np.ndarray] = []
    for cf in freqs_hz:
        best_shape = None
        best_score = -1.0
        for q in q_candidates:
            s = peaking_response(f, cf, 1.0, q)
            score = float(np.linalg.norm(s))
            if score > best_score:
                best_score = score
                best_shape = s
        shapes.append(best_shape)

    A = np.column_stack(shapes)            # (61, n)
    # 带阻尼的最小二乘，避免段数过多时解发散
    # 阻尼系数：越大越保守。用户已有一个"听着舒服"的预设时，
    # 我们只应做小幅修正，不该借拟合之力把整个预设重写。
    # 实测 lam=0.05 时会把 +9dB 的段改成 -3dB（变化 12dB），完全破坏了原听感。
    lam = 0.5
    AtA = A.T @ A + lam * np.eye(n)
    Atb = A.T @ target
    try:
        g = np.linalg.solve(AtA, Atb)
    except np.linalg.LinAlgError:
        g = np.linalg.lstsq(AtA, Atb, rcond=None)[0]
    g = np.clip(g, -max_gain, max_gain)

    out = []
    for cf, gi in zip(freqs_hz, g):
        out.append(EqBand(cf=float(cf), gain=round(float(gi), 2)))
    return out
