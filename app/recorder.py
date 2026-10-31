"""场景录制：按快捷键开始/结束录制某个声学场景，累积该场景的频谱统计。

设计要点：
  * 每个场景独立累积"能量平均后的频谱形状"和"响度"。
  * 统计量用**功率域平均**（不是 dB 平均）——dB 平均会让瞬态声被低估。
  * 同时记录 P95，因为脚步声、枪声都是瞬态，均值会被人声/环境摊平。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .analyzer import BAND_CENTERS, N_BANDS, format_hz


@dataclass
class Scene:
    """一个声学场景的定义。"""

    key: str            # 内部标识
    name: str           # 显示名
    hotkey: str         # 推荐快捷键
    role: str           # 音效听觉角色：'keep' / 'reduce' / 'ref'
    desc: str           # 说明


# 场景是**两个维度**的组合：位置 × 移动方式。
#   位置：自己 / 楼下 / 旁边 / 远处
#   方式：静步 / 正常 / 奔跑
# 组合起来 4×3 = 12 种。如果每种都配一个热键，F 键不够用，
# 而且玩家在游戏里也记不住。所以用"模式选择器"：
#   先 Ctrl+Shift+1/2/3 选当前方式，再按位置键录制。
# 这样热键数量恒定，录制时按键变成 key = "位置:方式"。
#
# role 决定它在优化中的角色：
#   keep   = 想听清的信息声（敌人脚步），要增强
#   mask   = 会盖住信息声的自身声（自己脚步），要避开它的频段
#   reduce = 想压掉的干扰声（枪声/爆炸）
#   ref    = 参考底噪
#
# 关于"自己脚步"为什么是 mask 而不是 keep：
# 它和敌人脚步频段高度重叠，EQ 无法区分"这是谁的脚步"。
# 把它标成 keep 会掩盖这个事实，导致软件建议"抬脚步频段"，
# 结果自己的脚步也被同步放大，反而更吵。

GAITS: list[tuple[str, str]] = [
    ("sneak", "静步"),
    ("walk",  "正常走"),
    ("run",   "奔跑"),
]
GAIT_BY_KEY = {k: n for k, n in GAITS}

# 位置型场景：需要和当前移动方式组合
LOCATIONS: list[Scene] = [
    Scene("own",   "自己脚步", "F2", "mask", "自己移动的脚步声（会盖住敌人脚步）"),
    Scene("below", "楼下脚步", "F3", "keep", "楼上/楼下敌人的脚步"),
    Scene("near",  "旁边脚步", "F4", "keep", "近处敌人的脚步"),
    Scene("far",   "远处脚步", "F5", "keep", "较远敌人的脚步"),
]

# 固定场景：与移动方式无关
FIXED_SCENES: list[Scene] = [
    Scene("idle",    "静止底噪",  "F1", "ref",    "什么都不做，只有环境底噪"),
    Scene("jump",    "跳跃/落地", "F6", "keep",   "起跳与落地"),
    Scene("fire",    "开火",      "F7", "reduce", "步枪/手枪连续开火"),
    Scene("grenade", "手榴弹",    "F8", "reduce", "破片手雷爆炸"),
    Scene("flash",   "震撼弹",    "F9", "reduce", "闪光/震撼弹爆响"),
]

# 全部可选热键（位置型 + 固定型）
SCENES: list[Scene] = LOCATIONS + FIXED_SCENES


def combo_key(location: str, gait: str) -> str:
    """组合场景的 key，例如 'near:run'。"""
    return f"{location}:{gait}"


def combo_name(location: str, gait: str) -> str:
    """组合场景的显示名，例如 '旁边脚步·奔跑'。"""
    loc = next(s for s in LOCATIONS if s.key == location)
    return f"{loc.name}·{GAIT_BY_KEY.get(gait, gait)}"


# 敌人脚步按位置分组，用于「共同频段」分析与距离衰减。
# own 单独一类：它和敌人脚步频段重叠但语义相反（是干扰源，不是目标）。
ENEMY_STEP_LOCATIONS = ("near", "below", "far")
OWN_STEP_LOCATION = "own"
# 全部位置型场景（含自己）
ALL_LOCATIONS = (OWN_STEP_LOCATION,) + ENEMY_STEP_LOCATIONS
# 枪声类干扰
GUNSHOT_KEYS = ("fire", "grenade", "flash")
# 枪声类干扰
GUNSHOT_KEYS = ("fire", "grenade", "flash")

SCENE_BY_KEY = {s.key: s for s in SCENES}
SCENE_BY_HOTKEY = {s.hotkey: s for s in SCENES}

LOCATION_BY_KEY = {s.key: s for s in LOCATIONS}


def parse_key(key: str) -> tuple[str, str | None]:
    """把场景 key 拆成 (位置, 移动方式)。固定场景的方式为 None。"""
    if ":" in key:
        loc, gait = key.split(":", 1)
        return loc, gait
    return key, None


def display_name(key: str) -> str:
    """任意 key（含组合）的显示名。"""
    loc, gait = parse_key(key)
    base = SCENE_BY_KEY.get(loc)
    if base is None:
        return key
    if gait is None:
        return base.name
    return f"{base.name}·{GAIT_BY_KEY.get(gait, gait)}"


@dataclass
class Recorder:
    """管理单场景的录制状态与统计累积。"""

    sample_rate_hz: float = 20.0     # 每秒送入的分析帧数
    min_loud_db: float = -75.0       # 低于此响度的帧视为无效（避免把静音录进场景）
    current: str | None = None
    started_at: float = 0.0
    frames: int = 0

    # 每个场景的原始帧收集（只存最近 30 秒，防止长时间录制吃内存）
    _buf: dict[str, list[np.ndarray]] = field(default_factory=dict)
    _loud: dict[str, list[float]] = field(default_factory=dict)

    MAX_FRAMES = 600  # 30 秒 @20fps

    # ---------------- 录制控制 ----------------

    def start(self, key: str) -> None:
        self.current = key
        self.started_at = time.perf_counter()
        self.frames = 0
        self._buf.setdefault(key, [])
        self._loud.setdefault(key, [])

    def stop(self) -> str | None:
        k = self.current
        self.current = None
        return k

    @property
    def is_recording(self) -> bool:
        return self.current is not None

    def elapsed(self) -> float:
        return time.perf_counter() - self.started_at if self.current else 0.0

    # ---------------- 数据摄入 ----------------

    def feed(self, inst_bands_db: np.ndarray, loud_db: float) -> None:
        """在录制中时，把一帧瞬时频谱累积到当前场景。"""
        if self.current is None:
            return
        if loud_db < self.min_loud_db:
            return  # 静音帧不录

        buf = self._buf.setdefault(self.current, [])
        buf.append(inst_bands_db.copy())
        if len(buf) > self.MAX_FRAMES:
            del buf[0 : len(buf) - self.MAX_FRAMES]
        self._loud.setdefault(self.current, []).append(loud_db)
        self.frames += 1

    # ---------------- 统计 ----------------

    def has_data(self, key: str) -> bool:
        return len(self._buf.get(key, [])) >= 3

    # ---- 已录制 key 的枚举（组合键支持的核心）----
    def recorded_keys(self, prefix: str | None = None) -> list[str]:
        """所有已录到的场景 key（含组合键）。

        为什么需要它：场景 key 可能是 "near:run" 这种组合，
        任何"按固定裸键名查找"的逻辑都会漏掉它们。
        例如要求 keep_keys=("near","below","far") 时，
        直接 `has_data("near")` 在只录了 near:run 的情况下返回 False，
        于是建议引擎认为"没有脚步声场景"，整套推导静默失效。
        """
        keys = [k for k, v in self._buf.items() if len(v) >= 3]
        if prefix is None:
            return sorted(keys)
        return sorted(k for k in keys if parse_key(k)[0] == prefix)

    def location_keys(self, location: str) -> list[str]:
        """某个位置下所有已录的移动方式 key。"""
        return self.recorded_keys(location)

    def coverage(self) -> dict[str, list[str]]:
        """位置 → 已录到的移动方式列表，用于显示录制进度。"""
        out: dict[str, list[str]] = {}
        for loc in ALL_LOCATIONS:
            gaits = [parse_key(k)[1] for k in self.location_keys(loc)]
            gaits = [g for g in gaits if g is not None]
            if gaits:
                out[loc] = [g for g, _ in GAITS if g in gaits]
        return out

    def clear(self, key: str | None = None) -> None:
        if key is None:
            self._buf.clear()
            self._loud.clear()
        else:
            self._buf.pop(key, None)
            self._loud.pop(key, None)

    def seconds_of(self, key: str) -> float:
        return len(self._buf.get(key, [])) / max(1e-6, self.sample_rate_hz)

    def profile(self, key: str) -> "SceneProfile | None":
        """把收集到的帧折算成场景画像。"""
        buf = self._buf.get(key, [])
        if len(buf) < 3:
            return None

        arr = np.array(buf)                       # (frames, 29) dB
        # dB → 线性功率，再在时间上平均（功率域平均才对）
        p = 10.0 ** (arr / 10.0)
        p_mean = np.mean(p, axis=0)
        mean_db = 10.0 * np.log10(np.maximum(p_mean, 1e-20))
        p95_db = 10.0 * np.log10(np.maximum(np.percentile(p, 95, axis=0), 1e-20))

        loud = np.array(self._loud.get(key, [0.0]))
        return SceneProfile(
            key=key,
            mean_db=mean_db,
            p95_db=p95_db,
            loud_mean_db=float(np.mean(loud)),
            loud_p95_db=float(np.percentile(loud, 95)),
            frames=len(buf),
            seconds=len(buf) / max(1e-6, self.sample_rate_hz),
        )


@dataclass
class SceneProfile:
    """一个场景的统计结果。"""

    key: str
    mean_db: np.ndarray      # 29 段，时间平均后的频谱形状
    p95_db: np.ndarray       # 29 段，95 分位（代表瞬态峰值）
    loud_mean_db: float
    loud_p95_db: float
    frames: int
    seconds: float

    @property
    def name(self) -> str:
        # 必须走 display_name：key 可能是 "near:run" 这种组合键，
        # 直接用 SCENE_BY_KEY[self.key] 会 KeyError。
        return display_name(self.key)

    def tilt(self) -> np.ndarray:
        """去掉整体电平，只保留频谱形状（相对倾斜）。"""
        return self.mean_db - float(np.mean(self.mean_db))


def describe(profile: SceneProfile, top: int = 4) -> str:
    """生成一句话的频谱特征描述。"""
    t = profile.tilt()
    order = np.argsort(t)[::-1][:top]
    parts = [f"{format_hz(BAND_CENTERS[i])}({t[i]:+.0f})" for i in order]
    return "、".join(parts)
