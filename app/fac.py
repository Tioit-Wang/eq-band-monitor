"""FxSound 预设（.fac）读写。

格式（实测解析自 Export/暗区突围.fac，CRLF 换行，UTF-8）：

    CLASS1 : Effect Type
    9: Version
    <预设名>
    0: Double Params Flag
    1: Total number of elements
    25: Main 0 ... 25: Main 5
    0: Element Number
       <n>: Param <m>            ← 6 个
    7: Number of Application Dependent Integers
    0: Number of Application Dependent Reals
    0: Number of Application Dependent Strings
    1: Integer[0] ... 2: Integer[6]   ← 7 个
    10: Number of EQ Bands
    1: On/Off Flag
    Band 1
       62.5: CF
       1.9685: Boost/Cut
    ... 共 10 组

关键设计决定：**尽量"保留原始行、只改数值"**，而不是重新生成整个文件。
这样 FxSound 里那些我们没完全理解的字段（Main 0-5、Integer[0-6]）会原样保留，
避免因为猜错含义导致导入失败或行为改变。
"""

from __future__ import annotations

from dataclasses import dataclass, field

CRLF = "\r\n"


@dataclass
class EqBand:
    """一个 EQ 频段。"""

    cf: float = 1000.0        # 中心频率 Hz
    gain: float = 0.0         # 增益 dB（Boost/Cut）
    q: float = 1.4            # 品质因数（.fac 里不存，仅用于计算/显示）

    def __str__(self) -> str:
        return f"{self.cf:g}Hz {self.gain:+.2f}dB Q={self.q:.2f}"


@dataclass
class FacPreset:
    """一个 FxSound 预设。"""

    name: str = "EQ Band Monitor"
    bands: list[EqBand] = field(default_factory=list)
    enabled: bool = True
    version: int = 9
    # 原样保留的行（那些没解析语义的字段）
    raw_lines: list[str] = field(default_factory=list)

    MAX_BANDS = 10

    def trimmed(self) -> list[EqBand]:
        """补齐或截断到 10 个频段（FxSound 固定 10 段）。"""
        out = list(self.bands[: self.MAX_BANDS])
        while len(out) < self.MAX_BANDS:
            out.append(EqBand(16000.0, 0.0))
        return out


def parse(text: str) -> FacPreset:
    """解析 .fac 文本。"""
    lines = text.split("\r\n") if "\r\n" in text else text.split("\n")
    preset = FacPreset(raw_lines=lines)

    bands: list[EqBand] = []
    in_bands = False
    pending_cf: float | None = None
    pending_idx = -1

    for i, raw in enumerate(lines):
        s = raw.strip()

        if s.startswith("CLASS1"):
            continue
        if s.endswith(": Version") and preset.version == 9:
            try:
                preset.version = int(s.split(":")[0])
            except ValueError:
                pass
            # 名字在版本行下一行
            if i + 1 < len(lines):
                preset.name = lines[i + 1].strip()
            continue

        low = s.lower()
        if low.endswith(": number of eq bands"):
            in_bands = True
            continue
        if low.endswith(": on/off flag"):
            try:
                preset.enabled = s.split(":")[0].strip() != "0"
            except Exception:
                preset.enabled = True
            continue

        if in_bands:
            if low.startswith("band "):
                pending_idx += 1
                continue
            if low.endswith(": cf"):
                try:
                    pending_cf = float(s.split(":")[0])
                except ValueError:
                    pending_cf = None
                continue
            if low.endswith(": boost/cut"):
                try:
                    g = float(s.split(":")[0])
                except ValueError:
                    g = 0.0
                if pending_cf is not None:
                    bands.append(EqBand(pending_cf, g))
                pending_cf = None
                continue

    preset.bands = bands
    return preset


_VALUE_SUFFIXES = (": boost/cut", ": cf", ": on/off flag")


def render(preset: FacPreset, band_lines: list[str] | None = None) -> str:
    """把预设写回 .fac 文本。

    策略：以原始行为骨架，仅替换
      * 预设名（版本行下一行）
      * On/Off Flag
      * 每个 Band 的 CF / Boost/Cut 数值行
    其余行（Main 0-5、Integer[0-6] 等）原样输出。
    """
    src = list(preset.raw_lines)
    if not src:
        src = _synthesize_skeleton(preset)

    bands = preset.trimmed()
    out: list[str] = []

    # 定位版本行与名字行
    ver_idx = -1
    for i, raw in enumerate(src):
        if raw.strip().endswith(": Version"):
            ver_idx = i
            break
    name_idx = ver_idx + 1 if ver_idx >= 0 and ver_idx + 1 < len(src) else -1

    band_i = -1        # 当前是第几个 Band
    waiting_cf = False

    for i, raw in enumerate(src):
        if i == name_idx:
            out.append(preset.name)
            continue

        s = raw.strip()
        low = s.lower()

        if name_idx < 0 and ver_idx >= 0 and i > ver_idx and low.startswith("class"):
            # 极端情况兜底：不处理
            pass

        if low.endswith(": on/off flag"):
            indent = raw[: len(raw) - len(raw.lstrip())]
            out.append(f"{indent}{1 if preset.enabled else 0}: On/Off Flag")
            continue

        if low.startswith("band "):
            band_i += 1
            out.append(raw)
            continue

        if low.endswith(": cf"):
            waiting_cf = False
            if 0 <= band_i < len(bands):
                indent = raw[: len(raw) - len(raw.lstrip())]
                out.append(f"{indent}{bands[band_i].cf:g}: CF")
            else:
                out.append(raw)
            continue

        if low.endswith(": boost/cut"):
            if 0 <= band_i < len(bands):
                indent = raw[: len(raw) - len(raw.lstrip())]
                g = bands[band_i].gain
                out.append(f"{indent}{_fmt_gain(g)}: Boost/Cut")
            else:
                out.append(raw)
            continue

        out.append(raw)

    return CRLF.join(out)


def _fmt_gain(g: float) -> str:
    """FxSound 的数值格式：整数不写小数点（原文件里 '9:' 而不是 '9.0:'）。"""
    if abs(g - round(g)) < 1e-9:
        return f"{int(round(g))}"
    return f"{g:g}"


def _synthesize_skeleton(preset: FacPreset) -> list[str]:
    """从零构造一个合法的 .fac 骨架（当没有原始文件可参考时）。"""
    lines = [
        "CLASS1 : Effect Type",
        f"{preset.version}: Version",
        preset.name,
        "0: Double Params Flag",
        "1: Total number of elements",
        "25: Main 0",
        "0: Main 1",
        "0: Main 2",
        "0: Main 3",
        "85: Main 4",
        "25: Main 5",
        "0: Element Number",
        "   0: Param 0",
        "   0: Param 1",
        "   0: Param 2",
        "   0: Param 3",
        "   0: Param 4",
        "   0: Param 5",
        "   0: Param 6",
        "7: Number of Application Dependent Integers",
        "0: Number of Application Dependent Reals",
        "0: Number of Application Dependent Strings",
        "1: Integer[0]",
        "0: Integer[1]",
        "0: Integer[2]",
        "1: Integer[3]",
        "1: Integer[4]",
        "0: Integer[5]",
        "2: Integer[6]",
        f"{preset.MAX_BANDS}: Number of EQ Bands",
        f"{1 if preset.enabled else 0}: On/Off Flag",
    ]
    for i, b in enumerate(preset.trimmed(), start=1):
        lines.append(f"Band {i}")
        lines.append(f"   {b.cf:g}: CF")
        lines.append(f"   {_fmt_gain(b.gain)}: Boost/Cut")
    # 原文件结尾有一个空行（CRLF 结尾）
    lines.append("")
    return lines


def load(path: str) -> FacPreset:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return parse(f.read())


def save(path: str, preset: FacPreset) -> None:
    text = render(preset)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(text)
