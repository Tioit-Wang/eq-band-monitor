"""文本对齐工具：中英混排时按显示宽度补齐。

Python 的 str.ljust/rjust 按**字符数**补齐，但 CJK 字符占两列显示宽度，
所以「走路」只算 2 个字符却居 4 列，与英文混排时列就会错位。
"""

from __future__ import annotations

import unicodedata


def dwidth(s: str) -> int:
    """字符串的显示宽度（CJK 全角算 2，其余算 1）。"""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def pad(s: str, width: int, align: str = "left") -> str:
    """按显示宽度补齐到 width 列。"""
    gap = max(0, width - dwidth(s))
    if align == "right":
        return " " * gap + s
    if align == "center":
        left = gap // 2
        return " " * left + s + " " * (gap - left)
    return s + " " * gap
