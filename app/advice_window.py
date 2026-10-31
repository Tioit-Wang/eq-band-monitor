"""交接包窗口：展示原始音频分析结果，并给出文件所在位置。

这个窗口的职责变了：
  旧版是"软件替你算好 EQ 建议"；
  新版是"把原始音频 + 已算好的事实交给你/AI 去判断"。

所以它主要显示：
  * 每个文件的数据质量（削波/静音/时长）
  * 场景的声学特征（瞬态性、音调性、频谱）
  * 交接包路径（report.md / report.json / audio/）
"""

from __future__ import annotations

import os
import subprocess

from PySide6.QtCore import QTimer
from PySide6.QtGui import QFont, QGuiApplication
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
)

from .analyzer import BAND_CENTERS, format_hz
from .handoff import render_report
from .rawanalyze import RawReport
from .textfmt import pad

BG = "#0d1117"
PANEL = "#161b22"
TEXT = "#e6edf3"
DIM = "#7d8590"
ACCENT = "#22d3ee"
RED = "#f87171"
GREEN = "#3fb950"


class AdviceWindow(QDialog):
    """展示交接包内容。"""

    def __init__(self, report: RawReport, bundle_dir: str,
                 base_preset_info: str | None = None, parent=None) -> None:
        super().__init__(parent)
        self.report = report
        self.bundle_dir = bundle_dir
        self.base_info = base_preset_info

        self.setWindowTitle("交接包已生成 — EQ Band Monitor")
        self.resize(860, 780)
        self.setStyleSheet(
            f"QDialog{{background:{BG};}} QLabel{{color:{TEXT};}} "
            f"QTextEdit{{background:{PANEL};color:{TEXT};border:1px solid #30363d;"
            f"border-radius:6px;font-family:Consolas,'Microsoft YaHei';font-size:12px;}} "
            f"QPushButton{{background:#21262d;color:{TEXT};border:1px solid #30363d;"
            f"border-radius:6px;padding:7px 14px;}} "
            f"QPushButton:hover{{background:#30363d;}}"
        )

        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 14, 16, 14)
        lay.setSpacing(9)

        title = QLabel("原始音频已录制并分析")
        title.setFont(QFont("Segoe UI Semibold", 13))
        lay.addWidget(title)

        n_ok = len(report.clips)
        bad = [c for c in report.clips if c.clipped or c.silent or c.seconds < 1.5]
        color = RED if bad else GREEN
        status = QLabel(
            f"共 {n_ok} 个场景"
            + (f"，其中 {len(bad)} 个有数据质量问题（见下方）" if bad else "，数据质量正常")
        )
        status.setWordWrap(True)
        status.setStyleSheet(f"color:{color};font-size:12px;padding:8px;"
                             f"background:{PANEL};border-radius:6px;")
        lay.addWidget(status)

        path_lbl = QLabel(
            f"交接包位置：\n{self.bundle_dir}\n\n"
            "把 report.md（或整个文件夹）交给 AI，它就能基于这些事实给出 EQ 调整方案。"
        )
        path_lbl.setWordWrap(True)
        path_lbl.setStyleSheet(f"color:{ACCENT};font-size:11px;padding:8px;"
                               f"background:{PANEL};border-radius:6px;")
        lay.addWidget(path_lbl)

        # 概览表
        tbl_lbl = QLabel("录制概览：")
        tbl_lbl.setStyleSheet(f"color:{DIM};font-size:11px;")
        lay.addWidget(tbl_lbl)

        self.overview = QTextEdit()
        self.overview.setReadOnly(True)
        self.overview.setPlainText(self._overview_text())
        self.overview.setMinimumHeight(170)
        lay.addWidget(self.overview)

        # 完整报告
        rep_lbl = QLabel("完整报告（report.md）：")
        rep_lbl.setStyleSheet(f"color:{DIM};font-size:11px;")
        lay.addWidget(rep_lbl)

        self.full = QTextEdit()
        self.full.setReadOnly(True)
        self.full.setPlainText(render_report(report, base_preset_info))
        lay.addWidget(self.full, 1)

        row = QHBoxLayout()
        row.addStretch(1)

        self.btn_open = QPushButton("打开文件夹")
        self.btn_open.clicked.connect(self._open_dir)
        row.addWidget(self.btn_open)

        self.btn_copy = QPushButton("复制报告全文")
        self.btn_copy.clicked.connect(self._copy_all)
        row.addWidget(self.btn_copy)

        self.btn_copy_path = QPushButton("复制文件夹路径")
        self.btn_copy_path.clicked.connect(self._copy_path)
        row.addWidget(self.btn_copy_path)

        bclose = QPushButton("关闭")
        bclose.clicked.connect(self.close)
        row.addWidget(bclose)
        lay.addLayout(row)

    # ---------------- 内容 ----------------

    def _overview_text(self) -> str:
        lines = [
            pad("场景", 10) + pad("时长", 8, "right") + pad("峰值dB", 10, "right")
            + pad("RMS dB", 10, "right") + pad("底噪dB", 10, "right")
            + pad("起音/秒", 10, "right") + pad("音调性", 9, "right") + "  状态",
            "-" * 84,
        ]
        for c in self.report.clips:
            t = c.transients
            flags = []
            if c.clipped:
                flags.append("削波")
            if c.silent:
                flags.append("静音")
            if c.seconds < 1.5:
                flags.append("太短")
            lines.append(
                pad(c.name, 10)
                + pad(f"{c.seconds:.1f}s", 8, "right")
                + pad(f"{c.peak_db:.1f}", 10, "right")
                + pad(f"{c.rms_db:.1f}", 10, "right")
                + pad(f"{c.noise_floor_db:.1f}", 10, "right")
                + pad(f"{t.onsets_per_sec:.1f}" if t else "—", 10, "right")
                + pad(f"{c.tonality:.1f}", 9, "right")
                + "  " + (("⚠ " + " ".join(flags)) if flags else "OK")
            )
        lines.append("")
        lines.append("起音/秒 高 = 脉冲声（枪声/脚步）；低 = 持续声")
        lines.append("音调性 高 = 有离散音高（乐音）；低 = 宽带噪声")
        return "\n".join(lines)

    # ---------------- 动作 ----------------

    def _open_dir(self) -> None:
        try:
            os.startfile(self.bundle_dir)  # noqa: S606
        except Exception:
            pass

    def _copy_all(self) -> None:
        QGuiApplication.clipboard().setText(self.full.toPlainText())
        self.btn_copy.setText("已复制 ✓")
        QTimer.singleShot(1500, lambda: self.btn_copy.setText("复制报告全文"))

    def _copy_path(self) -> None:
        QGuiApplication.clipboard().setText(self.bundle_dir)
        self.btn_copy_path.setText("已复制 ✓")
        QTimer.singleShot(1500, lambda: self.btn_copy_path.setText("复制文件夹路径"))
