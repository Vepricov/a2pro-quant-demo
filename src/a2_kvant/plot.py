"""Render a FP16-vs-quantized comparison figure from results/compare.json.

Four panels: disk size, GPU memory, generation speed, quality (perplexity).
Each shows FP16 baseline vs the quantized model as annotated bars with a
green improvement badge. House style: narrow figure, large fonts, dpi 200.

Run locally (matplotlib available): does NOT need torch/GPU.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)

__all__ = ["render_comparison"]

# (json key, panel title, unit, lower_is_better)
_PANELS = [
    ("disk_size_gb", "Хранение на диске", "ГБ", True),
    ("vram_alloc_gb", "Память GPU (веса)", "ГБ", True),
    ("tokens_per_s", "Скорость генерации", "tok/s", False),
    ("hellaswag_acc_norm", "Качество (HellaSwag)", "", False),
]

_PARITY_THRESHOLD = 1.5  # |Δ%| ниже порога — статистический паритет, не «улучшение»

_BASE_COLOR = "#94A3B8"   # FP16 — нейтральный серо-синий
_OURS_COLOR = "#E4572E"   # квантованная — акцент
_GOOD_COLOR = "#1B9E4B"   # бейдж улучшения


def _badge(base: float, quant: float, lower_is_better: bool) -> str:
    """Improvement badge, e.g. '▼ 42%' / '▲ 46%'; '≈ паритет' внутри шума."""
    if base in (None, 0) or quant is None:
        return ""
    delta = (quant - base) / base * 100.0
    if abs(delta) < _PARITY_THRESHOLD:
        return "≈ паритет"
    arrow = "▼" if delta < 0 else "▲"
    return f"{arrow} {abs(delta):.0f}%"


def render_comparison(compare_json: str | Path, out_png: str | Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reports: List[Dict] = json.loads(Path(compare_json).read_text())
    base = next(r for r in reports if r["label"].startswith("fp16"))
    quant = next(r for r in reports if not r["label"].startswith("fp16"))
    quant_name = quant["label"].replace("quant_", "").upper()

    plt.rcParams.update({
        "font.size": 14, "axes.titlesize": 16,
        "xtick.labelsize": 14, "ytick.labelsize": 11,
        "axes.edgecolor": "#CBD5E1", "figure.facecolor": "white",
    })
    fig, axes = plt.subplots(1, 4, figsize=(13.0, 4.6), dpi=200)
    fig.suptitle(title, fontsize=20, fontweight="bold", y=1.04)

    for ax, (key, panel_title, unit, lower_better) in zip(axes, _PANELS):
        bvals = [base.get(key) or 0.0, quant.get(key) or 0.0]
        bars = ax.bar(["FP16", quant_name], bvals,
                      color=[_BASE_COLOR, _OURS_COLOR], width=0.62, zorder=3)
        ax.set_title(panel_title, fontweight="bold", pad=30)

        top = max(bvals) if max(bvals) > 0 else 1.0
        ax.set_ylim(0, top * 1.22)
        for bar, v in zip(bars, bvals):
            label = f"{v:.1f}" if v >= 10 else (f"{v:.3f}" if v < 1 else f"{v:.2f}")
            if unit:
                label += f" {unit}"
            ax.text(bar.get_x() + bar.get_width() / 2, v + top * 0.03, label,
                    ha="center", va="bottom", fontsize=13, fontweight="bold",
                    color="#0F172A")

        badge = _badge(base.get(key), quant.get(key), lower_better)
        if badge:
            # бейдж улучшения над панелью, зелёный
            ax.text(0.5, 1.06, badge, transform=ax.transAxes, ha="center",
                    fontsize=15, fontweight="bold", color=_GOOD_COLOR,
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="#E8F5EC",
                              edgecolor="none"))

        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.25, zorder=0)
        ax.set_yticks([])

    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Comparison figure -> %s", out_png)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser(description="Plot FP16 vs quantized comparison")
    ap.add_argument("--compare", default="results/compare.json")
    ap.add_argument("--out", default="results/comparison.png")
    ap.add_argument("--title", default="FP16 vs квантизация")
    args = ap.parse_args()
    render_comparison(args.compare, args.out, args.title)


if __name__ == "__main__":
    main()
