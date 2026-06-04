"""Render a FP16-vs-quantized comparison figure from results/compare.json.

Four panels (2x2): disk size, VRAM, throughput, perplexity. Each shows the
FP16 baseline vs the quantized model as annotated bars, with the relative
change called out. House style: narrow figure, large fonts, dpi 200.

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
    ("disk_size_gb", "Размер на диске", "ГБ", True),
    ("vram_alloc_gb", "VRAM при инференсе", "ГБ", True),
    ("tokens_per_s", "Скорость генерации", "tok/s", False),
    ("perplexity", "Perplexity (WikiText-2)", "", True),
]

_BASELINE_COLOR = "#4C72B0"  # FP16
_QUANT_COLOR = "#C44E52"     # quantized


def _pct_change(base: float, quant: float, lower_is_better: bool) -> str:
    """Human-readable relative change with a 'good/bad' framing arrow."""
    if base in (None, 0) or quant is None:
        return ""
    delta = (quant - base) / base * 100.0
    if lower_is_better:
        verb = "меньше" if delta < 0 else "больше"
        return f"{abs(delta):.0f}% {verb}"
    verb = "быстрее" if delta > 0 else "медленнее"
    return f"{abs(delta):.0f}% {verb}"


def render_comparison(compare_json: str | Path, out_png: str | Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reports: List[Dict] = json.loads(Path(compare_json).read_text())
    base = next(r for r in reports if r["label"].startswith("fp16"))
    quant = next(r for r in reports if not r["label"].startswith("fp16"))

    plt.rcParams.update({
        "font.size": 14, "axes.titlesize": 17, "axes.labelsize": 14,
        "xtick.labelsize": 13, "ytick.labelsize": 12,
    })
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 8.0), dpi=200)
    fig.suptitle(title, fontsize=20, fontweight="bold", y=0.98)

    for ax, (key, panel_title, unit, lower_better) in zip(axes.flat, _PANELS):
        bvals = [base.get(key) or 0.0, quant.get(key) or 0.0]
        bars = ax.bar(
            ["FP16", quant["label"].replace("quant_", "").upper()],
            bvals, color=[_BASELINE_COLOR, _QUANT_COLOR], width=0.6,
        )
        ax.set_title(panel_title, fontweight="bold")
        if unit:
            ax.set_ylabel(unit)
        top = max(bvals) if max(bvals) > 0 else 1.0
        ax.set_ylim(0, top * 1.25)
        for bar, v in zip(bars, bvals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + top * 0.03,
                    f"{v:.2f}{(' ' + unit) if unit else ''}",
                    ha="center", va="bottom", fontsize=13, fontweight="bold")
        change = _pct_change(base.get(key), quant.get(key), lower_better)
        if change:
            ax.text(0.5, 0.90, change, transform=ax.transAxes, ha="center",
                    fontsize=14, color=_QUANT_COLOR, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
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
