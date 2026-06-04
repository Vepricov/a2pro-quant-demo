"""Command-line entrypoint: quantize, evaluate, compare.

Subcommands:
  quantize   <config.json>            -> writes quantized checkpoint
  evaluate   <config.json> --path P --label L   -> writes one JSON report
  compare    <config.json>            -> full run: eval(FP16) + quantize +
                                          eval(quantized) + results table

All artifacts (config snapshot, per-model JSON reports, markdown table) go
to results/ — JSON I/O as required by ТЗ Дополнение п. 3.5.7.1.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

from a2_kvant.config import RunConfig, load_config

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")


def _set_seed(seed: int) -> None:
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _write_table(reports: List[Dict], path: Path) -> None:
    """Render a markdown comparison table (baseline vs quantized)."""
    cols = ["label", "perplexity", "disk_size_gb", "vram_alloc_gb",
            "tokens_per_s", "latency_ms_mean"]
    lines = ["| " + " | ".join(cols) + " |",
             "| " + " | ".join("---" for _ in cols) + " |"]
    for r in reports:
        cells = []
        for c in cols:
            v = r.get(c)
            cells.append(f"{v:.3f}" if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n")
    logger.info("Results table -> %s", path)


def _cmd_quantize(cfg: RunConfig) -> None:
    from a2_kvant.quantize import quantize_model

    out, size = quantize_model(cfg)
    logger.info("Quantized -> %s (%.2f GiB)", out, size)


def _cmd_evaluate(cfg: RunConfig, path: str, label: str) -> None:
    from a2_kvant.evaluate import evaluate_checkpoint

    RESULTS_DIR.mkdir(exist_ok=True)
    report = evaluate_checkpoint(path, cfg, label)
    out = RESULTS_DIR / f"report_{label}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    logger.info("Report -> %s", out)


def _cmd_compare(cfg: RunConfig) -> None:
    from a2_kvant.evaluate import evaluate_checkpoint
    from a2_kvant.quantize import quantize_model

    RESULTS_DIR.mkdir(exist_ok=True)
    cfg.to_json(RESULTS_DIR / "run_config.json")

    base = evaluate_checkpoint(cfg.model_id, cfg, "fp16_baseline")
    out_dir, _ = quantize_model(cfg)
    quant = evaluate_checkpoint(out_dir, cfg, f"quant_{cfg.quant.recipe}")

    reports = [base, quant]
    (RESULTS_DIR / "compare.json").write_text(
        json.dumps(reports, indent=2, ensure_ascii=False)
    )
    _write_table(reports, RESULTS_DIR / "results_table.md")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    ap = argparse.ArgumentParser(prog="a2-kvant")
    sub = ap.add_subparsers(dest="cmd", required=True)

    for name in ("quantize", "evaluate", "compare"):
        p = sub.add_parser(name)
        p.add_argument("config", help="path to run config JSON")
        if name == "evaluate":
            p.add_argument("--path", required=True, help="model path/id to evaluate")
            p.add_argument("--label", required=True, help="report label")

    args = ap.parse_args()
    cfg = load_config(args.config)
    _set_seed(cfg.seed)

    if args.cmd == "quantize":
        _cmd_quantize(cfg)
    elif args.cmd == "evaluate":
        _cmd_evaluate(cfg, args.path, args.label)
    elif args.cmd == "compare":
        _cmd_compare(cfg)


if __name__ == "__main__":
    main()
