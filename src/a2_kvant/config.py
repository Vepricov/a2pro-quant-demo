"""Immutable run configuration loaded from JSON.

The ТЗ (Дополнение, п. 3.5.7.1) requires all input/output to be JSON.
A run is fully described by one JSON config -> one frozen dataclass.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

__all__ = ["QuantConfig", "EvalConfig", "RunConfig", "load_config"]


@dataclass(frozen=True)
class QuantConfig:
    """Quantization recipe selection. Recipe bodies live in recipes.py."""

    recipe: str  # registered recipe name, e.g. "w8a8" / "w4a16"
    calibration_dataset: str = "HuggingFaceH4/ultrachat_200k"
    num_calibration_samples: int = 512
    max_seq_length: int = 2048
    smoothing_strength: float = 0.8  # used by SmoothQuant-based recipes


@dataclass(frozen=True)
class EvalConfig:
    """What to measure when comparing FP16 baseline vs quantized model."""

    ppl_dataset: str = "wikitext"
    ppl_dataset_config: str = "wikitext-2-raw-v1"
    ppl_max_samples: int = 256
    latency_prompt: str = "Explain quantization of large language models in two sentences."
    latency_max_new_tokens: int = 128
    latency_runs: int = 5


@dataclass(frozen=True)
class RunConfig:
    """Top-level config for one quantize+evaluate+serve demo run."""

    model_id: str  # HF id or local path, e.g. "Qwen/Qwen2.5-7B-Instruct"
    output_dir: str  # where the quantized checkpoint is written
    device: str = "cuda:0"
    quant: QuantConfig = field(default_factory=lambda: QuantConfig(recipe="w8a8"))
    eval: EvalConfig = field(default_factory=EvalConfig)
    seed: int = 42

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False))
        logger.info("Config written to %s", path)


def load_config(path: str | Path) -> RunConfig:
    """Load a RunConfig from JSON, building nested frozen dataclasses."""
    raw: Dict[str, Any] = json.loads(Path(path).read_text())
    quant = QuantConfig(**raw.pop("quant", {}))
    eval_cfg = EvalConfig(**raw.pop("eval", {}))
    cfg = RunConfig(quant=quant, eval=eval_cfg, **raw)
    logger.info("Loaded config for model=%s recipe=%s", cfg.model_id, cfg.quant.recipe)
    return cfg
