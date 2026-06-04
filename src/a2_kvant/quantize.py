"""Quantize a model with llm-compressor and save a vLLM-loadable checkpoint.

Pipeline: load FP16 -> calibrate on a small dataset -> apply recipe ->
save in `compressed-tensors` format. vLLM loads the result directly via
`--quantization compressed-tensors`.

This realizes the ТЗ flow "взять модель -> квантовать веса -> сохранить
сжатую для повторной загрузки" (ПМ «А2.Оптимизация», направление
"методы эффективного хранения больших моделей").
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

from a2_kvant.config import RunConfig
from a2_kvant.metrics import dir_size_gb
from a2_kvant.recipes import build_recipe

logger = logging.getLogger(__name__)

__all__ = ["quantize_model"]


def _load_calibration_dataset(cfg: RunConfig, tokenizer):
    """Load and tokenize a small calibration set for PTQ."""
    from datasets import load_dataset

    ds = load_dataset(cfg.quant.calibration_dataset, split="train_sft")
    ds = ds.shuffle(seed=cfg.seed).select(range(cfg.quant.num_calibration_samples))

    def _preprocess(example):
        text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
        return tokenizer(
            text,
            truncation=True,
            max_length=cfg.quant.max_seq_length,
            add_special_tokens=False,
        )

    return ds.map(_preprocess, remove_columns=ds.column_names)


def quantize_model(cfg: RunConfig) -> Tuple[str, float]:
    """Quantize `cfg.model_id` and write the checkpoint to `cfg.output_dir`.

    Returns (output_dir, on_disk_size_gb).
    """
    from llmcompressor import oneshot
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Loading FP16 model %s", cfg.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id, torch_dtype="auto", device_map=cfg.device
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)

    dataset = _load_calibration_dataset(cfg, tokenizer)
    recipe = build_recipe(cfg.quant)

    logger.info("Running oneshot quantization (recipe=%s)", cfg.quant.recipe)
    oneshot(
        model=model,
        dataset=dataset,
        recipe=recipe,
        max_seq_length=cfg.quant.max_seq_length,
        num_calibration_samples=cfg.quant.num_calibration_samples,
    )

    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, save_compressed=True)
    tokenizer.save_pretrained(out)

    size_gb = dir_size_gb(out)
    logger.info("Saved quantized checkpoint to %s (%.2f GiB on disk)", out, size_gb)
    return str(out), size_gb
