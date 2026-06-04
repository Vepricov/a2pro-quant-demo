"""Quality + resource comparison of a model checkpoint.

Computes WikiText perplexity (quality) and disk/VRAM/latency (resource)
for a single checkpoint. Run it once on the FP16 baseline and once on the
quantized checkpoint, then diff the two JSON reports.

Quality metric (perplexity) + resource metrics satisfy ТЗ Дополнение
п. 3.5.1.2 (режим инференса/оценки) and п. 3.7.2 (сравнение с базой).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

from a2_kvant.config import RunConfig
from a2_kvant.metrics import dir_size_gb, measure_latency, vram_used_gb

logger = logging.getLogger(__name__)

__all__ = ["evaluate_checkpoint", "perplexity"]


def perplexity(model, tokenizer, cfg: RunConfig) -> float:
    """Sliding-window perplexity on WikiText (lower is better)."""
    import torch
    from datasets import load_dataset

    data = load_dataset(
        cfg.eval.ppl_dataset, cfg.eval.ppl_dataset_config, split="test"
    )
    text = "\n\n".join(data["text"][: cfg.eval.ppl_max_samples])
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc["input_ids"].to(model.device)

    max_len = cfg.quant.max_seq_length
    nlls = []
    n_tokens = 0
    for begin in range(0, input_ids.shape[1] - 1, max_len):
        end = min(begin + max_len, input_ids.shape[1])
        ids = input_ids[:, begin:end]
        with torch.no_grad():
            out = model(ids, labels=ids)
        seq_len = ids.shape[1]
        nlls.append(out.loss * seq_len)
        n_tokens += seq_len

    return float(torch.exp(torch.stack(nlls).sum() / n_tokens))


def evaluate_checkpoint(model_path: str, cfg: RunConfig, label: str) -> Dict:
    """Load `model_path`, measure quality + resources, return a report dict."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("[%s] loading %s", label, model_path)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype="auto", device_map=cfg.device
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()

    ppl = perplexity(model, tokenizer, cfg)
    latency = measure_latency(
        model,
        tokenizer,
        prompt=cfg.eval.latency_prompt,
        max_new_tokens=cfg.eval.latency_max_new_tokens,
        runs=cfg.eval.latency_runs,
        device=cfg.device,
    )

    report = {
        "label": label,
        "model_path": model_path,
        "perplexity": ppl,
        "disk_size_gb": dir_size_gb(model_path) if Path(model_path).is_dir() else None,
        "vram_alloc_gb": vram_used_gb(),
        **latency,
    }
    logger.info("[%s] ppl=%.3f vram=%.2fGiB tok/s=%.1f", label, ppl,
                report["vram_alloc_gb"], report["tokens_per_s"])
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return report
