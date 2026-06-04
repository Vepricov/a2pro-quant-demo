"""Resource metrics: on-disk size, VRAM, latency/throughput.

The ТЗ (Дополнение, п. 3.7.2) requires comparison against the baseline by
quality metrics AND, where applicable, resource metrics. This module covers
the resource side; quality (perplexity) lives in evaluate.py.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)

__all__ = ["dir_size_gb", "vram_used_gb", "measure_latency"]


def dir_size_gb(path: str | Path) -> float:
    """Total size of a model directory in GiB (weights on disk)."""
    total = sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())
    return total / 1024**3


def vram_used_gb(device_index: int = 0) -> float:
    """Currently allocated CUDA memory on a device, in GiB."""
    import torch

    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated(device_index) / 1024**3


def measure_latency(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    runs: int,
    device: str,
) -> Dict[str, float]:
    """Greedy-decode `prompt` `runs` times; report tokens/s and ms/run.

    One warmup run is discarded so kernels are compiled before timing.
    """
    import torch

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=False)

    with torch.no_grad():  # warmup
        model.generate(**inputs, **gen_kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    durations: List[float] = []
    n_tokens = 0
    for _ in range(runs):
        start = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**inputs, **gen_kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        durations.append(time.perf_counter() - start)
        n_tokens = out.shape[-1] - inputs["input_ids"].shape[-1]

    mean_s = sum(durations) / len(durations)
    return {
        "latency_ms_mean": 1000.0 * mean_s,
        "latency_ms_min": 1000.0 * min(durations),
        "tokens_per_s": n_tokens / mean_s if mean_s > 0 else 0.0,
        "generated_tokens": float(n_tokens),
    }
