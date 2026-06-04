"""Quantization recipe registry.

Each recipe maps a short name to a list of llm-compressor modifiers.
Recipes are built lazily (factory) so importing this module does not
require llm-compressor unless a recipe is actually requested.

Registered recipes:
  - w8a8  : SmoothQuant + GPTQ, INT8 weights + INT8 activations (the
            method named in the ТЗ: "воспроизведение существующих техник").
  - w4a16 : GPTQ INT4 weights, FP16 activations (optimal-brain postановка,
            ТЗ п. 3.2 "методы на основе оптимизационных постановок").
  - fp8   : FP8 dynamic weights+activations (modern storage format).
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List

from a2_kvant.config import QuantConfig

logger = logging.getLogger(__name__)

__all__ = ["build_recipe", "list_recipes", "register_recipe"]

# name -> builder(cfg) -> list of modifiers
RECIPE_FACTORY: Dict[str, Callable[[QuantConfig], List]] = {}


def register_recipe(name: str) -> Callable:
    """Decorator registering a recipe builder under `name`."""

    def decorator(fn: Callable[[QuantConfig], List]) -> Callable:
        RECIPE_FACTORY[name] = fn
        return fn

    return decorator


@register_recipe("w8a8")
def _w8a8(cfg: QuantConfig) -> List:
    from llmcompressor.modifiers.quantization import GPTQModifier
    from llmcompressor.modifiers.transform.smoothquant import SmoothQuantModifier

    return [
        SmoothQuantModifier(smoothing_strength=cfg.smoothing_strength),
        GPTQModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"]),
    ]


@register_recipe("w4a16")
def _w4a16(cfg: QuantConfig) -> List:
    from llmcompressor.modifiers.quantization import GPTQModifier

    return [GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])]


@register_recipe("fp8")
def _fp8(cfg: QuantConfig) -> List:
    from llmcompressor.modifiers.quantization import QuantizationModifier

    return [QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"])]


def build_recipe(cfg: QuantConfig) -> List:
    """Return the modifier list for the recipe named in `cfg.recipe`."""
    if cfg.recipe not in RECIPE_FACTORY:
        raise KeyError(f"Unknown recipe '{cfg.recipe}'. Available: {list_recipes()}")
    logger.info("Building recipe '%s'", cfg.recipe)
    return RECIPE_FACTORY[cfg.recipe](cfg)


def list_recipes() -> List[str]:
    return sorted(RECIPE_FACTORY)
