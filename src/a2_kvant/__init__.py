"""A2.Pro quantization demo: quantize -> save -> evaluate -> serve (vLLM).

Realizes the ТЗ direction "методы эффективного хранения больших моделей"
(ПМ «А2.Оптимизация») as a runnable demonstration on Qwen 2.5/3 models.
"""

from a2_kvant.config import EvalConfig, QuantConfig, RunConfig, load_config

__all__ = ["RunConfig", "QuantConfig", "EvalConfig", "load_config"]
__version__ = "0.1.0"
