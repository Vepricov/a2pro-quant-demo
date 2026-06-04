# Квантизация LLM: сжатие, хранение и инференс через OpenAI-эндпоинт

Полный конвейер: **берём модель → квантуем веса → сохраняем сжатую →
поднимаем в vLLM → подключаемся по OpenAI API**. Модель: Qwen3-8B,
квантизация W8A8 (SmoothQuant + GPTQ) через
[`llm-compressor`](https://github.com/vllm-project/llm-compressor).

![Qwen3-8B: FP16 vs W8A8](results/comparison.png)

| | FP16 | **W8A8 (наша)** | Δ |
|---|---|---|---|
| Хранение на диске | 16.0 ГБ | **8.81 ГБ** | **−45%** |
| Память GPU (веса) | 15.3 ГБ | **8.80 ГБ** | **−42%** |
| Скорость (vLLM, H200) | 78.2 tok/s | **114.0 tok/s** | **+46%** |
| Качество (perplexity ↓) | 8.62 | **8.53** | без потери |

Подробный отчёт с методикой: **[REPORT.md](REPORT.md)**.

## 🔧 Главные файлы квантизации

| Файл | Что делает |
|---|---|
| **[`src/a2_kvant/quantize.py`](src/a2_kvant/quantize.py)** | ядро: загрузка FP16 → калибровка → квантизация → сохранение `compressed-tensors` (vLLM грузит напрямую) |
| **[`src/a2_kvant/recipes.py`](src/a2_kvant/recipes.py)** | реестр рецептов: `w8a8` (SmoothQuant+GPTQ INT8), `w4a16` (GPTQ INT4), `fp8` |
| [`configs/qwen3-8b-w8a8.json`](configs/qwen3-8b-w8a8.json) | конфиг прогона: модель, рецепт, калибровочный датасет, eval |

Запуск квантизации:

```bash
uv venv && uv pip install -e .
PYTHONPATH=src python -m a2_kvant.cli quantize configs/qwen3-8b-w8a8.json
# или полный цикл с метриками до/после:
scripts/run_compare.sh configs/qwen3-8b-w8a8.json
```

## 🚀 Запуск наших квантованных моделей

В проекте можно поднимать **наши квантованные модели** (и оригинал для
сравнения) одной командой — обычный OpenAI-совместимый эндпоинт:

```bash
uv venv .venv-vllm && uv pip install --python .venv-vllm vllm

scripts/serve_model.sh w8a8    # 🔥 наша квантованная (8.8 ГБ, +46% скорости)
scripts/serve_model.sh fp16    # оригинальная FP16 — для сравнения
```

Подключение — как к любому OpenAI API (ключ генерируется в `.api_key`):

```python
from openai import OpenAI
client = OpenAI(base_url="http://<host>:8011/v1", api_key="<KEY>")
r = client.chat.completions.create(
    model="qwen3-8b-w8a8",
    messages=[{"role": "user", "content": "Привет!"}],
    max_tokens=100,
)
print(r.choices[0].message.content)
```

Вариант через Docker: [`docker/docker-compose.yml`](docker/docker-compose.yml)
(образ `vllm/vllm-openai`, чекпоинт пробрасывается volume-ом).

Интеграция с платформой (self-service и любой OpenAI-совместимый клиент) —
раздел 4 в [REPORT.md](REPORT.md).

## Структура

```
configs/          JSON-конфиги запусков (один конфиг = один прогон)
src/a2_kvant/     quantize ⭐, recipes ⭐, evaluate, metrics, plot, cli, serve_client
scripts/          run_compare.sh, serve_model.sh (w8a8|fp16), bench_endpoint.py
docker/           docker-compose для vLLM OpenAI-эндпоинта
results/          таблицы, JSON-отчёты, comparison.png
```

## Замечание по измерениям

Скорость и память квантованной модели нужно мерить **под vLLM**: у HF
transformers нет быстрых INT8-ядер, он распаковывает веса обратно в FP16 и
искажает обе метрики. Perplexity от движка не зависит. Детали — в
[REPORT.md](REPORT.md).
