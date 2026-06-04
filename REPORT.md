# Демонстрация: эффективное хранение больших моделей через квантизацию

**Платформа A2.Pro · ПМ «Оптимизация» · направление «методы эффективного хранения больших моделей»**

Сценарий: берём модель → квантуем веса → сохраняем сжатую → загружаем и
инферим → поднимаем по OpenAI-совместимому эндпоинту → платформа (self-service)
работает с ней как с обычной моделью.

## 1. Результаты

Модель: **Qwen3-8B**. Квантизация: **W8A8** (SmoothQuant + GPTQ, INT8 веса и
активации) через [`llm-compressor`](https://github.com/vllm-project/llm-compressor),
формат `compressed-tensors` — vLLM загружает напрямую.

| | FP16 (оригинал) | W8A8 (квантованная) | Δ |
|---|---|---|---|
| Размер на диске | 16.0 ГБ | **8.81 ГБ** | **−45%** |
| Память GPU (веса) | 15.27 ГБ | **8.80 ГБ** | **−42%** |
| Скорость генерации (vLLM, H200) | 78.2 tok/s | **114.0 tok/s** | **+46%** |
| Perplexity (WikiText-2) | 8.622 | **8.533** | качество сохранено |

![Сравнение FP16 vs W8A8](results/comparison.png)

Замечание по методике: скорость и память измерены под **vLLM** (у HF
transformers нет быстрых INT8-ядер: он распаковывает квантованные веса в FP16,
что искажает обе метрики). Perplexity посчитана через transformers, она от
движка не зависит. Конфигурация замеров: `results/run_config.json`, сырые
отчёты: `results/compare_vllm.json`.

## 2. Как воспроизвести

```bash
uv venv && uv pip install -e .            # окружение квантизации
uv venv .venv-vllm && uv pip install --python .venv-vllm vllm  # окружение сервинга

# полный прогон: метрики FP16 -> квантизация -> метрики W8A8 -> таблица
scripts/run_compare.sh configs/qwen3-8b-w8a8.json

# графики из результатов
python src/a2_kvant/plot.py --compare results/compare_vllm.json \
    --out results/comparison.png --title "Qwen3-8B: FP16 vs W8A8 (vLLM)"
```

Код квантизации: `src/a2_kvant/quantize.py` (oneshot-пайплайн) и
`src/a2_kvant/recipes.py` (реестр рецептов: `w8a8`, `w4a16`, `fp8` — рецепт
выбирается полем `quant.recipe` в JSON-конфиге).

## 3. Выбор модели при сервинге: квантованная или оригинал

```bash
scripts/serve_model.sh w8a8    # квантованная  -> http://<host>:8011/v1, модель qwen3-8b-w8a8
scripts/serve_model.sh fp16    # оригинальная  -> модель qwen3-8b-fp16
```

Эндпоинт OpenAI-совместимый, защищён API-ключом (генерируется в `.api_key`).
Проверка:

```python
from openai import OpenAI
client = OpenAI(base_url="http://<host>:8011/v1", api_key="<KEY>")
r = client.chat.completions.create(model="qwen3-8b-w8a8",
    messages=[{"role": "user", "content": "Привет!"}], max_tokens=100)
print(r.choices[0].message.content)
```

Альтернатива через Docker: `docker/docker-compose.yml` (образ
`vllm/vllm-openai`, чекпоинт пробрасывается volume-ом, параметры в `.env`).

## 4. Интеграция с платформой (self-service)

Проверено на развёрнутом [deeppavlov/self-service]: в `secrets/llm.env`
указывается адрес нашего vLLM и имя модели:

```bash
LLM_API_KEY="<KEY>"
LLM_BASE_URL="http://<vllm-host>:8011/v1"   # из docker-сети платформы: gateway, напр. http://172.20.0.1:8011/v1
LLM_MODEL_NAME="qwen3-8b-w8a8"              # или qwen3-8b-fp16 — выбор модели
OVERRIDE_MODEL_NAME="qwen3-8b-w8a8"
```

После `docker compose up --build` ассистент платформы отвечает через
квантованную модель (проверено по логам vLLM: запросы приходят с IP контейнера
ассистента). Переключение на оригинал = поменять `LLM_MODEL_NAME` и адрес/порт
на FP16-инстанс и пересобрать backend.

Практические замечания по развёртыванию:
- секреты копируются в образ backend при сборке — после смены `llm.env` нужны
  пересборка backend и удаление `tmp-*` образов ассистентов;
- из docker-сети платформы хост доступен по gateway сети
  (`docker network inspect <net> --format '{{(index .IPAM.Config 0).Gateway}}'`),
  а не по внешнему IP.

## 5. Структура репозитория

```
configs/          JSON-конфиги прогонов (модель, рецепт, калибровка, eval)
src/a2_kvant/     quantize / recipes / evaluate / metrics / plot / cli / serve_client
scripts/          run_compare.sh, serve_model.sh (выбор w8a8|fp16), bench_endpoint.py
docker/           docker-compose для vLLM OpenAI-эндпоинта
results/          таблицы, JSON-отчёты, comparison.png
```

## 6. Соответствие ТЗ

| Требование | Реализация |
|---|---|
| Методы эффективного хранения больших моделей (ТЗ Платформа §3.2.3.7) | квантизация W8A8/W4A16/FP8 + save/reload `compressed-tensors` |
| Воспроизведение существующих подходов (ТЗ §3.2/3.3) | SmoothQuant, GPTQ (optimal-brain постановка) |
| Режим инференса/оценки с метриками качества (Доп. §3.5.1.2) | perplexity + сравнение с базой |
| Сравнение с базой по качеству и ресурсам (Доп. §3.7.2) | таблица и график: диск/VRAM/скорость/ppl |
| Docker/Docker Compose (Доп. §3.5.3.2) | `docker/docker-compose.yml` |
| JSON вход/выход, конфиги и логи (Доп. §3.5.7.1) | JSON-конфиги и JSON-отчёты, фиксированный seed |
| Сценарий «инференс» в ПМ «Сервер», клиент-сервер | OpenAI-эндпоинт vLLM + интеграция self-service |
