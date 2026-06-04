# A2.Pro — демонстрация квантизации и эффективного хранения моделей

Демо-цепочка для направления **«методы эффективного хранения больших моделей»**
(ПМ «А2.Оптимизация», ТЗ 4 Сгибнев п. 3.2.3.7; ТЗ 3 Безносиков + Дополнение,
мероприятие «эффективное хранение больших моделей»).

Сценарий: **взять модель → квантовать веса → сохранить сжатую → загрузить и
инферить → поднять по OpenAI-совместимому эндпоинту для платформы**.

Модели: Qwen 2.5 / 3 (7–8B). Инструмент квантизации: `llm-compressor`
(SmoothQuant + GPTQ + AWQ + FP8), формат `compressed-tensors` — vLLM грузит нативно.

## Что закрывает в ТЗ

| Требование ТЗ | Где реализовано |
| --- | --- |
| «методы эффективного хранения больших моделей» (ТЗ4 §3.2.3.7) | `quantize.py` + recipes (`w8a8`, `w4a16`, `fp8`) |
| «воспроизвести основные существующие подходы» (ТЗ3 §3.2/3.3) | recipe `w8a8` = SmoothQuant; `w4a16` = GPTQ (optimal-brain) |
| режим «инференс/оценка с расчётом метрик качества» (Доп. §3.5.1.2) | `evaluate.py` (perplexity) |
| сравнение с базой по качеству и ресурсам (Доп. §3.7.2) | `compare` → `results/results_table.md` (FP16 vs quant) |
| хранение/повторная загрузка сжатой модели | `save_pretrained(save_compressed=True)` → vLLM `--quantization compressed-tensors` |
| сценарий «инференс» в ПМ «А2.Сервер», клиент-сервер (ТЗ4 §3.2.3.10, §3.2.1.4) | `docker/docker-compose.yml` (vLLM OpenAI endpoint) + `serve_client.py` |
| Docker / Docker Compose (Доп. §3.5.3.2) | `docker/docker-compose.yml` |
| JSON вход/выход, фиксация конфигов/логов/метрик (Доп. §3.5.3.3, §3.5.7.1) | `config.py` (JSON), `results/*.json`, `seed` |

**Оговорка для отчёта.** Эта демка закрывает ногу «воспроизведение существующих
техник квантизации + интеграция в платформу». Ноги «новые методы на основе
оптимизационных постановок + теория сходимости» и «QAT» — отдельные научные
результаты раздела, демкой не закрываются.

## Структура

```
configs/        JSON-конфиги запусков (один конфиг = один прогон)
src/a2_kvant/   config, recipes (registry), quantize, evaluate, metrics, serve_client, cli
docker/         docker-compose.yml + .env.example для vLLM OpenAI endpoint
scripts/        run_compare.sh, serve.sh
results/        run_config.json, report_*.json, results_table.md (артефакты ПМИ)
```

## Запуск (на сервере с GPU, напр. brain_lab)

```bash
uv venv && uv pip install -e .

# 1) FP16 baseline -> квантизация -> метрики quant -> сравнительная таблица
scripts/run_compare.sh configs/qwen2.5-7b-w8a8.json

# 2) поднять квантованную модель по OpenAI-эндпоинту и проверить
cp docker/.env.example docker/.env   # поправить MODEL/порт
scripts/serve.sh
```

Платформа A2.Pro подключается к `http://<host>:8011/v1` (OpenAI-совместимый).

## Статус

Каркас. Локально проверены синтаксис и парсинг конфигов. Прогон на GPU
(квантизация + метрики + поднятие vLLM) — следующий шаг на brain_lab.
См. `PLAN.md`.
