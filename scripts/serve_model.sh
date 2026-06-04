#!/usr/bin/env bash
# Поднять vLLM OpenAI-эндпоинт с ВЫБОРОМ модели: квантованная или оригинал.
#   scripts/serve_model.sh w8a8        # квантованная (по умолчанию), порт 8011
#   scripts/serve_model.sh fp16        # оригинальная FP16
#   scripts/serve_model.sh w8a8 8012   # свой порт
# Ключ берётся из .api_key (создаётся автоматически при первом запуске).
set -euo pipefail
KIND="${1:-w8a8}"
PORT="${2:-8011}"
cd "$(dirname "$0")/.."

[ -f .api_key ] || openssl rand -hex 12 > .api_key
KEY=$(cat .api_key)

case "$KIND" in
  w8a8) MODEL="checkpoints/Qwen3-8B-W8A8"; NAME="qwen3-8b-w8a8" ;;
  fp16) MODEL="Qwen/Qwen3-8B";             NAME="qwen3-8b-fp16" ;;
  *) echo "usage: $0 [w8a8|fp16] [port]"; exit 1 ;;
esac

# .venv-vllm/bin в PATH обязателен: vLLM зовёт ninja сабпроцессом (JIT INT8-кернелы)
export PATH="$PWD/.venv-vllm/bin:$PATH"
echo "Serving $NAME ($MODEL) on port $PORT"
exec .venv-vllm/bin/vllm serve "$MODEL" \
  --served-model-name "$NAME" \
  --port "$PORT" \
  --api-key "$KEY" \
  --gpu-memory-utilization 0.30 \
  --max-model-len 8192
