#!/usr/bin/env bash
# Bring up the vLLM OpenAI-compatible server for a quantized checkpoint,
# then smoke-test it. Run after the checkpoint exists in checkpoints/.
set -euo pipefail
cd "$(dirname "$0")/../docker"
[ -f .env ] || { echo "Create docker/.env from .env.example first"; exit 1; }
docker compose up -d
echo "Waiting for health..."
until curl -sf "http://localhost:${HOST_PORT:-8011}/health" >/dev/null; do sleep 5; done
echo "Endpoint up. Smoke test:"
cd ..
export PYTHONPATH="src:${PYTHONPATH:-}"
SERVED_NAME=$(grep -E '^SERVED_NAME=' docker/.env | cut -d= -f2)
python -m a2_kvant.serve_client --base-url "http://localhost:${HOST_PORT:-8011}/v1" --model "$SERVED_NAME"
