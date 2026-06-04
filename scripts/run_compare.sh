#!/usr/bin/env bash
# Full demo: evaluate FP16 baseline -> quantize -> evaluate quantized -> table.
# Usage: scripts/run_compare.sh configs/qwen2.5-7b-w8a8.json
set -euo pipefail
CONFIG="${1:-configs/qwen2.5-7b-w8a8.json}"
cd "$(dirname "$0")/.."
export PYTHONPATH="src:${PYTHONPATH:-}"
python -m a2_kvant.cli compare "$CONFIG"
echo "Done. See results/results_table.md"
