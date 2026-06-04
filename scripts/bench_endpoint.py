"""Measure single-request generation speed of an OpenAI-compatible endpoint.

Sends `runs` chat completions (greedy, fixed max_tokens), reports mean
tokens/s computed from usage.completion_tokens / elapsed wall time.
Usage:
  python bench_endpoint.py --base-url http://localhost:8011/v1 \
      --model qwen3-8b-w8a8 --api-key KEY --runs 3 --max-tokens 256
"""

from __future__ import annotations

import argparse
import json
import time

from openai import OpenAI

PROMPT = "Explain quantization of large language models in two sentences."


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=256)
    args = ap.parse_args()

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    # warmup
    client.chat.completions.create(
        model=args.model, messages=[{"role": "user", "content": PROMPT}],
        max_tokens=16, temperature=0.0,
    )
    speeds, latencies = [], []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        resp = client.chat.completions.create(
            model=args.model, messages=[{"role": "user", "content": PROMPT}],
            max_tokens=args.max_tokens, temperature=0.0,
        )
        dt = time.perf_counter() - t0
        n = resp.usage.completion_tokens
        speeds.append(n / dt)
        latencies.append(dt * 1000)

    print(json.dumps({
        "model": args.model,
        "tokens_per_s": sum(speeds) / len(speeds),
        "latency_ms_mean": sum(latencies) / len(latencies),
        "completion_tokens": args.max_tokens,
        "runs": args.runs,
    }, indent=2))


if __name__ == "__main__":
    main()
