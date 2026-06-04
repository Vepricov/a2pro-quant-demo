"""Smoke-test a vLLM OpenAI-compatible endpoint serving the quantized model.

The A2.Pro platform connects to models over OpenAI-compatible endpoints, so
the acceptance check is "can the platform talk to our served quantized model".
This client mirrors that call path (ПМ «А2.Сервер», сценарий "инференс").
"""

from __future__ import annotations

import argparse
import logging

logger = logging.getLogger(__name__)

__all__ = ["chat"]


def chat(base_url: str, model: str, prompt: str, api_key: str = "EMPTY") -> str:
    """Send one chat completion to an OpenAI-compatible endpoint."""
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=256,
        temperature=0.0,
    )
    return resp.choices[0].message.content


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser(description="Test vLLM OpenAI endpoint")
    ap.add_argument("--base-url", default="http://localhost:8011/v1")
    ap.add_argument("--model", required=True, help="served model name/path")
    ap.add_argument("--prompt", default="Кратко: что такое квантизация LLM?")
    args = ap.parse_args()
    print(chat(args.base_url, args.model, args.prompt))


if __name__ == "__main__":
    main()
