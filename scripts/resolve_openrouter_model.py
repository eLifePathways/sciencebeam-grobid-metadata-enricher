"""Resolve a HuggingFace model id to an OpenRouter model id.

Usage:
    python scripts/resolve_openrouter_model.py <hf-id>

Hits OpenRouter's /v1/models endpoint, builds a canonical-form index, and
prints the matching OpenRouter id to stdout. Exits 1 with a list of the
closest candidates if no match is found, so the failing label is easy to
fix.

Canonicalisation strips case and all separators (`/`, `-`, `_`, `.`) so
`Qwen/Qwen2.5-7B-Instruct` matches `qwen/qwen-2.5-7b-instruct`.
"""
from __future__ import annotations

import difflib
import json
import os
import re
import sys
import urllib.request


OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"


def _canonical(s: str) -> str:
    return re.sub(r"[-_./]+", "", s.lower())


def _fetch_models(api_key: str | None) -> list[dict]:
    req = urllib.request.Request(OPENROUTER_MODELS_URL)
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    with urllib.request.urlopen(req, timeout=20) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body.get("data", [])


def resolve(hf_id: str, models: list[dict]) -> str | None:
    ids = [m["id"] for m in models]
    target_lower = hf_id.lower()
    for mid in ids:
        if mid.lower() == target_lower:
            return mid
    target_canon = _canonical(hf_id)
    for mid in ids:
        if _canonical(mid) == target_canon:
            return mid
    return None


def _closest(hf_id: str, models: list[dict], n: int = 5) -> list[str]:
    ids = [m["id"] for m in models]
    return difflib.get_close_matches(hf_id.lower(), [i.lower() for i in ids], n=n, cutoff=0.4)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: resolve_openrouter_model.py <hf-id>", file=sys.stderr)
        return 2
    hf_id = sys.argv[1]
    api_key = os.environ.get("OPENROUTER_API_KEY")
    models = _fetch_models(api_key)
    match = resolve(hf_id, models)
    if match:
        print(match)
        return 0
    print(f"::error::No OpenRouter model matches {hf_id!r}", file=sys.stderr)
    near = _closest(hf_id, models)
    if near:
        print("Closest OpenRouter ids:", file=sys.stderr)
        for n in near:
            print(f"  {n}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
