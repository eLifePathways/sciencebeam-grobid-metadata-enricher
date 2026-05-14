"""Resolve a HuggingFace base model id to the corresponding fine-tuned
LoRA repo under the elifepathways org.

Usage:
    python scripts/resolve_hf_ft.py <hf-base-id>

Convention (the simplest): repo name `elifepathways/<flat-base>-ft`,
where `<flat-base>` is the basename after `/`. So `Qwen/Qwen3.5-9B` →
`elifepathways/Qwen3.5-9B-ft`. We HEAD the repo; if 404, exit 1. If
present, fetch `adapter_config.json` and print a JSON record:

    {"repo": "elifepathways/Qwen3.5-9B-ft",
     "base": "Qwen/Qwen3.5-9B",
     "rank": 16,
     "alpha": 32}
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request


HF_API = "https://huggingface.co/api/models"
HF_RESOLVE = "https://huggingface.co/{repo}/resolve/main/{path}"
ORG = "elifepathways"


def _hf_get_json(url: str, token: str | None) -> dict:
    req = urllib.request.Request(url)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: resolve_hf_ft.py <hf-base-id>", file=sys.stderr)
        return 2
    base = sys.argv[1]
    flat = base.split("/", 1)[1] if "/" in base else base
    repo = f"{ORG}/{flat}-ft"
    token = os.environ.get("HF_TOKEN")

    try:
        _hf_get_json(f"{HF_API}/{repo}", token)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            print(f"::error::HF repo {repo!r} not found (no fine-tune exists for {base!r})", file=sys.stderr)
            return 1
        if e.code == 401:
            hint = "" if token else " (set HF_TOKEN if the repo is private)"
            print(f"::error::HF repo {repo!r} unauthorised{hint}", file=sys.stderr)
            return 1
        raise

    try:
        cfg = _hf_get_json(HF_RESOLVE.format(repo=repo, path="adapter_config.json"), token)
    except urllib.error.HTTPError as e:
        if e.code in (401, 404):
            print(f"::error::adapter_config.json not accessible in {repo} (HTTP {e.code})", file=sys.stderr)
            return 1
        raise

    cfg_base = cfg.get("base_model_name_or_path", "")
    if cfg_base and cfg_base.lower() != base.lower():
        print(
            f"::warning::adapter base_model_name_or_path={cfg_base!r} disagrees with label base {base!r}; "
            f"using label base for vLLM",
            file=sys.stderr,
        )

    out = {
        "repo": repo,
        "base": base,
        "rank": cfg.get("r"),
        "alpha": cfg.get("lora_alpha"),
    }
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
