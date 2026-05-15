"""Resolve a HuggingFace base model id to the corresponding fine-tuned
LoRA repo under the elifepathways org.

Convention (the simplest): repo name `elifepathways/<flat-base>-ft`,
where `<flat-base>` is the basename after `/`. So `Qwen/Qwen3.5-9B` →
`elifepathways/Qwen3.5-9B-ft`.

The repo can hold either:
  - a 7-task PACK with `pack.json` at the root and one subdir per task,
    each containing `adapter_config.json` + `adapter_model.safetensors`
    (see scripts/upload_lora_pack_to_hf.py for the canonical layout); or
  - a single LoRA adapter at the root (back-compat with the original
    single-adapter convention).

Usage:
    python scripts/resolve_hf_ft.py <hf-base-id>

Pack output:
    {"mode": "pack",
     "repo": "elifepathways/Qwen3.5-9B-ft",
     "base": "Qwen/Qwen3.5-9B",
     "tasks": ["body_sections", "figure_captions", ...],
     "step_lora_map": {"CONTENT_BODY_SECTIONS": "body_sections", ...}}

Single-adapter output:
    {"mode": "single",
     "repo": "elifepathways/Qwen3.5-9B-ft",
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


def _resolve_pack(repo: str, base: str, token: str | None) -> dict | None:
    try:
        manifest = _hf_get_json(HF_RESOLVE.format(repo=repo, path="pack.json"), token)
    except urllib.error.HTTPError as e:
        if e.code in (401, 404):
            return None
        raise
    tasks = manifest.get("tasks") or []
    if not tasks:
        return None
    manifest_base = manifest.get("base_model") or manifest.get("base") or ""
    if manifest_base and manifest_base.lower() != base.lower():
        print(
            f"::warning::pack.json base={manifest_base!r} disagrees with label base {base!r}",
            file=sys.stderr,
        )
    return {
        "mode": "pack",
        "repo": repo,
        "base": base,
        "tasks": tasks,
        "step_lora_map": manifest.get("step_lora_map") or {},
    }


def _resolve_single(repo: str, base: str, token: str | None) -> dict | None:
    try:
        cfg = _hf_get_json(HF_RESOLVE.format(repo=repo, path="adapter_config.json"), token)
    except urllib.error.HTTPError as e:
        if e.code in (401, 404):
            return None
        raise
    cfg_base = cfg.get("base_model_name_or_path", "")
    if cfg_base and cfg_base.lower() != base.lower():
        print(
            f"::warning::adapter base_model_name_or_path={cfg_base!r} disagrees with label base {base!r}",
            file=sys.stderr,
        )
    return {
        "mode": "single",
        "repo": repo,
        "base": base,
        "rank": cfg.get("r"),
        "alpha": cfg.get("lora_alpha"),
    }


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

    out = _resolve_pack(repo, base, token) or _resolve_single(repo, base, token)
    if out is None:
        print(f"::error::repo {repo} has neither pack.json nor adapter_config.json", file=sys.stderr)
        return 1
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
