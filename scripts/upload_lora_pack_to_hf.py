"""Upload a 7-stage LoRA pack to HuggingFace under elifepathways/<flat>-ft.

Each task adapter lands as a top-level subdir of the HF repo:

    elifepathways/Qwen3.5-9B-ft/
        pack.json                 # manifest: base, tasks, step_lora_map
        body_sections/
            adapter_config.json
            adapter_model.safetensors
            ...
        figure_captions/...
        header_metadata/...
        identifiers/...
        keywords/...
        references/...
        table_captions/...

Task name detection: the script accepts subdirs named like
`body_sections_qwen35/` (suffix is the model lineage on GCS) and uploads
them under the canonical task name `body_sections/`.

Usage:
    python scripts/upload_lora_pack_to_hf.py \\
        --local-dir /path/to/pack \\
        --base-model Qwen/Qwen3.5-9B

    # or pull directly from GCS:
    python scripts/upload_lora_pack_to_hf.py \\
        --gcs-uri gs://bucket/lora/ \\
        --gcs-task-suffix '_qwen35' \\
        --base-model Qwen/Qwen3.5-9B
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional


# Canonical task list and the pipeline step -> task routing that the bench
# client uses via STEP_LORA_MAP_JSON. Kept aligned with the hardcoded map
# in .github/workflows/bench-all.yml; if you add a task here, update the
# workflow too (or pull this from a shared YAML).
CANONICAL_TASKS = [
    "body_sections",
    "figure_captions",
    "header_metadata",
    "identifiers",
    "keywords",
    "references",
    "table_captions",
]

DEFAULT_STEP_LORA_MAP: Dict[str, str] = {
    "HEADER_METADATA": "header_metadata",
    "TEI_METADATA": "header_metadata",
    "TEI_VALIDATED": "header_metadata",
    "CONTENT_BODY_SECTIONS": "body_sections",
    "CONTENT_REFERENCES": "references",
    "CONTENT_FIGURE_CAPTIONS": "figure_captions",
    "CONTENT_TABLE_CAPTIONS": "table_captions",
    "KEYWORD_TRANSLATE": "keywords",
    "KEYWORD_SELECT": "keywords",
    "KEYWORD_EXTRACT": "keywords",
    "KEYWORD_INFER": "keywords",
    "IDENTIFIER_SELECT": "identifiers",
}

# Files vLLM (and peft) actually need at serve time. Everything else
# (training state, optimizer, full tokenizer.json, etc) is excluded so
# the HF repo stays small and downloads are fast on cold-start.
ADAPTER_INCLUDE_PATTERNS = [
    "adapter_config.json",
    "adapter_model.safetensors",
    "adapter_model.bin",   # legacy float-name
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "processor_config.json",
]


def _flat_base(base_model: str) -> str:
    return base_model.split("/", 1)[1] if "/" in base_model else base_model


def _strip_task_suffix(name: str, suffix: str) -> str:
    """Map e.g. body_sections_qwen35 -> body_sections."""
    if suffix and name.endswith(suffix):
        return name[: -len(suffix)]
    # Generic: drop any trailing _<lineage> if it doesn't match a canonical task.
    if name in CANONICAL_TASKS:
        return name
    m = re.match(r"^(.+?)_[a-z0-9]+$", name)
    if m and m.group(1) in CANONICAL_TASKS:
        return m.group(1)
    return name


def _stage_local_from_gcs(gcs_uri: str, dest: Path) -> None:
    # Mirror just one level deep — sub-adapter dirs sit directly under gcs_uri.
    print(f"[stage] gsutil rsync {gcs_uri} -> {dest}", flush=True)
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["gsutil", "-m", "rsync", "-r", "-x", r"^checkpoint-[0-9]+/", gcs_uri, str(dest)],
        check=True,
    )


def _discover_tasks(root: Path, base_model: str, suffix: str) -> List[Dict[str, str]]:
    """Find adapter sub-dirs under root. Each must contain adapter_config.json."""
    tasks: List[Dict[str, str]] = []
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        cfg = sub / "adapter_config.json"
        if not cfg.exists():
            continue
        try:
            adapter_base = json.loads(cfg.read_text()).get("base_model_name_or_path", "")
        except Exception:  # pylint: disable=broad-except
            adapter_base = ""
        task = _strip_task_suffix(sub.name, suffix)
        tasks.append({"name": task, "local_dir": str(sub), "adapter_base": adapter_base})
    return tasks


def _build_manifest(base_model: str, tasks: List[Dict[str, str]]) -> Dict:
    # Only ship step_lora_map entries that map to a present task name —
    # avoids dangling references when a pack is missing one of the 7.
    present = {t["name"] for t in tasks}
    step_map = {k: v for k, v in DEFAULT_STEP_LORA_MAP.items() if v in present}
    return {
        "base_model": base_model,
        "tasks": [t["name"] for t in tasks],
        "step_lora_map": step_map,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--local-dir", type=Path,
                     help="Local dir holding one subdir per task")
    src.add_argument("--gcs-uri", type=str,
                     help="GCS dir with subdirs per task (eg gs://bucket/lora/)")
    ap.add_argument("--base-model", required=True,
                    help="HF base model id, eg Qwen/Qwen3.5-9B")
    ap.add_argument("--gcs-task-suffix", default="",
                    help="Suffix to strip from each subdir name "
                         "(eg _qwen35 -> body_sections_qwen35 becomes body_sections)")
    ap.add_argument("--repo",
                    help="HF repo id (default: elifepathways/<flat>-ft)")
    ap.add_argument("--private", action="store_true", default=True,
                    help="Create the HF repo as private (default true)")
    ap.add_argument("--public", action="store_true",
                    help="Override --private and create as public")
    ap.add_argument("--dry-run", action="store_true",
                    help="List planned uploads without writing anything")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token and not args.dry_run:
        print("::error::HF_TOKEN not set", file=sys.stderr)
        return 1

    flat = _flat_base(args.base_model)
    repo_id = args.repo or f"elifepathways/{flat}-ft"

    # Stage GCS to a temp dir if needed; local dir is used as-is.
    stage_root: Optional[Path] = None
    if args.gcs_uri:
        stage_root = Path(tempfile.mkdtemp(prefix="lora_pack_"))
        _stage_local_from_gcs(args.gcs_uri.rstrip("/") + "/", stage_root)
        root = stage_root
    else:
        root = args.local_dir
    if not root.is_dir():
        print(f"::error::not a dir: {root}", file=sys.stderr)
        return 1

    tasks = _discover_tasks(root, args.base_model, args.gcs_task_suffix)
    if not tasks:
        print(f"::error::no adapter sub-dirs (with adapter_config.json) under {root}", file=sys.stderr)
        return 1

    print(f"[plan] repo={repo_id} private={(not args.public)}")
    for t in tasks:
        print(f"  task={t['name']:20s} local={t['local_dir']} adapter_base={t['adapter_base']}")
    manifest = _build_manifest(args.base_model, tasks)
    print(f"[plan] manifest tasks={manifest['tasks']}")
    print(f"[plan] step_lora_map keys={list(manifest['step_lora_map'])}")

    if args.dry_run:
        print("[dry-run] not uploading.")
        if stage_root:
            shutil.rmtree(stage_root, ignore_errors=True)
        return 0

    from huggingface_hub import HfApi  # local import keeps script importable without hf_hub
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model",
                    exist_ok=True, private=(not args.public))

    # Upload each task subdir under <task>/, applying include patterns so
    # we keep only files vLLM/peft need at serve time.
    for t in tasks:
        print(f"[upload] {t['name']} -> {repo_id}/{t['name']}/")
        api.upload_folder(
            folder_path=t["local_dir"],
            path_in_repo=t["name"],
            repo_id=repo_id,
            repo_type="model",
            allow_patterns=ADAPTER_INCLUDE_PATTERNS,
            commit_message=f"upload {t['name']} adapter",
        )

    # Write manifest at repo root.
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(manifest, f, indent=2)
        manifest_path = f.name
    api.upload_file(
        path_or_fileobj=manifest_path,
        path_in_repo="pack.json",
        repo_id=repo_id,
        repo_type="model",
        commit_message="update pack manifest",
    )
    Path(manifest_path).unlink(missing_ok=True)
    print(f"[done] repo: https://huggingface.co/{repo_id}")

    if stage_root:
        shutil.rmtree(stage_root, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
