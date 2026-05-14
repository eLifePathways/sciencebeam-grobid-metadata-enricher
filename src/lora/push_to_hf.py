from __future__ import annotations

import argparse
import os
from pathlib import Path

ADAPTER_FILES = (
    "adapter_model.safetensors",
    "adapter_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "training_args.bin",
    "README.md",
)


def _push_one(adapter_dir: Path, repo_id: str, subfolder: str, commit_message: str) -> None:
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN env var required to push to HF")

    required = adapter_dir / "adapter_model.safetensors"
    if not required.exists():
        raise FileNotFoundError(f"missing {required} — refusing to push partial adapter")

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(
        folder_path=str(adapter_dir),
        repo_id=repo_id,
        repo_type="model",
        path_in_repo=subfolder,
        commit_message=f"{commit_message} [{subfolder}]",
        allow_patterns=list(ADAPTER_FILES),
        ignore_patterns=["checkpoint-*", "*.bin.tmp", "runs/*"],
    )
    print(f"pushed {adapter_dir} -> {repo_id}/{subfolder}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter-dir", required=True, type=Path,
                    help="Dir containing {task}_qwen35/ subdirs")
    ap.add_argument("--repo-id", required=True,
                    help="HF model repo, e.g. elifepathways/sciencebeam-lora-qwen3-8b")
    ap.add_argument("--commit-message", default="upload Qwen3-8B LoRA adapters")
    ap.add_argument("--tasks", nargs="*", default=None,
                    help="Subset of task names (without _qwen35 suffix); default = all found")
    args = ap.parse_args()

    candidates = sorted(p for p in args.adapter_dir.glob("*_qwen35") if p.is_dir())
    if not candidates:
        raise SystemExit(f"no *_qwen35 adapter dirs found under {args.adapter_dir}")

    if args.tasks:
        wanted = {f"{t}_qwen35" for t in args.tasks}
        candidates = [p for p in candidates if p.name in wanted]
        if not candidates:
            raise SystemExit(f"no matches for tasks={args.tasks}")

    for d in candidates:
        task = d.name[: -len("_qwen35")]
        _push_one(d, args.repo_id, subfolder=task, commit_message=args.commit_message)


if __name__ == "__main__":
    main()
