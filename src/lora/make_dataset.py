"""Build per-task LoRA training/validation/test JSONLs from the HF benchmark dataset.

Output: 21 JSONLs (7 tasks × 3 HF splits) at <out-dir>/{train,validation,test}/task_<task>.jsonl.
Splits come from the HF dataset's own train/validation/test partition — never recomputed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

ENR_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ENR_ROOT / "src"))

from grobid_metadata_enricher.pipeline import (  # noqa: E402
    build_body_section_candidate_evidence,
    build_figure_caption_candidate_evidence,
    build_reference_candidate_evidence,
    build_table_caption_candidate_evidence,
    extract_alto_lines,
    prune_layout_lines,
)
from grobid_metadata_enricher.prompts import (  # noqa: E402
    BODY_SECTIONS_EXTRACTION_PROMPT,
    FIGURE_CAPTIONS_SELECTION_PROMPT,
    HEADER_METADATA_PROMPT,
    IDENTIFIER_SELECTION_PROMPT,
    KEYWORD_EXTRACTION_PROMPT,
    REFERENCES_EXTRACTION_PROMPT,
    TABLE_CAPTIONS_SELECTION_PROMPT,
)

HF_REPO = "elifepathways/sciencebeam-v2-benchmarking"
HF_CORPORA = ("biorxiv", "ore", "pkp", "scielo_br", "scielo_mx", "scielo-preprints")
# Output splits: HF train is bucketed into train/ + train_val/ (deterministic
# SHA-256 mod 5, bucket 0 → train_val) so early-stopping never sees HF
# validation (= bench:all production test set).
OUTPUT_SPLITS = ("train", "train_val", "validation", "test")
TRAIN_VAL_BUCKET = 0
TRAIN_VAL_MOD = 5


def _front_matter(lines, max_lines: int = 120) -> str:
    out = []
    for i, ln in enumerate(lines[:max_lines]):
        text = ln.get("text", "")
        if not text:
            continue
        out.append(f"{i+1:03d} | y={ln.get('y', 0):.1f} x={ln.get('x', 0):.1f} | {text}")
    return "\n".join(out)


def _gold_ref(g: dict) -> dict:
    titles = g.get("reference_titles") or []
    dois = g.get("reference_dois") or []
    refs = []
    for i, title in enumerate(titles):
        item = {"title": title}
        if i < len(dois) and dois[i]:
            item["doi"] = dois[i]
        refs.append(item)
    return {"references": refs}


def _gold_header(g: dict) -> dict:
    return {
        "title": g.get("title", ""),
        "authors": g.get("authors", []),
        "abstract": (g.get("abstract") or "")[:2500],
        "keywords": g.get("keywords", []),
        "identifiers": g.get("identifiers", []),
    }


TASKS: Dict[str, dict] = {
    "body_sections": dict(
        sys=BODY_SECTIONS_EXTRACTION_PROMPT,
        cand_fn=build_body_section_candidate_evidence,
        gold_fn=lambda g: {"body_sections": g.get("body_sections", [])},
        gold_key="body_sections",
    ),
    "figure_captions": dict(
        sys=FIGURE_CAPTIONS_SELECTION_PROMPT,
        cand_fn=build_figure_caption_candidate_evidence,
        gold_fn=lambda g: {"figure_captions": g.get("figure_captions", [])},
        gold_key="figure_captions",
    ),
    "table_captions": dict(
        sys=TABLE_CAPTIONS_SELECTION_PROMPT,
        cand_fn=build_table_caption_candidate_evidence,
        gold_fn=lambda g: {"table_captions": g.get("table_captions", [])},
        gold_key="table_captions",
    ),
    "references": dict(
        sys=REFERENCES_EXTRACTION_PROMPT,
        cand_fn=build_reference_candidate_evidence,
        gold_fn=_gold_ref,
        gold_key="reference_titles",
    ),
    "header_metadata": dict(
        sys=HEADER_METADATA_PROMPT,
        cand_fn=_front_matter,
        gold_fn=_gold_header,
        gold_key="title",
    ),
    "identifiers": dict(
        sys=IDENTIFIER_SELECTION_PROMPT,
        cand_fn=_front_matter,
        gold_fn=lambda g: {"identifiers": g.get("identifiers", [])},
        gold_key="identifiers",
    ),
    "keywords": dict(
        sys=KEYWORD_EXTRACTION_PROMPT,
        cand_fn=_front_matter,
        gold_fn=lambda g: {"keywords": g.get("keywords", [])},
        gold_key="keywords",
    ),
}


def _train_val_bucket(rid: str) -> int:
    h = hashlib.sha256(rid.encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big") % TRAIN_VAL_MOD


def _hf_split_ids(parquet_dir: Path, hf_repo: str = HF_REPO) -> Dict[str, set]:
    """Return {split: set(ppr_id)} for the 4 output splits (train/train_val/validation/test)."""
    import pyarrow.parquet as pq

    raw: Dict[str, set] = {"train": set(), "validation": set(), "test": set()}
    for split in raw:
        for corpus in HF_CORPORA:
            path = parquet_dir / f"{corpus}-jats" / f"{split}-00000-of-00001.parquet"
            if not path.exists():
                continue
            ids = pq.ParquetFile(path).read(columns=["ppr_id"]).column("ppr_id").to_pylist()
            raw[split].update(ids)

    train_val = {rid for rid in raw["train"] if _train_val_bucket(rid) == TRAIN_VAL_BUCKET}
    return {
        "train": raw["train"] - train_val,
        "train_val": train_val,
        "validation": raw["validation"],
        "test": raw["test"],
    }


def _candidate_text(rid: str, corpus: str, alto_root: Path, cand_fn: Callable) -> Optional[str]:
    alto = alto_root / corpus / "alto_full" / f"{rid}.alto.xml"
    if not alto.exists():
        return None
    lines = extract_alto_lines(str(alto))
    content = prune_layout_lines(lines)
    out = cand_fn(content)
    if isinstance(out, list):
        return out[0] if out else None
    return out


def _doc_id(d: dict) -> Optional[str]:
    return d.get("record_id") or d.get("id")


def _iter_per_doc(per_doc_path: Path) -> Iterable[dict]:
    with per_doc_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def build(
    task: str,
    per_doc_path: Path,
    alto_root: Path,
    split_ids: Dict[str, set],
    cap_chars: int,
) -> Dict[str, List[dict]]:
    spec = TASKS[task]
    out: Dict[str, List[dict]] = {s: [] for s in OUTPUT_SPLITS}
    seen = kept = no_alto = empty_cand = empty_gold = unassigned = 0

    for d in _iter_per_doc(per_doc_path):
        seen += 1
        rid = _doc_id(d)
        corpus = d.get("corpus")
        if not rid or not corpus:
            continue

        target_split = next((s for s, ids in split_ids.items() if rid in ids), None)
        if target_split is None:
            unassigned += 1
            continue

        gold_obj = spec["gold_fn"](d.get("gold") or {})
        primary = next(iter(gold_obj.values()), None)
        if primary in (None, "", [], {}):
            empty_gold += 1
            continue

        cand = _candidate_text(rid, corpus, alto_root, spec["cand_fn"])
        if cand is None:
            no_alto += 1
            continue
        if len(cand.strip()) < 100:
            empty_cand += 1
            continue
        cand = cand[:cap_chars]

        out[target_split].append({
            "messages": [
                {"role": "system", "content": spec["sys"]},
                {"role": "user", "content": cand},
                {"role": "assistant", "content": json.dumps(gold_obj, ensure_ascii=False)},
            ],
            "corpus": corpus,
            "id": rid,
        })
        kept += 1

    print(
        f"[{task}] seen={seen} kept={kept} no_alto={no_alto} "
        f"empty_cand={empty_cand} empty_gold={empty_gold} unassigned={unassigned}",
        flush=True,
    )

    ids = {s: {r["id"] for r in out[s]} for s in OUTPUT_SPLITS}
    for a, b in [(x, y) for i, x in enumerate(OUTPUT_SPLITS) for y in OUTPUT_SPLITS[i + 1:]]:
        if ids[a] & ids[b]:
            raise RuntimeError(f"[{task}] split overlap {a}∩{b} = {len(ids[a] & ids[b])}")
    return out


def _write_split(rows: List[dict], path: Path, seed: int) -> None:
    rng = random.Random(seed)
    rng.shuffle(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _push_to_hf(out_dir: Path, repo_id: str, commit_message: str) -> None:
    from huggingface_hub import HfApi  # local import

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN env var required to push to HF")
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    api.upload_folder(
        folder_path=str(out_dir),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=commit_message,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--parquet-dir", required=True, type=Path,
                    help="Local dir holding HF parquets (e.g. data_benchmark_200/parquet)")
    ap.add_argument("--alto-root", required=True, type=Path,
                    help="Dir holding {corpus}/alto_full/{rid}.alto.xml")
    ap.add_argument("--per-doc", required=True, type=Path,
                    help="per_document.jsonl with gold annotations per record")
    ap.add_argument("--tasks", nargs="+", default=list(TASKS.keys()))
    ap.add_argument("--cap-chars", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--push-to-hf", default=None,
                    help="If set, upload <out-dir> to this HF dataset repo (e.g. user/lora-data)")
    ap.add_argument("--commit-message", default="rebuild from sciencebeam-v2-benchmarking")
    args = ap.parse_args()

    for t in args.tasks:
        if t not in TASKS:
            raise SystemExit(f"unknown task: {t}")

    split_ids = _hf_split_ids(args.parquet_dir)
    overall_ids: Dict[str, set] = {s: set() for s in OUTPUT_SPLITS}

    for task in args.tasks:
        rows_by_split = build(task, args.per_doc, args.alto_root, split_ids, args.cap_chars)
        for split, rows in rows_by_split.items():
            if not rows:
                print(f"  [{task}/{split}] EMPTY — skipping write", flush=True)
                continue
            path = args.out_dir / split / f"task_{task}.jsonl"
            _write_split(rows, path, args.seed)
            overall_ids[split].update(r["id"] for r in rows)
            print(f"  wrote {path} ({len(rows)})", flush=True)

    for a, b in [(x, y) for i, x in enumerate(OUTPUT_SPLITS) for y in OUTPUT_SPLITS[i + 1:]]:
        overlap = overall_ids[a] & overall_ids[b]
        if overlap:
            raise RuntimeError(f"cross-task split bleed: {a}∩{b} = {len(overlap)}")

    fingerprint = hashlib.sha256()
    for split in OUTPUT_SPLITS:
        for rid in sorted(overall_ids[split]):
            fingerprint.update(f"{split}/{rid}\n".encode())
    (args.out_dir / "DATASET_FINGERPRINT").write_text(fingerprint.hexdigest() + "\n")

    if args.push_to_hf:
        _push_to_hf(args.out_dir, args.push_to_hf, args.commit_message)


if __name__ == "__main__":
    main()
