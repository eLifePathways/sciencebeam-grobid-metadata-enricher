from __future__ import annotations

import argparse
import difflib
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

_RESET = "\033[0m"
_RED = "\033[91m"
_GREEN = "\033[92m"

_SHORT_WORD_THRESHOLD = 50

_METRIC_TO_FIELD: Dict[str, str] = {
    "title_match": "title",
    "title_edit_sim": "title",
    "abstract_precision": "abstract",
    "abstract_recall": "abstract",
    "abstract_f1": "abstract",
    "abstract_edit_sim": "abstract",
    "language_match": "language",
    "date_match": "date",
    "rights_match": "rights",
    "publisher_match": "publisher",
    # list fields — word-diff not yet supported
    "authors_recall": "authors",
    "keywords_precision": "keywords",
    "keywords_recall": "keywords",
    "keywords_f1": "keywords",
    "types_precision": "types",
    "types_recall": "types",
    "types_f1": "types",
    "formats_precision": "formats",
    "formats_recall": "formats",
    "formats_f1": "formats",
    "relations_precision": "relations",
    "relations_recall": "relations",
    "relations_f1": "relations",
    "identifiers_precision": "identifiers",
    "identifiers_recall": "identifiers",
    "identifiers_f1": "identifiers",
    "body_section_precision": "body_sections",
    "body_section_recall": "body_sections",
    "body_section_f1": "body_sections",
    "figure_caption_precision": "figure_captions",
    "figure_caption_recall": "figure_captions",
    "figure_caption_f1": "figure_captions",
    "table_caption_precision": "table_captions",
    "table_caption_recall": "table_captions",
    "table_caption_f1": "table_captions",
    "reference_precision": "reference_titles",
    "reference_recall": "reference_titles",
    "reference_f1": "reference_titles",
    "reference_combined_precision": "reference_records",
    "reference_recall_combined": "reference_records",
    "reference_combined_recall": "reference_records",
    "reference_combined_f1": "reference_records",
}


def word_diff(reference: Optional[str], candidate: Optional[str]) -> str:
    if reference is None and candidate is None:
        return "(both None)"
    if reference is None:
        return f"(reference=None) {_GREEN}{candidate}{_RESET}"
    if candidate is None:
        return f"{_RED}(candidate=None){_RESET} reference={reference}"
    ref_words = reference.split()
    cand_words = candidate.split()
    matcher = difflib.SequenceMatcher(None, ref_words, cand_words, autojunk=False)
    parts = []
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            parts.append(" ".join(ref_words[i1:i2]))
        elif op == "replace":
            parts.append(f"{_RED}[-{' '.join(ref_words[i1:i2])}-]{_RESET}")
            parts.append(f"{_GREEN}{{+{' '.join(cand_words[j1:j2])}+}}{_RESET}")
        elif op == "delete":
            parts.append(f"{_RED}[-{' '.join(ref_words[i1:i2])}-]{_RESET}")
        elif op == "insert":
            parts.append(f"{_GREEN}{{+{' '.join(cand_words[j1:j2])}+}}{_RESET}")
    return " ".join(parts)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _find_jsonl(run: Path) -> Path:
    if run.is_file():
        return run
    candidate = run / "per_document.jsonl"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"per_document.jsonl not found in {run}")


def _pred_key(metrics_key: str) -> str:
    return metrics_key.removesuffix("_metrics") + "_pred"


def _label(pred_key: str) -> str:
    return pred_key.removesuffix("_pred")


def _print_record(
    r: Dict[str, Any],
    metric: str,
    field: str,
    system_a: str,
    system_b: str,
    system_a_pred: str,
    system_b_pred: str,
) -> None:
    score_a = (r.get(system_a) or {}).get(metric)
    score_b = (r.get(system_b) or {}).get(metric)
    gold_val = (r.get("gold") or {}).get(field)
    val_a = (r.get(system_a_pred) or {}).get(field)
    val_b = (r.get(system_b_pred) or {}).get(field)

    label_b = _label(system_b_pred)
    label_a = _label(system_a_pred)
    width = max(len(label_b), len(label_a), len("gold"))
    score_b_str = f"{score_b:.3f}" if score_b is not None else "n/a"
    score_a_str = f"{score_a:.3f}" if score_a is not None else "n/a"

    print(f"record_id : {r['record_id']}")
    is_str = isinstance(gold_val or val_a or val_b, str)
    if is_str:
        if gold_val and len(str(gold_val).split()) < _SHORT_WORD_THRESHOLD:
            print(f"  {'gold':<{width}} : {gold_val}")
        print(f"  {label_b:<{width}} : {metric}={score_b_str} | {word_diff(gold_val, val_b)}")
        print(f"  {label_a:<{width}} : {metric}={score_a_str} | {word_diff(gold_val, val_a)}")
    else:
        print(f"  {label_b:<{width}} : {metric}={score_b_str} | {gold_val!r} -> {val_b!r}")
        print(f"  {label_a:<{width}} : {metric}={score_a_str} | {gold_val!r} -> {val_a!r}")
    print()


def _export_record(
    r: Dict[str, Any],
    run_dir: Path,
    metric: str,
    field: str,
    mode: str,
    system_a_pred: str,
    system_b_pred: str,
    parser: str = "grobid",
) -> None:
    record_id = r["record_id"]
    corpus = r["corpus"]
    data_dir = run_dir / "data" / corpus
    out_dir = run_dir / "examples" / mode / metric / corpus
    out_dir.mkdir(parents=True, exist_ok=True)

    for ext in ("pdf", "xml"):
        src = data_dir / f"{record_id}.{ext}"
        if src.exists():
            shutil.copy(src, out_dir / f"{record_id}.{ext}")

    tei_src = run_dir / corpus / "tei" / parser / f"{record_id}.tei.xml"
    if tei_src.exists():
        shutil.copy(tei_src, out_dir / f"{record_id}.tei.xml")

    gold_val = (r.get("gold") or {}).get(field)
    val_b = (r.get(system_b_pred) or {}).get(field)
    val_a = (r.get(system_a_pred) or {}).get(field)
    label_b = _label(system_b_pred)
    label_a = _label(system_a_pred)

    is_str = isinstance(gold_val or val_a or val_b, str)
    suffix = "txt" if is_str else "json"
    for label, val in [("gold", gold_val), (label_b, val_b), (label_a, val_a)]:
        content = (
            str(val or "") if is_str else json.dumps(val, ensure_ascii=False, indent=2)
        )
        (out_dir / f"{record_id}.{label}.{field}.{suffix}").write_text(
            content, encoding="utf-8"
        )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Find regression or improvement cases in benchmark results."
    )
    ap.add_argument(
        "--run", required=True, type=Path,
        help="Run directory (or path to per_document.jsonl directly)",
    )
    ap.add_argument("--metric", required=True, help="Metric key, e.g. abstract_edit_sim")
    ap.add_argument(
        "--field",
        help="Field name in gold/pred dicts (auto-derived from --metric if omitted)",
    )
    ap.add_argument("--corpus", help="Filter to one corpus (default: all)")
    ap.add_argument("--mode", required=True, choices=["regression", "improvement"])
    ap.add_argument(
        "--system-a", default="llm_metrics", dest="system_a",
        help="Metrics key for the system under evaluation (default: llm_metrics)",
    )
    ap.add_argument(
        "--system-b", default="grobid_metrics", dest="system_b",
        help="Metrics key for the baseline (default: grobid_metrics)",
    )
    ap.add_argument(
        "--system-a-pred", default=None, dest="system_a_pred",
        help="Prediction key for system-a (default: derived from --system-a)",
    )
    ap.add_argument(
        "--system-b-pred", default=None, dest="system_b_pred",
        help="Prediction key for system-b (default: derived from --system-b)",
    )
    ap.add_argument(
        "--parser", default="grobid",
        help="Parser name used to locate TEI files (default: grobid)",
    )
    ap.add_argument(
        "--limit", type=int,
        help="Max records to print to console (all cases are always exported to files)",
    )
    args = ap.parse_args()

    field = args.field or _METRIC_TO_FIELD.get(args.metric)
    if not field:
        ap.error(f"Cannot auto-derive field for '{args.metric}'. Use --field.")

    system_a_pred: str = args.system_a_pred or _pred_key(args.system_a)
    system_b_pred: str = args.system_b_pred or _pred_key(args.system_b)

    jsonl = _find_jsonl(args.run)
    run_dir = jsonl.parent if jsonl.name == "per_document.jsonl" else args.run

    all_rows = _load_jsonl(jsonl)
    rows = [r for r in all_rows if not args.corpus or r.get("corpus") == args.corpus]
    corpus_label = args.corpus or "all corpora"
    print(f"Loaded {len(rows)} records ({corpus_label})")

    def _is_case(r: Dict[str, Any]) -> bool:
        a = (r.get(args.system_a) or {}).get(args.metric)
        b = (r.get(args.system_b) or {}).get(args.metric)
        if a is None or b is None:
            return False
        return bool(a < b) if args.mode == "regression" else bool(a > b)

    cases = [r for r in rows if _is_case(r)]
    label_a = _label(system_a_pred)
    print(f"{len(cases)} documents where {label_a} {args.mode}d {args.metric}\n")

    to_print = cases if args.limit is None else cases[: args.limit]
    for r in to_print:
        _print_record(r, args.metric, field, args.system_a, args.system_b, system_a_pred, system_b_pred)

    for r in cases:
        _export_record(r, run_dir, args.metric, field, args.mode, system_a_pred, system_b_pred, args.parser)

    corpora = sorted({r["corpus"] for r in cases})
    if corpora:
        if len(corpora) == 1:
            out_dir = run_dir / "examples" / args.mode / args.metric / corpora[0]
        else:
            out_dir = run_dir / "examples" / args.mode / args.metric
        print(f"Exported {len(cases)} examples to {out_dir}")


if __name__ == "__main__":
    main()
