"""Side-by-side scoring across N parallel LLM runs (same Grobid baseline).

Usage:
    python -m benchmarks.score_multi \\
        --config benchmarks/bench.yaml \\
        --run base=runs/123/qwen_base \\
        --run lora=runs/123/qwen_lora \\
        --run azure=runs/123/azure \\
        --out runs/123/report.md

Each `--run label=path` is a directory produced by `benchmarks.predict`. The
Grobid metrics are taken from the first run (they match across runs for the
same record). The LLM metrics column is rendered once per labelled run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

from benchmarks.score import (
    _aggregate_tokens,
    _ci,
    _load_jsonl,
    _paired,
    _values,
)


def _parse_run_spec(spec: str) -> Tuple[str, Path]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError(
            f"--run expects label=path, got {spec!r}"
        )
    label, raw = spec.split("=", 1)
    label = label.strip()
    path = Path(raw.strip())
    if not label:
        raise argparse.ArgumentTypeError(f"--run label is empty in {spec!r}")
    return label, path


def _section_for_label(
    rows: List[Dict[str, Any]],
    metrics: List[str],
    n_resamples: int,
    confidence_level: float,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {"n": len(rows), "metrics": {}}
    for metric in metrics:
        grobid_vals = _values(rows, "grobid_metrics", metric)
        llm_vals = _values(rows, "llm_metrics", metric)
        delta = (
            float(np.mean(llm_vals) - np.mean(grobid_vals))
            if len(llm_vals) and len(grobid_vals) else None
        )
        wil = (
            _paired(llm_vals, grobid_vals)
            if len(llm_vals) == len(grobid_vals) else None
        )
        out["metrics"][metric] = {
            "grobid": dict(zip(("mean", "ci_low", "ci_high"),
                               _ci(grobid_vals, n_resamples, confidence_level))),
            "llm": dict(zip(("mean", "ci_low", "ci_high"),
                            _ci(llm_vals, n_resamples, confidence_level))),
            "delta_llm_minus_grobid": delta,
            "wilcoxon_p_llm_vs_grobid": wil,
        }
    return out


def _format_cell(d: Dict[str, float]) -> str:
    return f"{d['mean']:.3f} [{d['ci_low']:.3f}, {d['ci_high']:.3f}]"


def _bold_winner(cells: List[str], scores: List[float]) -> List[str]:
    # Bold the cell with the highest mean. NaN scores are ignored.
    finite = [(i, s) for i, s in enumerate(scores) if not (s is None or np.isnan(s))]
    if not finite:
        return cells
    winner = max(finite, key=lambda x: x[1])[0]
    return [
        f"**{c}**" if i == winner else c
        for i, c in enumerate(cells)
    ]


def _backend_description(record: Optional[Dict[str, Any]]) -> str:
    """One-line summary of an LLM backend: model + LoRA pack + parser image."""
    if not record:
        return "(unknown)"
    llm = record.get("llm") or {}
    model = llm.get("model") or "?"
    parts = [f"model=`{model}`"]
    lora_map = llm.get("step_lora_map") or {}
    if lora_map:
        adapters = sorted(set(lora_map.values()))
        parts.append(f"LoRA adapters=[{', '.join(adapters)}]")
    parser_image = record.get("parser_image")
    if parser_image:
        parts.append(f"parser=`{parser_image}`")
    return " · ".join(parts)


def _render(
    labels: List[str],
    sections: Dict[str, Dict[str, Dict[str, Any]]],
    tokens_by_label: Dict[str, Dict[str, Any]],
    records: Dict[str, Optional[Dict[str, Any]]],
    metrics: List[str],
    title: str,
) -> str:
    # sections: section_name -> label -> _section_for_label() result
    lines: List[str] = [f"# {title}", ""]
    if records:
        lines.append("**Backends:**")
        lines.append("")
        for label in labels:
            lines.append(f"- `{label}` — {_backend_description(records.get(label))}")
        lines.append("")
    section_names = ["overall"] + [s for s in sorted(sections) if s != "overall"]

    for section_name in section_names:
        per_label = sections[section_name]
        # Use the first label's N (they all share documents).
        n = per_label[labels[0]]["n"]
        lines.append("<details" + (" open" if section_name == "overall" else "") + ">")
        lines.append(f"<summary><b>{section_name} (N={n})</b></summary>")
        lines.append("")
        header = ["Metric", "Grobid"] + labels + [f"Δ({lbl}−Grobid)" for lbl in labels]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] * len(header)) + "|")
        for metric in metrics:
            grobid = per_label[labels[0]]["metrics"][metric]["grobid"]
            llm_cells: List[str] = []
            llm_means: List[float] = []
            delta_cells: List[str] = []
            for label in labels:
                entry = per_label[label]["metrics"][metric]
                llm_cells.append(_format_cell(entry["llm"]))
                llm_means.append(entry["llm"]["mean"])
                delta = entry["delta_llm_minus_grobid"]
                delta_cells.append(f"{delta:+.3f}" if delta is not None else "n/a")
            llm_cells_bold = _bold_winner(llm_cells, llm_means)
            row = [metric, _format_cell(grobid)] + llm_cells_bold + delta_cells
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
        lines.append("</details>")
        lines.append("")

    # Per-backend token totals.
    if tokens_by_label:
        lines.append("<details>")
        lines.append("<summary><b>Tokens by backend</b></summary>")
        lines.append("")
        lines.append("| Backend | Prompt | Completion | Total | n_calls |")
        lines.append("|---|---|---|---|---|")
        for label in labels:
            tot = (tokens_by_label.get(label) or {}).get("overall", {}).get("total", {}) or {}
            lines.append(
                f"| {label} | {int(tot.get('prompt_tokens', 0))} "
                f"| {int(tot.get('completion_tokens', 0))} "
                f"| {int(tot.get('total_tokens', 0))} "
                f"| {int(tot.get('n_calls', 0))} |"
            )
        lines.append("")
        lines.append("</details>")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument(
        "--run", action="append", required=True, type=_parse_run_spec,
        metavar="label=path",
        help="Repeatable. Label and path of a predict run dir.",
    )
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument(
        "--title", default="Benchmark report — side-by-side",
        help="Report title.",
    )
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    metrics: List[str] = cfg["metrics"]
    n_res = cfg["bootstrap"]["n_resamples"]
    cl = cfg["bootstrap"]["confidence_level"]

    labels = [lbl for lbl, _ in args.run]
    if len(set(labels)) != len(labels):
        ap.error(f"Duplicate --run labels: {labels}")

    sections: Dict[str, Dict[str, Dict[str, Any]]] = {}
    tokens_by_label: Dict[str, Dict[str, Any]] = {}
    records: Dict[str, Optional[Dict[str, Any]]] = {}
    for label, run_dir in args.run:
        rows = _load_jsonl(run_dir / "per_document.jsonl")
        if not rows:
            raise SystemExit(f"No rows in {run_dir/'per_document.jsonl'}")
        rec_path = run_dir / "run_record.json"
        records[label] = (
            json.loads(rec_path.read_text(encoding="utf-8")) if rec_path.exists() else None
        )
        corpora = sorted({r["corpus"] for r in rows})
        # Overall + per-corpus.
        sections.setdefault("overall", {})[label] = _section_for_label(
            rows, metrics, n_res, cl,
        )
        for corpus in corpora:
            subset = [r for r in rows if r["corpus"] == corpus]
            sections.setdefault(corpus, {})[label] = _section_for_label(
                subset, metrics, n_res, cl,
            )
        tokens_by_label[label] = {"overall": _aggregate_tokens(rows, n_res, cl)}

    md = _render(labels, sections, tokens_by_label, records, metrics, args.title)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(md, encoding="utf-8")
    print(f"Wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
