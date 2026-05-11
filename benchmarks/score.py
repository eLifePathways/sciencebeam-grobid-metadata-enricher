from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml
from scipy.stats import bootstrap, wilcoxon


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _values(rows: List[Dict[str, Any]], metrics_key: str, metric: str) -> np.ndarray:
    vals = [r[metrics_key].get(metric) for r in rows if r.get(metrics_key, {}).get(metric) is not None]
    return np.asarray(vals, dtype=float)


def _ci(vals: np.ndarray, n_resamples: int, confidence_level: float, seed: int = 0) -> Tuple[float, float, float]:
    if len(vals) == 0:
        return (float("nan"), float("nan"), float("nan"))
    if len(vals) == 1:
        v = float(vals[0])
        return (v, v, v)
    mean = float(np.mean(vals))
    res = bootstrap(
        (vals,), np.mean,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        method="percentile",
        random_state=np.random.default_rng(seed),
    )
    return (mean, float(res.confidence_interval.low), float(res.confidence_interval.high))


def _paired(vals_a: np.ndarray, vals_b: np.ndarray) -> Optional[float]:
    # Paired Wilcoxon. Requires same length and at least one non-zero difference.
    if len(vals_a) != len(vals_b) or len(vals_a) < 2:
        return None
    diffs = vals_a - vals_b
    if np.all(diffs == 0):
        return 1.0
    try:
        _, p = wilcoxon(vals_a, vals_b, zero_method="wilcox", alternative="two-sided")
        return float(p)
    except ValueError:
        return None


_TOKEN_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens", "reasoning_tokens")


def _empty_token_bucket() -> Dict[str, int]:
    return {field: 0 for field in _TOKEN_FIELDS} | {"n_calls": 0}


def _aggregate_tokens(
    rows: List[Dict[str, Any]],
    n_resamples: int,
    confidence_level: float,
) -> Dict[str, Any]:
    # Sum per-stage and per-metric-group token usage across rows, and report
    # per-document total means with 95% bootstrap CIs so run-to-run
    # comparisons are possible. Rows without a `tokens` key are tolerated
    # (legacy jsonl from older benchmarks): they are treated as zero usage.
    total_bucket = _empty_token_bucket()
    by_stage: Dict[str, Dict[str, int]] = {}
    by_group: Dict[str, Dict[str, int]] = {}
    per_doc_totals: Dict[str, List[float]] = {field: [] for field in _TOKEN_FIELDS}
    per_doc_n_calls: List[float] = []

    for r in rows:
        tok = r.get("tokens") or {}
        total = tok.get("total") or {}
        for field in _TOKEN_FIELDS:
            v = int(total.get(field, 0) or 0)
            total_bucket[field] += v
            per_doc_totals[field].append(float(v))
        n_calls = int(total.get("n_calls", 0) or 0)
        total_bucket["n_calls"] += n_calls
        per_doc_n_calls.append(float(n_calls))
        for stage, stage_tok in (tok.get("by_stage") or {}).items():
            bucket = by_stage.setdefault(stage, _empty_token_bucket())
            for field in _TOKEN_FIELDS:
                bucket[field] += int(stage_tok.get(field, 0) or 0)
            bucket["n_calls"] += int(stage_tok.get("n_calls", 0) or 0)
        for group, group_tok in (tok.get("by_metric_group") or {}).items():
            bucket = by_group.setdefault(group, _empty_token_bucket())
            for field in _TOKEN_FIELDS:
                bucket[field] += int(group_tok.get(field, 0) or 0)
            bucket["n_calls"] += int(group_tok.get("n_calls", 0) or 0)

    per_doc_mean_ci: Dict[str, Dict[str, float]] = {}
    for field in _TOKEN_FIELDS:
        arr = np.asarray(per_doc_totals[field], dtype=float)
        mean, lo, hi = _ci(arr, n_resamples, confidence_level)
        per_doc_mean_ci[field] = {"mean": mean, "ci_low": lo, "ci_high": hi}
    n_calls_arr = np.asarray(per_doc_n_calls, dtype=float)
    mean, lo, hi = _ci(n_calls_arr, n_resamples, confidence_level)
    per_doc_mean_ci["n_calls"] = {"mean": mean, "ci_low": lo, "ci_high": hi}

    return {
        "n": len(rows),
        "total": total_bucket,
        "per_doc_mean_ci": per_doc_mean_ci,
        "by_stage": by_stage,
        "by_metric_group": by_group,
    }


def score(
    rows: List[Dict[str, Any]],
    metrics: List[str],
    n_resamples: int,
    confidence_level: float,
    baseline_rows: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    # Paired Wilcoxon: llm_metrics vs grobid_metrics within this run, and if baseline_rows is
    # given, llm_metrics of this run vs the baseline run matched on (corpus, record_id).
    corpora = sorted({r["corpus"] for r in rows})
    sections: Dict[str, Any] = {}

    def _section(subset: List[Dict[str, Any]], baseline_subset: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
        out: Dict[str, Any] = {"n": len(subset), "metrics": {}}
        baseline_index = (
            {(b["corpus"], b["record_id"]): b for b in baseline_subset}
            if baseline_subset else {}
        )
        for metric in metrics:
            grobid_vals = _values(subset, "grobid_metrics", metric)
            llm_vals = _values(subset, "llm_metrics", metric)
            delta = (
                float(np.mean(llm_vals) - np.mean(grobid_vals))
                if len(llm_vals) and len(grobid_vals) else None
            )
            wilcoxon_p = (
                _paired(llm_vals, grobid_vals)
                if len(llm_vals) == len(grobid_vals) else None
            )
            entry = {
                "grobid": dict(zip(("mean", "ci_low", "ci_high"), _ci(grobid_vals, n_resamples, confidence_level))),
                "llm": dict(zip(("mean", "ci_low", "ci_high"), _ci(llm_vals, n_resamples, confidence_level))),
                "delta_llm_minus_grobid": delta,
                "wilcoxon_p_llm_vs_grobid": wilcoxon_p,
            }
            if baseline_index:
                paired_cur, paired_base = [], []
                for r in subset:
                    key = (r["corpus"], r["record_id"])
                    if key in baseline_index:
                        c = r.get("llm_metrics", {}).get(metric)
                        b = baseline_index[key].get("llm_metrics", {}).get(metric)
                        if c is not None and b is not None:
                            paired_cur.append(c)
                            paired_base.append(b)
                if paired_cur:
                    arr_c = np.asarray(paired_cur, dtype=float)
                    arr_b = np.asarray(paired_base, dtype=float)
                    entry["vs_baseline"] = {
                        "n_paired": len(arr_c),
                        "delta_mean": float(np.mean(arr_c - arr_b)),
                        "wilcoxon_p": _paired(arr_c, arr_b),  # type: ignore[dict-item]
                    }
            out["metrics"][metric] = entry
        return out

    sections["overall"] = _section(rows, baseline_rows)
    for c in corpora:
        subset = [r for r in rows if r["corpus"] == c]
        base = [b for b in (baseline_rows or []) if b["corpus"] == c] or None
        sections[c] = _section(subset, base)

    tokens_section: Dict[str, Any] = {
        "overall": _aggregate_tokens(rows, n_resamples, confidence_level),
    }
    for c in corpora:
        subset = [r for r in rows if r["corpus"] == c]
        tokens_section[c] = _aggregate_tokens(subset, n_resamples, confidence_level)
    sections["tokens"] = tokens_section
    return sections


_METRIC_TO_TOKEN_GROUP: Dict[str, str] = {
    "title_match": "header",
    "title_edit_sim": "header",
    "authors_precision": "header",
    "authors_recall": "header",
    "authors_f1": "header",
    "abstract_precision": "abstract",
    "abstract_recall": "abstract",
    "abstract_f1": "abstract",
    "abstract_edit_sim": "abstract",
    "keywords_precision": "keywords",
    "keywords_recall": "keywords",
    "keywords_f1": "keywords",
    "identifiers_precision": "header",
    "identifiers_recall": "header",
    "identifiers_f1": "header",
    "language_match": "header",
    "body_section_precision": "content",
    "body_section_recall": "content",
    "body_section_f1": "content",
    "figure_caption_precision": "content",
    "figure_caption_recall": "content",
    "figure_caption_f1": "content",
    "table_caption_precision": "content",
    "table_caption_recall": "content",
    "table_caption_f1": "content",
    "reference_precision": "content",
    "reference_recall": "content",
    "reference_f1": "content",
    "reference_combined_precision": "content",
    "reference_recall_combined": "content",
    "reference_combined_recall": "content",
    "reference_combined_f1": "content",
}


def _format_kilo(n: int) -> str:
    # Compact rendering for header summaries: 1234 -> "1.2k", 999 -> "999".
    if n >= 1000:
        return f"{n / 1000:.1f}k"
    return str(int(n))


def _section_tokens_summary(tokens_for_section: Optional[Dict[str, Any]]) -> str:
    # Empty, compact summary when no token data is present so legacy runs render unchanged.
    if not tokens_for_section:
        return ""
    total = tokens_for_section.get("total") or {}
    p = int(total.get("prompt_tokens", 0) or 0)
    c = int(total.get("completion_tokens", 0) or 0)
    t = int(total.get("total_tokens", 0) or 0)
    n_calls = int(total.get("n_calls", 0) or 0)
    if p == 0 and c == 0 and t == 0 and n_calls == 0:
        return ""
    per_doc = tokens_for_section.get("per_doc_mean_ci") or {}
    t_mean = (per_doc.get("total_tokens") or {}).get("mean")
    calls_mean = (per_doc.get("n_calls") or {}).get("mean")
    per_doc_bits: List[str] = []
    if isinstance(t_mean, (int, float)):
        per_doc_bits.append(f"{_format_kilo(int(round(t_mean)))} total/doc")
    if isinstance(calls_mean, (int, float)):
        per_doc_bits.append(f"{calls_mean:.1f} calls/doc")
    tail = " · ".join(per_doc_bits)
    suffix = f" · {tail}" if tail else ""
    return (
        f"LLM tokens: {_format_kilo(p)} prompt / {_format_kilo(c)} completion / "
        f"{_format_kilo(t)} total, {n_calls} calls{suffix}"
    )


def _format_llm_client(client: Dict[str, Any]) -> str:
    """One-line summary of the LLM client used for the run, for the report
    header. Handles both OpenAIClient (single model) and AoaiPool (one or more
    Azure deployments) descriptors produced by predict._describe_client."""
    provider = client.get("provider") or "unknown"
    if provider == "openai":
        model = client.get("model") or "unknown"
        base_url = client.get("base_url")
        return f"{model} (openai, base: {base_url})" if base_url else f"{model} (openai)"
    if provider == "azure_openai":
        deployments = client.get("deployments") or []
        label = ", ".join(deployments) if deployments else "unknown"
        n = client.get("n_backends")
        suffix = f", {n} backends" if n else ""
        return f"{label} (azure_openai{suffix})"
    return f"{provider}"


def _render_run_info_markdown(run_record: Dict[str, Any]) -> List[str]:
    """Render a short metadata block at the top of the report so readers know
    which LLM, parser, and dataset slice produced the numbers below. Omitted
    when no run_record is available (e.g. legacy runs scored before this was
    added) so historical reports can still be regenerated."""
    llm = run_record.get("llm") or {}
    client = llm.get("client") or {}
    parts: List[str] = []
    if client:
        parts.append(f"- **LLM model:** {_format_llm_client(client)}")
    parser = run_record.get("parser")
    if parser:
        parts.append(f"- **Parser:** {parser}")
    n_records = run_record.get("n_records")
    n_errors = run_record.get("n_errors")
    elapsed = run_record.get("elapsed_s")
    if n_records is not None:
        bits = [f"{n_records} records"]
        if n_errors is not None:
            bits.append(f"{n_errors} errors")
        if elapsed is not None:
            bits.append(f"{elapsed}s")
        parts.append("- **Run:** " + " · ".join(bits))
    commit = run_record.get("git_commit")
    if commit and commit != "unknown":
        parts.append(f"- **Commit:** {commit}")
    if not parts:
        return []
    return parts + [""]


def render_markdown(
    result: Dict[str, Any],
    metrics: List[str],
    title: str = "Benchmark report",
    run_record: Optional[Dict[str, Any]] = None,
) -> str:
    tokens_by_section = result.get("tokens") or {}
    lines = [f"# {title}", ""]
    if run_record:
        lines.extend(_render_run_info_markdown(run_record))
    for section_name, section in result.items():
        if section_name == "tokens":
            continue  # rendered separately below so the metric tables stay unchanged
        section_tokens = tokens_by_section.get(section_name) if tokens_by_section else None
        token_summary = _section_tokens_summary(section_tokens)
        header_line = f"## {section_name} (N={section['n']})"
        if token_summary:
            header_line += f" — {token_summary}"
        lines.append(header_line)
        lines.append("")
        has_baseline = any("vs_baseline" in section["metrics"][m] for m in metrics)
        # Attach a per-metric tokens column (summing by the metric's stage-group) when
        # token data is available for this section. Unknown metric mappings render "—".
        has_tokens = bool(section_tokens and section_tokens.get("by_metric_group"))
        header = ["Metric", "Grobid (95% CI)", "LLM (95% CI)", "Δ LLM−Grobid", "Wilcoxon p"]
        if has_baseline:
            header += ["Δ vs baseline", "p vs baseline"]
        if has_tokens:
            header += ["Prompt tok", "Completion tok"]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] * len(header)) + "|")
        for m in metrics:
            e = section["metrics"][m]
            g, llm = e["grobid"], e["llm"]
            row = [
                m,
                f"{g['mean']:.3f} [{g['ci_low']:.3f}, {g['ci_high']:.3f}]",
                f"{llm['mean']:.3f} [{llm['ci_low']:.3f}, {llm['ci_high']:.3f}]",
                f"{e['delta_llm_minus_grobid']:+.3f}" if e['delta_llm_minus_grobid'] is not None else "n/a",
                f"{e['wilcoxon_p_llm_vs_grobid']:.3g}" if e['wilcoxon_p_llm_vs_grobid'] is not None else "n/a",
            ]
            if has_baseline:
                vb = e.get("vs_baseline")
                if vb:
                    row.append(f"{vb['delta_mean']:+.3f}")
                    row.append(f"{vb['wilcoxon_p']:.3g}" if vb['wilcoxon_p'] is not None else "n/a")
                else:
                    row += ["n/a", "n/a"]
            if has_tokens:
                group = _METRIC_TO_TOKEN_GROUP.get(m)
                by_group = (section_tokens or {}).get("by_metric_group") or {}
                bucket = by_group.get(group) if group else None
                if bucket:
                    row.append(_format_kilo(int(bucket.get("prompt_tokens", 0) or 0)))
                    row.append(_format_kilo(int(bucket.get("completion_tokens", 0) or 0)))
                else:
                    row += ["—", "—"]
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    tokens = result.get("tokens")
    if tokens:
        lines.extend(_render_tokens_markdown(tokens))
    return "\n".join(lines)


def _render_tokens_markdown(tokens: Dict[str, Any]) -> List[str]:
    lines: List[str] = ["## Tokens", ""]
    overall = tokens.get("overall") or {}
    total = overall.get("total") or {}
    per_doc = overall.get("per_doc_mean_ci") or {}
    lines.append(f"Overall (N={overall.get('n', 0)}):")
    lines.append("")
    lines.append("| Field | Sum | Per-doc mean (95% CI) |")
    lines.append("|---|---|---|")
    for field in _TOKEN_FIELDS:
        mc = per_doc.get(field) or {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
        lines.append(
            f"| {field} | {int(total.get(field, 0))} "
            f"| {mc['mean']:.1f} [{mc['ci_low']:.1f}, {mc['ci_high']:.1f}] |"
        )
    nc = per_doc.get("n_calls") or {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    lines.append(
        f"| n_calls | {int(total.get('n_calls', 0))} "
        f"| {nc['mean']:.1f} [{nc['ci_low']:.1f}, {nc['ci_high']:.1f}] |"
    )
    lines.append("")

    by_stage = overall.get("by_stage") or {}
    if by_stage:
        lines.append("### Per stage (overall sums)")
        lines.append("")
        lines.append("| Stage | Prompt | Completion | Total | Cached | Reasoning | n_calls |")
        lines.append("|---|---|---|---|---|---|---|")
        for stage in sorted(by_stage):
            b = by_stage[stage]
            lines.append(
                f"| {stage} | {int(b.get('prompt_tokens', 0))} | {int(b.get('completion_tokens', 0))} "
                f"| {int(b.get('total_tokens', 0))} | {int(b.get('cached_tokens', 0))} "
                f"| {int(b.get('reasoning_tokens', 0))} | {int(b.get('n_calls', 0))} |"
            )
        lines.append("")

    by_group = overall.get("by_metric_group") or {}
    if by_group:
        lines.append("### Per metric group (overall sums)")
        lines.append("")
        lines.append("| Group | Prompt | Completion | Total | Cached | Reasoning | n_calls |")
        lines.append("|---|---|---|---|---|---|---|")
        for group in sorted(by_group):
            b = by_group[group]
            lines.append(
                f"| {group} | {int(b.get('prompt_tokens', 0))} | {int(b.get('completion_tokens', 0))} "
                f"| {int(b.get('total_tokens', 0))} | {int(b.get('cached_tokens', 0))} "
                f"| {int(b.get('reasoning_tokens', 0))} | {int(b.get('n_calls', 0))} |"
            )
        lines.append("")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path, help="Current run directory (expects per_document.jsonl)")
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--baseline", type=Path, help="Baseline run directory for paired comparison")
    ap.add_argument("--out", type=Path, help="Markdown report output path (defaults to <run>/report.md)")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    rows = _load_jsonl(args.run / "per_document.jsonl")
    baseline_rows = _load_jsonl(args.baseline / "per_document.jsonl") if args.baseline else None

    result = score(
        rows, cfg["metrics"],
        n_resamples=cfg["bootstrap"]["n_resamples"],
        confidence_level=cfg["bootstrap"]["confidence_level"],
        baseline_rows=baseline_rows,
    )

    (args.run / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    run_record_path = args.run / "run_record.json"
    run_record = (
        json.loads(run_record_path.read_text(encoding="utf-8"))
        if run_record_path.exists()
        else None
    )
    md = render_markdown(
        result,
        cfg["metrics"],
        title=f"Benchmark report: {args.run.name}",
        run_record=run_record,
    )
    out = args.out or (args.run / "report.md")
    out.write_text(md, encoding="utf-8")
    print(f"Wrote {args.run / 'metrics.json'} and {out}", flush=True)


if __name__ == "__main__":
    main()
