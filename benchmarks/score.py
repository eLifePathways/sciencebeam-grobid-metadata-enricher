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
                            paired_cur.append(c); paired_base.append(b)
                if paired_cur:
                    arr_c = np.asarray(paired_cur, dtype=float)
                    arr_b = np.asarray(paired_base, dtype=float)
                    entry["vs_baseline"] = {
                        "n_paired": len(arr_c),
                        "delta_mean": float(np.mean(arr_c - arr_b)),
                        "wilcoxon_p": _paired(arr_c, arr_b),
                    }
            out["metrics"][metric] = entry
        return out

    sections["overall"] = _section(rows, baseline_rows)
    for c in corpora:
        subset = [r for r in rows if r["corpus"] == c]
        base = [b for b in (baseline_rows or []) if b["corpus"] == c] or None
        sections[c] = _section(subset, base)
    return sections


def render_markdown(result: Dict[str, Any], metrics: List[str], title: str = "Benchmark report") -> str:
    lines = [f"# {title}", ""]
    for section_name, section in result.items():
        lines.append(f"## {section_name} (N={section['n']})")
        lines.append("")
        has_baseline = any("vs_baseline" in section["metrics"][m] for m in metrics)
        header = ["Metric", "Grobid (95% CI)", "LLM (95% CI)", "Δ LLM−Grobid", "Wilcoxon p"]
        if has_baseline:
            header += ["Δ vs baseline", "p vs baseline"]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] * len(header)) + "|")
        for m in metrics:
            e = section["metrics"][m]
            g, l = e["grobid"], e["llm"]
            row = [
                m,
                f"{g['mean']:.3f} [{g['ci_low']:.3f}, {g['ci_high']:.3f}]",
                f"{l['mean']:.3f} [{l['ci_low']:.3f}, {l['ci_high']:.3f}]",
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
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    return "\n".join(lines)


def main():
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
    md = render_markdown(result, cfg["metrics"], title=f"Benchmark report: {args.run.name}")
    out = args.out or (args.run / "report.md")
    out.write_text(md, encoding="utf-8")
    print(f"Wrote {args.run / 'metrics.json'} and {out}", flush=True)


if __name__ == "__main__":
    main()
