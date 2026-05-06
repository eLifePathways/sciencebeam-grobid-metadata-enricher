from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from benchmarks.score import render_markdown, score

METRICS = ["title_match", "authors_recall"]
CFG_N_RESAMPLES = 200
CFG_CL = 0.95
CONFIG_PATHS = [
    Path("benchmarks/bench.yaml"),
    Path("benchmarks/bench-train.yaml"),
]
F1_PRIMARY_METRICS = {
    "abstract_f1",
    "keywords_f1",
    "identifiers_f1",
    "body_section_f1",
    "figure_caption_f1",
    "table_caption_f1",
    "reference_f1",
    "reference_combined_f1",
}
RECALL_PRIMARY_REPLACEMENTS = {
    "abstract_recall",
    "keywords_recall",
    "identifiers_recall",
    "body_section_recall",
    "figure_caption_recall",
    "table_caption_recall",
    "reference_recall",
    "reference_recall_combined",
}


def _row(corpus: str, rid: str, grobid: tuple, llm: tuple) -> dict:
    return {
        "corpus": corpus, "record_id": rid,
        "grobid_metrics": {"title_match": grobid[0], "authors_recall": grobid[1]},
        "llm_metrics":    {"title_match": llm[0],    "authors_recall": llm[1]},
    }


def test_benchmark_configs_use_f1_for_over_extraction_sensitive_metrics() -> None:
    for path in CONFIG_PATHS:
        metrics = set(yaml.safe_load(path.read_text(encoding="utf-8"))["metrics"])
        assert F1_PRIMARY_METRICS <= metrics
        assert not RECALL_PRIMARY_REPLACEMENTS & metrics


def test_identical_runs_wilcoxon_p_is_one() -> None:
    rows = [_row("scielo_preprints", f"r{i}", (0.5, 0.7), (0.5, 0.7)) for i in range(20)]
    result = score(rows, METRICS, CFG_N_RESAMPLES, CFG_CL, baseline_rows=rows)
    overall = result["overall"]["metrics"]["title_match"]
    assert overall["grobid"]["mean"] == 0.5
    assert overall["llm"]["mean"] == 0.5
    assert overall["delta_llm_minus_grobid"] == 0.0
    # llm == grobid everywhere: paired Wilcoxon short-circuits to p=1.0
    assert overall["wilcoxon_p_llm_vs_grobid"] == 1.0
    # Baseline identical: delta vs baseline is zero and p==1.0
    assert overall["vs_baseline"]["delta_mean"] == 0.0
    assert overall["vs_baseline"]["wilcoxon_p"] == 1.0


def test_llm_beats_grobid_has_small_p() -> None:
    rng = np.random.default_rng(0)
    rows = []
    for i in range(30):
        grobid = rng.uniform(0.30, 0.50)
        llm = grobid + rng.uniform(0.10, 0.20)
        rows.append(_row("scielo_preprints", f"r{i}", (grobid, 0.5), (llm, 0.7)))
    result = score(rows, METRICS, CFG_N_RESAMPLES, CFG_CL)
    e = result["overall"]["metrics"]["title_match"]
    assert e["delta_llm_minus_grobid"] > 0.05
    assert e["wilcoxon_p_llm_vs_grobid"] is not None
    assert e["wilcoxon_p_llm_vs_grobid"] < 0.01


def test_bootstrap_reproducibility_same_seed() -> None:
    # Same inputs + same seed: same CI bounds.
    rows = [_row("scielo_preprints", f"r{i}", (0.4, 0.5), (0.6, 0.7)) for i in range(25)]
    a = score(rows, METRICS, CFG_N_RESAMPLES, CFG_CL)
    b = score(rows, METRICS, CFG_N_RESAMPLES, CFG_CL)
    for m in METRICS:
        for side in ("grobid", "llm"):
            assert a["overall"]["metrics"][m][side] == b["overall"]["metrics"][m][side]


def test_per_corpus_sections_and_overall_n() -> None:
    rows = [
        _row("scielo_preprints", "a", (0.5, 0.5), (0.7, 0.6)),
        _row("scielo_preprints", "b", (0.4, 0.5), (0.6, 0.7)),
        _row("other",            "c", (0.3, 0.3), (0.4, 0.5)),
    ]
    result = score(rows, METRICS, CFG_N_RESAMPLES, CFG_CL)
    assert result["overall"]["n"] == 3
    assert result["scielo_preprints"]["n"] == 2
    assert result["other"]["n"] == 1


def test_render_markdown_contains_ci_and_wilcoxon_columns() -> None:
    rows = [_row("scielo_preprints", f"r{i}", (0.4, 0.5), (0.6, 0.7)) for i in range(10)]
    result = score(rows, METRICS, CFG_N_RESAMPLES, CFG_CL, baseline_rows=rows)
    md = render_markdown(result, METRICS)
    assert "Grobid (95% CI)" in md
    assert "Wilcoxon p" in md
    assert "Δ vs baseline" in md
    # Per-corpus heading rendered
    assert "scielo_preprints" in md
