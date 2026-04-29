from __future__ import annotations

from benchmarks.score import _aggregate_tokens, render_markdown, score

METRICS = ["title_match", "authors_recall"]
CFG_N_RESAMPLES = 50
CFG_CL = 0.95


def _row(corpus: str, rid: str, tokens: dict | None = None) -> dict:
    row = {
        "corpus": corpus,
        "record_id": rid,
        "grobid_metrics": {"title_match": 0.5, "authors_recall": 0.5},
        "llm_metrics": {"title_match": 0.6, "authors_recall": 0.7},
    }
    if tokens is not None:
        row["tokens"] = tokens
    return row


def _tok(
    prompt: int,
    completion: int,
    by_stage: dict | None = None,
    by_group: dict | None = None,
    cached: int = 0,
    reasoning: int = 0,
    n_calls: int = 1,
) -> dict:
    return {
        "total": {
            "prompt_tokens": prompt, "completion_tokens": completion,
            "total_tokens": prompt + completion,
            "cached_tokens": cached, "reasoning_tokens": reasoning,
            "n_calls": n_calls,
        },
        "by_stage": by_stage or {},
        "by_metric_group": by_group or {},
    }


class TestAggregateTokens:
    def test_sum_and_per_doc_mean(self) -> None:
        rows = [
            _row("scielo_preprints", "a", _tok(100, 50, n_calls=3)),
            _row("scielo_preprints", "b", _tok(200, 60, n_calls=4)),
        ]
        agg = _aggregate_tokens(rows, CFG_N_RESAMPLES, CFG_CL)

        assert agg["n"] == 2
        assert agg["total"]["prompt_tokens"] == 300
        assert agg["total"]["completion_tokens"] == 110
        assert agg["total"]["total_tokens"] == 410
        assert agg["total"]["n_calls"] == 7
        assert agg["per_doc_mean_ci"]["prompt_tokens"]["mean"] == 150.0

    def test_rows_without_tokens_are_zeroed(self) -> None:
        # Legacy per_document.jsonl rows without the tokens field must not crash aggregation.
        rows = [_row("scielo_preprints", "a"), _row("scielo_preprints", "b", _tok(10, 5, n_calls=1))]
        agg = _aggregate_tokens(rows, CFG_N_RESAMPLES, CFG_CL)
        assert agg["total"]["prompt_tokens"] == 10
        assert agg["total"]["n_calls"] == 1

    def test_by_stage_and_metric_group_roll_up(self) -> None:
        by_stage_a = {
            "HEADER_METADATA": {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70,
                                "cached_tokens": 0, "reasoning_tokens": 0, "n_calls": 1},
            "CONTENT_HEAD": {"prompt_tokens": 500, "completion_tokens": 200, "total_tokens": 700,
                             "cached_tokens": 0, "reasoning_tokens": 0, "n_calls": 1},
        }
        by_group_a = {
            "header": {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70,
                       "cached_tokens": 0, "reasoning_tokens": 0, "n_calls": 1},
            "content": {"prompt_tokens": 500, "completion_tokens": 200, "total_tokens": 700,
                        "cached_tokens": 0, "reasoning_tokens": 0, "n_calls": 1},
        }
        rows = [_row("scielo_preprints", "a", _tok(550, 220, by_stage_a, by_group_a, n_calls=2))]
        agg = _aggregate_tokens(rows, CFG_N_RESAMPLES, CFG_CL)
        assert agg["by_stage"]["HEADER_METADATA"]["prompt_tokens"] == 50
        assert agg["by_stage"]["CONTENT_HEAD"]["prompt_tokens"] == 500
        assert agg["by_metric_group"]["header"]["prompt_tokens"] == 50
        assert agg["by_metric_group"]["content"]["prompt_tokens"] == 500


class TestScoreIncludesTokensSection:
    def test_score_returns_tokens_key(self) -> None:
        rows = [_row("scielo_preprints", f"r{i}", _tok(100, 50, n_calls=2)) for i in range(5)]
        result = score(rows, METRICS, CFG_N_RESAMPLES, CFG_CL)
        assert "tokens" in result
        assert result["tokens"]["overall"]["total"]["prompt_tokens"] == 500

    def test_score_existing_keys_unchanged(self) -> None:
        # Regression safety: adding tokens doesn't remove or rename the metrics keys.
        rows = [_row("scielo_preprints", f"r{i}") for i in range(5)]
        result = score(rows, METRICS, CFG_N_RESAMPLES, CFG_CL)
        assert "overall" in result
        assert "scielo_preprints" in result
        assert "metrics" in result["overall"]
        assert "title_match" in result["overall"]["metrics"]


class TestRenderMarkdownTokensColumn:
    def test_section_heading_includes_token_summary(self) -> None:
        by_stage = {
            "HEADER_METADATA": {"prompt_tokens": 1500, "completion_tokens": 200, "total_tokens": 1700,
                                "cached_tokens": 0, "reasoning_tokens": 0, "n_calls": 3},
        }
        by_group = {
            "header": {"prompt_tokens": 1500, "completion_tokens": 200, "total_tokens": 1700,
                       "cached_tokens": 0, "reasoning_tokens": 0, "n_calls": 3},
        }
        rows = [_row("scielo_preprints", f"r{i}", _tok(500, 70, by_stage, by_group, n_calls=1)) for i in range(3)]
        result = score(rows, METRICS, CFG_N_RESAMPLES, CFG_CL)
        md = render_markdown(result, METRICS)
        # The per-corpus heading now carries an inline LLM token summary
        assert "LLM tokens:" in md
        assert "1.5k prompt" in md  # 1500 prompt tokens renders as 1.5k in the heading summary

    def test_metric_table_has_prompt_and_completion_columns(self) -> None:
        by_group = {
            "header": {"prompt_tokens": 1500, "completion_tokens": 200, "total_tokens": 1700,
                       "cached_tokens": 0, "reasoning_tokens": 0, "n_calls": 3},
        }
        rows = [_row("scielo_preprints", f"r{i}", _tok(500, 70, None, by_group, n_calls=1)) for i in range(3)]
        result = score(rows, METRICS, CFG_N_RESAMPLES, CFG_CL)
        md = render_markdown(result, METRICS)
        # New columns in the per-section metrics table
        assert "Prompt tok" in md
        assert "Completion tok" in md

    def test_no_token_columns_when_no_tokens(self) -> None:
        # Rows without any token field must render the legacy metrics table unchanged.
        rows = [_row("scielo_preprints", f"r{i}") for i in range(3)]
        result = score(rows, METRICS, CFG_N_RESAMPLES, CFG_CL)
        md = render_markdown(result, METRICS)
        assert "Prompt tok" not in md
        assert "LLM tokens:" not in md


class TestRenderMarkdownTokens:
    def test_tokens_section_rendered_when_present(self) -> None:
        by_stage = {
            "HEADER_METADATA": {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70,
                                "cached_tokens": 0, "reasoning_tokens": 0, "n_calls": 1},
        }
        by_group = {
            "header": {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70,
                       "cached_tokens": 0, "reasoning_tokens": 0, "n_calls": 1},
        }
        rows = [_row("scielo_preprints", f"r{i}", _tok(50, 20, by_stage, by_group, n_calls=1)) for i in range(4)]
        result = score(rows, METRICS, CFG_N_RESAMPLES, CFG_CL)
        md = render_markdown(result, METRICS)
        assert "## Tokens" in md
        assert "Per stage (overall sums)" in md
        assert "Per metric group (overall sums)" in md
        assert "HEADER_METADATA" in md
        assert "header" in md

    def test_tokens_section_absent_when_tokens_missing(self) -> None:
        # Back-compat: if there is no tokens key (hand-built result dict), rendering still works.
        result = {
            "overall": {
                "n": 1, "metrics": {
                    "title_match": {
                        "grobid": {"mean": 0.5, "ci_low": 0.4, "ci_high": 0.6},
                        "llm": {"mean": 0.6, "ci_low": 0.5, "ci_high": 0.7},
                        "delta_llm_minus_grobid": 0.1,
                        "wilcoxon_p_llm_vs_grobid": 0.5,
                    },
                    "authors_recall": {
                        "grobid": {"mean": 0.5, "ci_low": 0.4, "ci_high": 0.6},
                        "llm": {"mean": 0.7, "ci_low": 0.6, "ci_high": 0.8},
                        "delta_llm_minus_grobid": 0.2,
                        "wilcoxon_p_llm_vs_grobid": 0.5,
                    },
                },
            },
        }
        md = render_markdown(result, METRICS)
        assert "## Tokens" not in md
