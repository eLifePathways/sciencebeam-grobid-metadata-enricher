from __future__ import annotations

import pytest

from grobid_metadata_enricher.evaluation import evaluate_record, levenshtein_sim


class TestLevenshteinSim:
    def test_identical(self):
        assert levenshtein_sim("hello world", "hello world") == 1.0

    def test_empty_both(self):
        assert levenshtein_sim("", "") == 1.0

    def test_empty_gold(self):
        assert levenshtein_sim("", "some text") == 0.0

    def test_empty_predicted(self):
        assert levenshtein_sim("some text", "") == 0.0

    def test_none_coercion(self):
        assert levenshtein_sim(None, None) == 1.0  # type: ignore[arg-type]

    def test_partial_overlap(self):
        sim = levenshtein_sim("the quick brown fox", "the quick brown dog")
        assert 0.0 < sim < 1.0

    def test_symmetric(self):
        a, b = "abstract text here", "abstract text changed"
        assert levenshtein_sim(a, b) == pytest.approx(levenshtein_sim(b, a))

    def test_extra_predicted_text_penalised(self):
        gold = "short abstract"
        # jaccard_recall would score 1.0 here; levenshtein_sim should not
        predicted = "short abstract with a lot of extra words appended by the model"
        sim = levenshtein_sim(gold, predicted)
        assert sim < 1.0


class TestEvaluateRecordEditSim:
    def test_exact_match(self):
        metrics = evaluate_record({"abstract": "hello world"}, {"abstract": "hello world"})
        assert metrics["abstract_edit_sim"] == pytest.approx(1.0)

    def test_penalises_extra_text(self):
        gold = "short abstract"
        predicted = "short abstract with a lot of extra words appended"
        metrics = evaluate_record({"abstract": predicted}, {"abstract": gold})
        assert metrics["abstract_edit_sim"] < 1.0
        # abstract_recall (jaccard recall) scores 1.0 for the same input — showing the difference
        assert metrics["abstract_recall"] == pytest.approx(1.0)

    def test_multi_candidate_takes_max(self):
        predicted = "the correct abstract text"
        gold = {"abstracts": ["a completely different abstract", "the correct abstract text"]}
        metrics = evaluate_record({"abstract": predicted}, gold)
        assert metrics["abstract_edit_sim"] == pytest.approx(1.0)

    def test_partial_match_between_zero_and_one(self):
        metrics = evaluate_record(
            {"abstract": "the quick brown fox"},
            {"abstract": "the quick brown dog"},
        )
        assert 0.0 < metrics["abstract_edit_sim"] < 1.0
