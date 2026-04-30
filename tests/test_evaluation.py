from __future__ import annotations

import pytest

from grobid_metadata_enricher.evaluation import evaluate_record, levenshtein_sim


class TestLevenshteinSim:
    def test_identical(self) -> None:
        assert levenshtein_sim("hello world", "hello world") == 1.0

    def test_empty_both(self) -> None:
        assert levenshtein_sim("", "") == 1.0

    def test_empty_gold(self) -> None:
        assert levenshtein_sim("", "some text") == 0.0

    def test_empty_predicted(self) -> None:
        assert levenshtein_sim("some text", "") == 0.0

    def test_none_coercion(self) -> None:
        assert levenshtein_sim(None, None) == 1.0  # type: ignore[arg-type]

    def test_partial_overlap(self) -> None:
        sim = levenshtein_sim("the quick brown fox", "the quick brown dog")
        assert 0.0 < sim < 1.0

    def test_symmetric(self) -> None:
        a, b = "abstract text here", "abstract text changed"
        assert levenshtein_sim(a, b) == pytest.approx(levenshtein_sim(b, a))  # pylint: disable=arguments-out-of-order

    def test_extra_predicted_text_penalised(self) -> None:
        gold = "short abstract"
        # jaccard_recall would score 1.0 here; levenshtein_sim should not
        predicted = "short abstract with a lot of extra words appended by the model"
        sim = levenshtein_sim(gold, predicted)
        assert sim < 1.0


class TestEvaluateRecordEditSim:
    def test_exact_match(self) -> None:
        metrics = evaluate_record({"abstract": "hello world"}, {"abstract": "hello world"})
        assert metrics["abstract_edit_sim"] == pytest.approx(1.0)

    def test_penalises_extra_text(self) -> None:
        gold = "short abstract"
        predicted = "short abstract with a lot of extra words appended"
        metrics = evaluate_record({"abstract": predicted}, {"abstract": gold})
        assert metrics["abstract_edit_sim"] < 1.0
        # abstract_recall (jaccard recall) scores 1.0 for the same input — showing the difference
        assert metrics["abstract_recall"] == pytest.approx(1.0)

    def test_multi_candidate_takes_max(self) -> None:
        predicted = "the correct abstract text"
        gold = {"abstracts": ["a completely different abstract", "the correct abstract text"]}
        metrics = evaluate_record({"abstract": predicted}, gold)
        assert metrics["abstract_edit_sim"] == pytest.approx(1.0)

    def test_partial_match_between_zero_and_one(self) -> None:
        metrics = evaluate_record(
            {"abstract": "the quick brown fox"},
            {"abstract": "the quick brown dog"},
        )
        assert 0.0 < metrics["abstract_edit_sim"] < 1.0


class TestF1Metrics:
    def test_abstract_f1_recall_precision(self) -> None:
        gold = "short abstract"
        # pred has all gold tokens (recall=1) plus 5 extra tokens (precision=2/7)
        pred = "short abstract with a lot of extra"
        m = evaluate_record({"abstract": pred}, {"abstract": gold})
        assert m["abstract_recall"] == pytest.approx(1.0)
        assert m["abstract_precision"] == pytest.approx(2/7)
        assert m["abstract_f1"] == pytest.approx(2 * 1.0 * (2/7) / (1.0 + 2/7))

    def test_keywords_f1(self) -> None:
        m = evaluate_record(
            {"keywords": ["alpha", "beta", "gamma", "delta"]},
            {"keywords": ["alpha", "beta"]},
        )
        # gold=2, pred=4, inter=2 -> rec=1.0, pre=0.5, f1=2/3
        assert m["keywords_recall"] == pytest.approx(1.0)
        assert m["keywords_precision"] == pytest.approx(0.5)
        assert m["keywords_f1"] == pytest.approx(2 * 1.0 * 0.5 / 1.5)

    def test_identifiers_f1_penalises_orcid_dump(self) -> None:
        m = evaluate_record(
            {"identifiers": ["10.1234/abc", "0000-0001-2345-6789", "0000-0002-3456-7890"]},
            {"identifiers": ["10.1234/abc"]},
        )
        # gold=1, pred=3, inter=1 -> rec=1.0, pre=1/3, f1=0.5
        assert m["identifiers_recall"] == pytest.approx(1.0)
        assert m["identifiers_precision"] == pytest.approx(1/3)
        assert m["identifiers_f1"] == pytest.approx(0.5)

    def test_empty_pred_yields_zero_precision_and_f1(self) -> None:
        m = evaluate_record({"keywords": []}, {"keywords": ["alpha"]})
        assert m["keywords_recall"] == 0.0
        assert m["keywords_precision"] == 0.0
        assert m["keywords_f1"] == 0.0

    def test_empty_gold_yields_none(self) -> None:
        m = evaluate_record({"keywords": ["alpha"]}, {"keywords": []})
        assert m["keywords_recall"] is None
        assert m["keywords_precision"] is None
        assert m["keywords_f1"] is None

    def test_section_head_f1_with_bipartite_matching(self) -> None:
        # 2 gold heads, 4 pred heads (2 match, 2 are noise/duplicates).
        # bipartite: each pred used at most once -> matched_gold=2, matched_pred=2
        # rec = 2/2 = 1.0, pre = 2/4 = 0.5, f1 = 2/3
        m = evaluate_record(
            {"body_sections": ["Introduction", "Methods", "Introduction text", "Random"]},
            {"body_sections": ["Introduction", "Methods"]},
        )
        assert m["body_section_recall"] == pytest.approx(1.0)
        assert m["body_section_precision"] == pytest.approx(0.5)
        assert m["body_section_f1"] == pytest.approx(2/3)
