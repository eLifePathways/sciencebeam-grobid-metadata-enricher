from __future__ import annotations

import pytest

from grobid_metadata_enricher.evaluation import evaluate_record, levenshtein_sim, normalized_edit_sim


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
        # Token-recall style metrics would score 1.0 here; edit similarity should not.
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
        assert metrics["abstract_recall"] == pytest.approx(normalized_edit_sim(gold, predicted))

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

    def test_empty_gold_returns_none_for_both(self) -> None:
        m = evaluate_record({"abstract": "extracted from PDF"}, {"abstract": ""})
        assert m["abstract_edit_sim"] is None
        assert m["abstract_recall"] is None

        m = evaluate_record({"abstract": ""}, {"abstract": ""})
        assert m["abstract_edit_sim"] is None
        assert m["abstract_recall"] is None

    def test_empty_abstracts_list_returns_none(self) -> None:
        m = evaluate_record({"abstract": "extracted content"}, {"abstracts": ["", None]})
        assert m["abstract_edit_sim"] is None
        assert m["abstract_recall"] is None


class TestF1Metrics:
    def test_abstract_f1_recall_precision(self) -> None:
        gold = "short abstract"
        pred = "short abstract with a lot of extra"
        m = evaluate_record({"abstract": pred}, {"abstract": gold})
        expected = normalized_edit_sim(gold, pred)
        assert m["abstract_recall"] == pytest.approx(expected)
        assert m["abstract_precision"] == pytest.approx(expected)
        assert m["abstract_f1"] == pytest.approx(expected)

    def test_keywords_f1(self) -> None:
        m = evaluate_record(
            {"keywords": ["alpha", "beta", "gamma", "delta"]},
            {"keywords": ["alpha", "beta"]},
        )
        assert m["keywords_recall"] == pytest.approx(1.0)
        assert m["keywords_precision"] == pytest.approx(0.5)
        assert m["keywords_f1"] == pytest.approx(2 * 1.0 * 0.5 / 1.5)

    def test_identifiers_f1_penalises_orcid_dump(self) -> None:
        m = evaluate_record(
            {"identifiers": ["10.1234/abc", "0000-0001-2345-6789", "0000-0002-3456-7890"]},
            {"identifiers": ["10.1234/abc"]},
        )
        assert m["identifiers_recall"] == pytest.approx(1.0)
        assert m["identifiers_precision"] == pytest.approx(1/3)
        assert m["identifiers_f1"] == pytest.approx(0.5)

    def test_identifier_f1_ignores_doi_linebreak_spaces(self) -> None:
        m = evaluate_record(
            {"identifiers": ["10.1590/s1679-49742021000300023"]},
            {"identifiers": ["10.1590/s1679- 49742021000300023"]},
        )
        assert m["identifiers_f1"] == pytest.approx(1.0)

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
        m = evaluate_record(
            {"body_sections": ["Introduction", "Methods", "Introduction text", "Random"]},
            {"body_sections": ["Introduction", "Methods"]},
        )
        assert m["body_section_recall"] == pytest.approx(1.0)
        assert m["body_section_precision"] == pytest.approx(0.5)
        assert m["body_section_f1"] == pytest.approx(2/3)

    def test_section_head_matching_uses_edit_similarity(self) -> None:
        m = evaluate_record(
            {"body_sections": ["gamma beta alpha"]},
            {"body_sections": ["alpha beta gamma"]},
        )
        assert m["body_section_f1"] == pytest.approx(0.0)

    def test_caption_matching_uses_edit_similarity(self) -> None:
        m = evaluate_record(
            {"figure_captions": ["Figure 1. delta gamma beta alpha"]},
            {"figure_captions": ["Figure 1. alpha beta gamma delta"]},
        )
        assert m["figure_caption_f1"] == pytest.approx(0.0)

    def test_reference_combined_title_matching_uses_edit_similarity(self) -> None:
        m = evaluate_record(
            {"reference_titles": ["delta gamma beta alpha"]},
            {"reference_records": [{"title": "alpha beta gamma delta", "doi": ""}]},
        )
        assert m["reference_combined_f1"] == pytest.approx(0.0)
