from __future__ import annotations

import pytest

from grobid_metadata_enricher.evaluation import evaluate_record, get_max_levenshtein_sim, levenshtein_sim


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


class TestGetMaxLevenshteinSim:
    def test_picks_best_of_multiple_golds(self) -> None:
        assert get_max_levenshtein_sim("hello", ["world", "hello"]) == pytest.approx(1.0)

    def test_returns_max_not_first_or_last(self) -> None:
        best = levenshtein_sim("hello", "helo")
        worse = levenshtein_sim("hello", "world")
        result = get_max_levenshtein_sim("hello", ["world", "helo", "world"])
        assert result == pytest.approx(best)
        assert result > worse


class TestEvaluateRecord:
    @pytest.mark.parametrize("params", [
        pytest.param({"field": "abstract", "list_key": "abstracts"}, id="abstract"),
        pytest.param({"field": "title", "list_key": "titles"}, id="title"),
    ])
    class TestCommonEditSim:
        def test_edit_sim_exact_match(self, params: dict) -> None:
            field = params["field"]
            metrics = evaluate_record({field: "exact text"}, {field: "exact text"})
            assert metrics[f"{field}_edit_sim"] == pytest.approx(1.0)

        def test_edit_sim_partial_match_between_zero_and_one(self, params: dict) -> None:
            field = params["field"]
            metrics = evaluate_record(
                {field: "the quick brown fox"},
                {field: "the quick brown dog"},
            )
            assert 0.0 < metrics[f"{field}_edit_sim"] < 1.0

        def test_edit_sim_multi_candidate_takes_max(self, params: dict) -> None:
            field, list_key = params["field"], params["list_key"]
            predicted = "the correct text"
            golds = ["a completely different text", "the correct text"]
            metrics = evaluate_record({field: predicted}, {list_key: golds})
            assert metrics[f"{field}_edit_sim"] == pytest.approx(get_max_levenshtein_sim(predicted, golds))

    class TestTitleEditSim:
        def test_penalises_extra_text(self) -> None:
            gold = "A Great Paper"
            predicted = "A Great Paper: with subtitle and extra disclaimer text appended"
            metrics = evaluate_record({"title": predicted}, {"title": gold})
            assert metrics["title_edit_sim"] < 1.0
            # title_match is binary and also 0 here
            assert metrics["title_match"] == 0

        def test_empty_predicted(self) -> None:
            metrics = evaluate_record({"title": ""}, {"title": "Some Title"})
            assert metrics["title_edit_sim"] == pytest.approx(0.0)

        def test_empty_gold(self) -> None:
            metrics = evaluate_record({"title": "Some Title"}, {"title": ""})
            assert metrics["title_edit_sim"] == pytest.approx(0.0)
