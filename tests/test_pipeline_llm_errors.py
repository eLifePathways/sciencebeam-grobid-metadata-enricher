from __future__ import annotations

from typing import Any, Dict, List

import pytest

from grobid_metadata_enricher.clients import LLMCallError
from grobid_metadata_enricher.pipeline import (
    DocumentContext,
    _build_prediction_inner,
    _run_llm_task,
    predict_content_fields_from_alto,
)


def _minimal_context() -> DocumentContext:
    lines: List[Dict[str, Any]] = [{"text": "Title", "page": 0, "x": 0.0, "y": 0.0}]
    return DocumentContext(
        record_id="test-record",
        header_text="Title",
        lines=lines,
        first_page_lines=lines,
        tei_fields={
            "title": "Title",
            "authors": [],
            "abstract": "",
            "keywords": [],
            "identifiers": [],
            "language": "en",
        },
        tei_abstracts=[],
    )


class TestRunLlmTask:
    def test_returns_task_result_on_success(self) -> None:
        assert _run_llm_task(lambda: "ok", "default") == "ok"

    def test_returns_default_on_non_llm_exception(self) -> None:
        def task() -> str:
            raise ValueError("bad response")
        assert _run_llm_task(task, "default") == "default"

    def test_propagates_llm_call_error(self) -> None:
        def task() -> str:
            raise LLMCallError("network error")
        with pytest.raises(LLMCallError):
            _run_llm_task(task, "default")


class TestBuildPredictionInnerErrors:
    def _raising_chat(self, exc: Exception) -> Any:
        def chat(messages: Any, **kwargs: Any) -> str:
            raise exc
        return chat

    def test_propagates_llm_call_error_single_worker(self) -> None:
        with pytest.raises(LLMCallError):
            _build_prediction_inner(
                _minimal_context(),
                self._raising_chat(LLMCallError("auth failed")),
                per_document_llm_workers=1,
            )

    def test_propagates_llm_call_error_multi_worker(self) -> None:
        with pytest.raises(LLMCallError):
            _build_prediction_inner(
                _minimal_context(),
                self._raising_chat(LLMCallError("auth failed")),
                per_document_llm_workers=2,
            )

    def test_swallows_non_llm_exception_single_worker(self) -> None:
        result = _build_prediction_inner(
            _minimal_context(),
            self._raising_chat(ValueError("bad json")),
            per_document_llm_workers=1,
        )
        assert result is not None

    def test_swallows_non_llm_exception_multi_worker(self) -> None:
        result = _build_prediction_inner(
            _minimal_context(),
            self._raising_chat(ValueError("bad json")),
            per_document_llm_workers=2,
        )
        assert result is not None


class TestPredictContentFieldsErrors:
    def _lines(self) -> List[Dict[str, Any]]:
        return [
            {"text": "Introduction", "page": 0, "x": 72.0, "y": 100.0},
            # Matches _FIGURE_CAPTION_START_RE → produces figure_candidate_text > 20 chars,
            # ensuring at least one _call is submitted to the executor.
            {"text": "Figure 1. Schematic overview of the experimental pipeline.", "page": 1, "x": 72.0, "y": 200.0},
        ]

    def test_propagates_llm_call_error(self) -> None:
        def chat(messages: Any, **kwargs: Any) -> str:
            raise LLMCallError("rate limit exhausted")
        with pytest.raises(LLMCallError):
            predict_content_fields_from_alto(self._lines(), chat)

    def test_swallows_non_llm_exception(self) -> None:
        def chat(messages: Any, **kwargs: Any) -> str:
            raise ValueError("unexpected response")
        result = predict_content_fields_from_alto(self._lines(), chat)
        assert result is not None
