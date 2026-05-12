# pylint: disable=duplicate-code
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict
from unittest.mock import MagicMock

import pytest

from benchmarks import predict
from grobid_metadata_enricher.clients import LLMCallError


def _row(tmp_path: Path) -> Dict[str, str]:
    pdf_path = tmp_path / "doc.pdf"
    xml_path = tmp_path / "doc.xml"
    pdf_path.write_bytes(b"%PDF-1.4 fake")
    xml_path.write_text("<record/>", encoding="utf-8")
    return {
        "corpus": "scielo_preprints",
        "record_id": "doc-1",
        "pdf_path": str(pdf_path),
        "xml_path": str(xml_path),
    }


def _cfg() -> Dict[str, Any]:
    return {
        "grobid": {"parser": "grobid"},
        "llm": {"workers": 1, "temperature": 0.0, "max_tokens": 800},
    }


def _make_chat_fn(chat: Callable[..., str]) -> Callable[[Any], Callable[..., str]]:
    return lambda recorder: chat


def _patch_extraction(monkeypatch: Any) -> None:
    pred = {"title": "T", "authors": [], "abstract": "", "keywords": [], "identifiers": [], "language": "en"}
    monkeypatch.setattr(predict, "extract_alto_lines", lambda _: [])
    monkeypatch.setattr(predict, "read_tei_header", lambda _: "")
    monkeypatch.setattr(predict, "extract_tei_fields", lambda _: dict(pred))
    monkeypatch.setattr(predict, "extract_tei_abstracts", lambda _: [])


def test_process_prediction_propagates_llm_call_error(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    _patch_extraction(monkeypatch)
    monkeypatch.setattr(
        predict, "build_prediction", lambda *_, **__: (_ for _ in ()).throw(LLMCallError("network error"))
    )

    with pytest.raises(LLMCallError):
        predict.process_prediction(_row(tmp_path), _make_chat_fn(MagicMock()), tmp_path, _cfg())


def test_process_prediction_returns_error_dict_for_non_llm_exception(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    _patch_extraction(monkeypatch)
    monkeypatch.setattr(
        predict, "build_prediction", lambda *_, **__: (_ for _ in ()).throw(ValueError("bad response"))
    )

    result = predict.process_prediction(_row(tmp_path), _make_chat_fn(MagicMock()), tmp_path, _cfg())
    assert result is not None
    assert "error" in result
    assert "LLMCallError" not in result["error"]
