# pylint: disable=unused-argument
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict

from benchmarks import predict


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
        "grobid": {
            "parser": "grobid",
            "url": "http://grobid-test:8070/api",
            "pdfalto_start_page": 1,
            "pdfalto_end_page": 2,
        },
        "llm": {
            "workers": 1,
            "temperature": 0.0,
            "max_tokens": 800,
            "concurrency": 1,
        },
    }


def test_split_parse_and_llm_stages_preserve_process_one_result(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    def fake_run_grobid(pdf_path: Path, tei_path: Path, **_: Any) -> None:
        tei_path.write_text("<TEI><teiHeader/></TEI>", encoding="utf-8")

    def fake_run_pdfalto(pdf_path: Path, alto_path: Path, **_: Any) -> None:
        alto_path.write_text("<alto/>", encoding="utf-8")

    pred = {"title": "T", "abstract": "A", "keywords": ["K"]}
    metrics = {"title_match": 1.0, "abstract_f1": 1.0, "keywords_f1": 1.0}
    monkeypatch.setattr(predict, "run_grobid", fake_run_grobid)
    monkeypatch.setattr(predict, "run_pdfalto", fake_run_pdfalto)
    monkeypatch.setattr(predict, "extract_alto_lines", lambda _: [{"page": 0, "text": "line"}])
    monkeypatch.setattr(predict, "read_tei_header", lambda _: "header")
    monkeypatch.setattr(predict, "extract_tei_fields", lambda _: dict(pred))
    monkeypatch.setattr(predict, "extract_tei_abstracts", lambda _: ["A"])
    monkeypatch.setattr(predict, "build_prediction", lambda *_, **__: dict(pred))
    monkeypatch.setattr(predict, "extract_gold", lambda *_: dict(pred))
    monkeypatch.setattr(predict, "evaluate_record", lambda *_: dict(metrics))

    def make_chat(_recorder: Any) -> Callable[..., str]:
        return lambda *_, **__: ""
    row = _row(tmp_path)
    cfg = _cfg()

    legacy = predict.process_one(row, make_chat, tmp_path / "legacy", cfg)
    assert predict.process_inputs(row, tmp_path / "split", cfg) is None
    split = predict.process_prediction(row, make_chat, tmp_path / "split", cfg)

    assert split == legacy
