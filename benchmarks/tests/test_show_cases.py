from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from benchmarks.show_cases import (
    _export_record,
    _find_jsonl,
    _label,
    _pred_key,
    word_diff,
)

_RESET = "\033[0m"
_RED = "\033[91m"
_GREEN = "\033[92m"


# ---------------------------------------------------------------------------
# word_diff
# ---------------------------------------------------------------------------


class TestWordDiff:
    def test_both_none(self) -> None:
        assert word_diff(None, None) == "(both None)"

    def test_reference_none(self) -> None:
        result = word_diff(None, "hello world")
        assert "hello world" in result
        assert "reference=None" in result

    def test_candidate_none(self) -> None:
        result = word_diff("hello world", None)
        assert "candidate=None" in result

    def test_identical(self) -> None:
        result = word_diff("the quick brown fox", "the quick brown fox")
        assert "[-" not in result
        assert "{+" not in result
        assert "the quick brown fox" in result

    def test_deletion(self) -> None:
        result = word_diff("the quick brown fox", "the brown fox")
        assert "[-quick-]" in result
        assert "{+" not in result

    def test_insertion(self) -> None:
        result = word_diff("the brown fox", "the quick brown fox")
        assert "{+quick+}" in result
        assert "[-" not in result

    def test_replacement(self) -> None:
        result = word_diff("the quick brown fox", "the slow brown fox")
        assert "[-quick-]" in result
        assert "{+slow+}" in result

    def test_multi_word_deletion(self) -> None:
        result = word_diff("one two three four", "one four")
        assert "[-two three-]" in result

    def test_multi_word_insertion(self) -> None:
        result = word_diff("one four", "one two three four")
        assert "{+two three+}" in result

    def test_empty_strings(self) -> None:
        result = word_diff("", "")
        assert "[-" not in result
        assert "{+" not in result

    def test_reference_empty_candidate_not(self) -> None:
        result = word_diff("", "hello")
        assert "{+hello+}" in result

    def test_candidate_empty_reference_not(self) -> None:
        result = word_diff("hello", "")
        assert "[-hello-]" in result


# ---------------------------------------------------------------------------
# _pred_key / _label
# ---------------------------------------------------------------------------


class TestPredKey:
    def test_strips_metrics_suffix(self) -> None:
        assert _pred_key("llm_metrics") == "llm_pred"

    def test_strips_grobid_metrics(self) -> None:
        assert _pred_key("grobid_metrics") == "grobid_pred"

    def test_no_metrics_suffix(self) -> None:
        assert _pred_key("llm") == "llm_pred"


class TestLabel:
    def test_strips_pred_suffix(self) -> None:
        assert _label("llm_pred") == "llm"

    def test_strips_grobid_pred(self) -> None:
        assert _label("grobid_pred") == "grobid"

    def test_no_pred_suffix(self) -> None:
        assert _label("llm") == "llm"


# ---------------------------------------------------------------------------
# _find_jsonl
# ---------------------------------------------------------------------------


class TestFindJsonl:
    def test_returns_file_directly(self, tmp_path: Path) -> None:
        jsonl = tmp_path / "per_document.jsonl"
        jsonl.write_text("{}\n", encoding="utf-8")
        assert _find_jsonl(jsonl) == jsonl

    def test_finds_jsonl_in_directory(self, tmp_path: Path) -> None:
        jsonl = tmp_path / "per_document.jsonl"
        jsonl.write_text("{}\n", encoding="utf-8")
        assert _find_jsonl(tmp_path) == jsonl

    def test_raises_when_missing(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            _find_jsonl(tmp_path)


# ---------------------------------------------------------------------------
# _export_record
# ---------------------------------------------------------------------------

def _make_record(record_id: str = "doc1", corpus: str = "ore") -> Dict[str, Any]:
    return {
        "record_id": record_id,
        "corpus": corpus,
        "gold": {"title": "Gold Title", "abstract": "Gold abstract text."},
        "grobid_pred": {"title": "Grobid Title", "abstract": "Grobid abstract text."},
        "llm_pred": {"title": "LLM Title", "abstract": "LLM abstract text."},
    }


class TestExportRecord:
    def _out_dir(self, run_dir: Path, mode: str, metric: str, corpus: str) -> Path:
        return run_dir / "examples" / mode / metric / corpus

    def test_creates_output_directory(self, tmp_path: Path) -> None:
        r = _make_record()
        _export_record(r, tmp_path, "title_edit_sim", "title", "improvement", "llm_pred", "grobid_pred")
        assert self._out_dir(tmp_path, "improvement", "title_edit_sim", "ore").is_dir()

    def test_writes_text_fields(self, tmp_path: Path) -> None:
        r = _make_record(record_id="doc1", corpus="ore")
        _export_record(r, tmp_path, "title_edit_sim", "title", "improvement", "llm_pred", "grobid_pred")
        out = self._out_dir(tmp_path, "improvement", "title_edit_sim", "ore")
        assert (out / "doc1.gold.title.txt").read_text() == "Gold Title"
        assert (out / "doc1.grobid.title.txt").read_text() == "Grobid Title"
        assert (out / "doc1.llm.title.txt").read_text() == "LLM Title"

    def test_writes_json_fields_for_list_values(self, tmp_path: Path) -> None:
        r = _make_record()
        r["gold"]["authors"] = ["Alice", "Bob"]
        r["grobid_pred"]["authors"] = ["Alice"]
        r["llm_pred"]["authors"] = ["Alice", "Bob"]
        _export_record(r, tmp_path, "authors_recall", "authors", "improvement", "llm_pred", "grobid_pred")
        out = self._out_dir(tmp_path, "improvement", "authors_recall", "ore")
        assert json.loads((out / "doc1.gold.authors.json").read_text()) == ["Alice", "Bob"]

    def test_copies_pdf_when_present(self, tmp_path: Path) -> None:
        r = _make_record(record_id="doc1", corpus="ore")
        data_dir = tmp_path / "data" / "ore"
        data_dir.mkdir(parents=True)
        (data_dir / "doc1.pdf").write_bytes(b"%PDF fake")
        _export_record(r, tmp_path, "title_edit_sim", "title", "improvement", "llm_pred", "grobid_pred")
        out = self._out_dir(tmp_path, "improvement", "title_edit_sim", "ore")
        assert (out / "doc1.pdf").exists()

    def test_copies_xml_when_present(self, tmp_path: Path) -> None:
        r = _make_record(record_id="doc1", corpus="ore")
        data_dir = tmp_path / "data" / "ore"
        data_dir.mkdir(parents=True)
        (data_dir / "doc1.xml").write_text("<record/>", encoding="utf-8")
        _export_record(r, tmp_path, "title_edit_sim", "title", "improvement", "llm_pred", "grobid_pred")
        out = self._out_dir(tmp_path, "improvement", "title_edit_sim", "ore")
        assert (out / "doc1.xml").exists()

    def test_skips_missing_pdf_and_xml(self, tmp_path: Path) -> None:
        r = _make_record(record_id="doc1", corpus="ore")
        _export_record(r, tmp_path, "title_edit_sim", "title", "improvement", "llm_pred", "grobid_pred")
        out = self._out_dir(tmp_path, "improvement", "title_edit_sim", "ore")
        assert not (out / "doc1.pdf").exists()
        assert not (out / "doc1.xml").exists()

    def test_copies_tei_when_present(self, tmp_path: Path) -> None:
        r = _make_record(record_id="doc1", corpus="ore")
        tei_dir = tmp_path / "ore" / "tei" / "grobid"
        tei_dir.mkdir(parents=True)
        (tei_dir / "doc1.tei.xml").write_text("<TEI/>", encoding="utf-8")
        _export_record(r, tmp_path, "title_edit_sim", "title", "improvement", "llm_pred", "grobid_pred")
        out = self._out_dir(tmp_path, "improvement", "title_edit_sim", "ore")
        assert (out / "doc1.tei.xml").read_text() == "<TEI/>"

    def test_skips_tei_when_absent(self, tmp_path: Path) -> None:
        r = _make_record(record_id="doc1", corpus="ore")
        _export_record(r, tmp_path, "title_edit_sim", "title", "improvement", "llm_pred", "grobid_pred")
        out = self._out_dir(tmp_path, "improvement", "title_edit_sim", "ore")
        assert not (out / "doc1.tei.xml").exists()

    def test_tei_uses_custom_parser(self, tmp_path: Path) -> None:
        r = _make_record(record_id="doc1", corpus="ore")
        tei_dir = tmp_path / "ore" / "tei" / "custom_parser"
        tei_dir.mkdir(parents=True)
        (tei_dir / "doc1.tei.xml").write_text("<TEI custom/>", encoding="utf-8")
        _export_record(
            r, tmp_path, "title_edit_sim", "title", "improvement",
            "llm_pred", "grobid_pred", parser="custom_parser",
        )
        out = self._out_dir(tmp_path, "improvement", "title_edit_sim", "ore")
        assert (out / "doc1.tei.xml").read_text() == "<TEI custom/>"

    def test_tei_default_parser_not_used_when_only_custom_exists(self, tmp_path: Path) -> None:
        r = _make_record(record_id="doc1", corpus="ore")
        tei_dir = tmp_path / "ore" / "tei" / "custom_parser"
        tei_dir.mkdir(parents=True)
        (tei_dir / "doc1.tei.xml").write_text("<TEI custom/>", encoding="utf-8")
        _export_record(r, tmp_path, "title_edit_sim", "title", "improvement", "llm_pred", "grobid_pred")
        out = self._out_dir(tmp_path, "improvement", "title_edit_sim", "ore")
        assert not (out / "doc1.tei.xml").exists()
