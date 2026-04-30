from __future__ import annotations

from grobid_metadata_enricher.pipeline import dedupe_blocks, marker_windows


class TestDedupeBlocks:
    def test_empty(self) -> None:
        assert not dedupe_blocks([])

    def test_drops_blank(self) -> None:
        assert not dedupe_blocks(["", "   ", "\t\n"])

    def test_keeps_distinct(self) -> None:
        assert dedupe_blocks(["one", "two", "three"]) == ["one", "two", "three"]

    def test_drops_exact_duplicate_after_whitespace_normalisation(self) -> None:
        assert dedupe_blocks(["hello world", "  hello   world  "]) == ["hello world"]

    def test_drops_subset_block(self) -> None:
        clean = "We report the results of the COVID Moonshot."
        noisy = (
            "We report the results of the COVID Moonshot. "
            "Introduction Despite rapid progress in vaccine development..."
        )
        assert dedupe_blocks([clean, noisy]) == [noisy]

    def test_keeps_two_genuinely_distinct_languages(self) -> None:
        pt = "Resumo: este artigo descreve um experimento."
        en = "Abstract: this article describes an experiment."
        assert dedupe_blocks([pt, en]) == [pt, en]


def _line(text: str) -> dict:
    return {"text": text, "page": 1, "y": 0.0}


class TestMarkerWindowsTailStop:
    def test_stops_at_introduction(self) -> None:
        lines = [
            _line("Title of the paper"),
            _line("Authors"),
            _line("Abstract"),
            _line("This study investigates the effects of X on Y."),
            _line("We find that X increases Y by 20%."),
            _line("Introduction"),
            _line("Despite the importance of X..."),
            _line("body line 2"),
        ]
        blocks = marker_windows(
            lines, max_blocks=4, prefix_lines=0, suffix_lines=18, fallback_lines=160
        )
        assert len(blocks) == 1
        text = blocks[0].lower()
        assert "this study investigates" in text
        assert "despite the importance" not in text
        assert "body line 2" not in text

    def test_stops_at_next_abstract_marker(self) -> None:
        lines = [
            _line("Resumo"),
            _line("Este artigo investiga os efeitos."),
            _line("Abstract"),
            _line("This article investigates the effects."),
            _line("Introduction"),
            _line("body bleed line"),
        ]
        blocks = marker_windows(
            lines, max_blocks=4, prefix_lines=0, suffix_lines=18, fallback_lines=160
        )
        assert len(blocks) == 2
        pt, en = blocks[0].lower(), blocks[1].lower()
        assert "este artigo" in pt and "this article" not in pt
        assert "this article" in en and "body bleed" not in en

    def test_no_section_marker_falls_back_to_suffix_cap(self) -> None:
        lines = [_line("Abstract")] + [_line(f"line {i}") for i in range(20)]
        blocks = marker_windows(
            lines, max_blocks=4, prefix_lines=0, suffix_lines=5, fallback_lines=160
        )
        assert len(blocks) == 1
        text = blocks[0]
        assert "line 0" in text and "line 3" in text
        assert "line 4" not in text
