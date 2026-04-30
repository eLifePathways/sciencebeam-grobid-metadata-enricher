from __future__ import annotations

from grobid_metadata_enricher.pipeline import dedupe_blocks


class TestDedupeBlocks:
    def test_empty(self) -> None:
        assert dedupe_blocks([]) == []

    def test_drops_blank(self) -> None:
        assert dedupe_blocks(["", "   ", "\t\n"]) == []

    def test_keeps_distinct(self) -> None:
        assert dedupe_blocks(["one", "two", "three"]) == ["one", "two", "three"]

    def test_drops_exact_duplicate_after_whitespace_normalisation(self) -> None:
        assert dedupe_blocks(["hello world", "  hello   world  "]) == ["hello world"]

    def test_drops_subset_block(self) -> None:
        # A clean TEI abstract that also appears as a prefix of an OCR marker-window
        # block must be dropped; only the longer block survives.
        clean = "We report the results of the COVID Moonshot."
        noisy = (
            "We report the results of the COVID Moonshot. "
            "Introduction Despite rapid progress in vaccine development..."
        )
        result = dedupe_blocks([clean, noisy])
        assert result == [noisy]

    def test_keeps_two_genuinely_distinct_languages(self) -> None:
        # Bilingual abstracts (e.g. SciELO PT + EN) must both survive.
        pt = "Resumo: este artigo descreve um experimento."
        en = "Abstract: this article describes an experiment."
        result = dedupe_blocks([pt, en])
        assert pt in result and en in result
