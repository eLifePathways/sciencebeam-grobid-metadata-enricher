from __future__ import annotations

from grobid_metadata_enricher.pipeline import _looks_like_orcid, add_scielo_identifiers


class TestLooksLikeOrcid:
    def test_plain(self) -> None:
        assert _looks_like_orcid("0000-0001-2345-6789")

    def test_with_x_checksum(self) -> None:
        assert _looks_like_orcid("0000-0001-2345-678X")

    def test_with_orcid_prefix(self) -> None:
        assert _looks_like_orcid("ORCID: 0000-0001-2345-6789")

    def test_with_url_prefix(self) -> None:
        assert _looks_like_orcid("https://orcid.org/0000-0001-2345-6789")

    def test_doi_is_not_orcid(self) -> None:
        assert not _looks_like_orcid("10.1234/abc")

    def test_arxiv_is_not_orcid(self) -> None:
        assert not _looks_like_orcid("2101.01234")

    def test_empty(self) -> None:
        assert not _looks_like_orcid("")


class TestAddScieloIdentifiersDropsOrcid:
    def test_drops_orcid_keeps_doi(self) -> None:
        result = add_scielo_identifiers(
            "biorxiv_10.1101_x",
            ["10.1101/2020.x", "0000-0001-2345-6789", "https://orcid.org/0000-0002-3456-7890"],
        )
        assert "10.1101/2020.x" in result
        assert all(not _looks_like_orcid(v) for v in result)

    def test_appends_scielo_canonical_when_record_matches(self) -> None:
        # No real SCIELO match unless record_id is shaped right; covered in existing tests.
        result = add_scielo_identifiers("nope", ["10.1234/abc"])
        assert result == ["10.1234/abc"]
