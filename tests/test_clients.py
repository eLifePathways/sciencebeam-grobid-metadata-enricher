from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from grobid_metadata_enricher.clients import (
    PARSER_GROBID,
    PARSER_SCIENCEBEAM,
    run_grobid,
)

GROBID_URL = "http://grobid-test:8070/api"


def _make_response(text: str = "<TEI/>") -> MagicMock:
    response = MagicMock()
    response.text = text
    response.headers = {"content-type": "application/tei+xml"}
    return response


@pytest.fixture(name="pdf_path")
def _pdf_path(tmp_path: Path) -> Path:
    path = tmp_path / "input.pdf"
    path.write_bytes(b"%PDF-1.4 fake")
    return path


@pytest.fixture(name="tei_path")
def _tei_path(tmp_path: Path) -> Path:
    return tmp_path / "output.tei.xml"


class TestRunGrobid:
    def test_success(self, pdf_path: Path, tei_path: Path) -> None:
        tei_xml = "<TEI><teiHeader/></TEI>"
        with patch(
            "grobid_metadata_enricher.clients.httpx.post",
            return_value=_make_response(tei_xml),
        ):
            run_grobid(pdf_path, tei_path, grobid_url=GROBID_URL)

        assert tei_path.read_text() == tei_xml

    def test_skips_if_output_already_exists(self, pdf_path: Path, tei_path: Path) -> None:
        tei_path.write_text("<TEI/>", encoding="utf-8")
        with patch("grobid_metadata_enricher.clients.httpx.post") as mock_post:
            run_grobid(pdf_path, tei_path, grobid_url=GROBID_URL)
        mock_post.assert_not_called()

    def test_retries_on_connect_error_then_succeeds(self, pdf_path: Path, tei_path: Path) -> None:
        tei_xml = "<TEI/>"
        with (
            patch(
                "grobid_metadata_enricher.clients.httpx.post",
                side_effect=[httpx.ConnectError("connection refused"), _make_response(tei_xml)],
            ),
            patch("grobid_metadata_enricher.clients.time.sleep") as mock_sleep,
        ):
            run_grobid(pdf_path, tei_path, grobid_url=GROBID_URL, timeout=60)

        mock_sleep.assert_called_once()
        assert tei_path.read_text() == tei_xml

    def test_raises_after_connect_timeout_exhausted(self, pdf_path: Path, tei_path: Path) -> None:
        with patch(
            "grobid_metadata_enricher.clients.httpx.post",
            side_effect=httpx.ConnectError("connection refused"),
        ):
            with pytest.raises(RuntimeError, match="could not connect to grobid"):
                run_grobid(pdf_path, tei_path, grobid_url=GROBID_URL, timeout=0)

    def test_raises_on_read_timeout(self, pdf_path: Path, tei_path: Path) -> None:
        with patch(
            "grobid_metadata_enricher.clients.httpx.post",
            side_effect=httpx.ReadTimeout("timed out"),
        ):
            with pytest.raises(RuntimeError, match="timed out"):
                run_grobid(pdf_path, tei_path, grobid_url=GROBID_URL, timeout=60)

    def test_raises_on_http_error(self, pdf_path: Path, tei_path: Path) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 503
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "503 Service Unavailable",
            request=httpx.Request("POST", GROBID_URL),
            response=httpx.Response(503),
        )
        with patch("grobid_metadata_enricher.clients.httpx.post", return_value=mock_response):
            with pytest.raises(RuntimeError, match="HTTP 503"):
                run_grobid(pdf_path, tei_path, grobid_url=GROBID_URL, timeout=60)

    def test_grobid_sends_grobid_form_fields_and_no_accept_header(
        self, pdf_path: Path, tei_path: Path
    ) -> None:
        # Pins that the GROBID consolidation/teiCoordinates flags are
        # still attached for the GROBID backend, and that no explicit
        # `headers` arg is passed to httpx.post (lfoppiano/grobid returns
        # 406 for Accept: application/tei+xml; only */* or an absent
        # Accept header yields 200).
        with patch(
            "grobid_metadata_enricher.clients.httpx.post",
            return_value=_make_response(),
        ) as mock_post:
            run_grobid(pdf_path, tei_path, grobid_url=GROBID_URL, parser=PARSER_GROBID)
        kwargs = mock_post.call_args.kwargs
        assert set(kwargs["data"].keys()) == {
            "teiCoordinates",
            "consolidateHeader",
            "consolidateCitations",
        }
        assert "headers" not in kwargs

    def test_sciencebeam_drops_grobid_form_fields(self, pdf_path: Path, tei_path: Path) -> None:
        # ScienceBeam Parser ignores GROBID-specific flags but accepts
        # the same multipart `input` field. The flags are dropped so the
        # wire payload is unambiguous and a 400-rejecting future parser
        # version cannot regress silently. As on the GROBID path, Accept
        # stays at the httpx default (*/*), which sciencebeam-parser
        # maps to TEI.
        with patch(
            "grobid_metadata_enricher.clients.httpx.post",
            return_value=_make_response(),
        ) as mock_post:
            run_grobid(
                pdf_path, tei_path, grobid_url=GROBID_URL, parser=PARSER_SCIENCEBEAM
            )
        kwargs = mock_post.call_args.kwargs
        assert kwargs["data"] == {}
        assert "headers" not in kwargs
        assert "input" in kwargs["files"]

    def test_unknown_parser_raises(self, pdf_path: Path, tei_path: Path) -> None:
        with pytest.raises(ValueError, match="Unsupported parser"):
            run_grobid(pdf_path, tei_path, grobid_url=GROBID_URL, parser="cermine")

    def test_silent_non_tei_response_raises(self, pdf_path: Path, tei_path: Path) -> None:
        # Reproduces the "everything failed silently" failure mode: the
        # upstream returns 200 but the body is not TEI. Without the
        # detector the empty/HTML body is written to disk and downstream
        # parsing returns zero metrics with no error trail.
        bogus = MagicMock()
        bogus.text = "<html><body>nothing to see here</body></html>"
        bogus.headers = {"content-type": "text/html"}
        with patch("grobid_metadata_enricher.clients.httpx.post", return_value=bogus):
            with pytest.raises(RuntimeError, match="non-TEI response"):
                run_grobid(
                    pdf_path,
                    tei_path,
                    grobid_url=GROBID_URL,
                    parser=PARSER_SCIENCEBEAM,
                )
        assert not tei_path.exists()
