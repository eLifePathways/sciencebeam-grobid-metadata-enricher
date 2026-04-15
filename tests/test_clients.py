from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from grobid_metadata_enricher.clients import run_grobid

GROBID_URL = "http://grobid-test:8070/api"


def _make_response(text: str = "<TEI/>") -> MagicMock:
    response = MagicMock()
    response.text = text
    return response


@pytest.fixture()
def pdf_path(tmp_path: Path) -> Path:
    path = tmp_path / "input.pdf"
    path.write_bytes(b"%PDF-1.4 fake")
    return path


@pytest.fixture()
def tei_path(tmp_path: Path) -> Path:
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
            with pytest.raises(RuntimeError, match="could not connect to GROBID"):
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
