from __future__ import annotations

import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from grobid_metadata_enricher.clients import (
    PARSER_GROBID,
    PARSER_SCIENCEBEAM,
    AoaiPool,
    LLMCallError,
    OpenAIClient,
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


class TestAoaiPoolRouting:
    def _pool_path(self, tmp_path: Path) -> Path:
        pool_path = tmp_path / "pool.json"
        pool_path.write_text(
            """
            [
              {"id":"b0","endpoint":"https://b0/","deployment":"d","apiKey":"k","apiVersion":"v1"},
              {"id":"b1","endpoint":"https://b1/","deployment":"d","apiKey":"k","apiVersion":"v1"},
              {"id":"b2","endpoint":"https://b2/","deployment":"d","apiKey":"k","apiVersion":"v1"}
            ]
            """,
            encoding="utf-8",
        )
        return pool_path

    def test_round_robin_routing_is_order_sensitive(self, tmp_path: Path) -> None:
        pool_path = self._pool_path(tmp_path)
        target_if_first = AoaiPool(pool_path).next_backend().backend_id
        pool = AoaiPool(pool_path)
        pool.next_backend()
        target_if_second = pool.next_backend().backend_id

        assert target_if_first == "b0"
        assert target_if_second == "b1"

    def test_stable_routing_ignores_unrelated_round_robin_calls(self, tmp_path: Path) -> None:
        pool = AoaiPool(self._pool_path(tmp_path), routing="stable")
        messages = [{"role": "user", "content": "same prompt"}]
        expected = pool.backend_for_request(messages, step_name="HEADER_METADATA").backend_id

        for _ in range(10):
            pool.next_backend()

        assert pool.backend_for_request(messages, step_name="HEADER_METADATA").backend_id == expected
        assert pool.backend_for_request(messages, step_name="HEADER_METADATA", attempt=1).backend_id != expected

    def test_rejects_unknown_routing(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Unsupported AOAI pool routing"):
            AoaiPool(self._pool_path(tmp_path), routing="random")


def _http_error(code: int) -> urllib.error.HTTPError:
    import http.client
    return urllib.error.HTTPError(
        url="http://example.com", code=code, msg="error", hdrs=http.client.HTTPMessage(), fp=None
    )


class TestAoaiPoolChatWithUsage:
    def _pool(self, tmp_path: Path) -> AoaiPool:
        pool_path = tmp_path / "pool.json"
        pool_path.write_text(
            '[{"id":"b0","endpoint":"https://b0/","deployment":"d","apiKey":"k","apiVersion":"v1"}]',
            encoding="utf-8",
        )
        return AoaiPool(pool_path)

    def test_raises_llm_call_error_on_non_retryable_http_error(self, tmp_path: Path) -> None:
        with (
            patch("grobid_metadata_enricher.clients.urllib.request.urlopen", side_effect=_http_error(401)),
            patch("grobid_metadata_enricher.clients.time.sleep"),
        ):
            with pytest.raises(LLMCallError, match="HTTP 401"):
                self._pool(tmp_path).chat_with_usage([{"role": "user", "content": "hi"}])

    def test_raises_llm_call_error_after_retryable_http_errors_exhausted(self, tmp_path: Path) -> None:
        with (
            patch("grobid_metadata_enricher.clients.urllib.request.urlopen", side_effect=_http_error(429)),
            patch("grobid_metadata_enricher.clients.time.sleep"),
        ):
            with pytest.raises(LLMCallError, match="after 3 attempts"):
                self._pool(tmp_path).chat_with_usage([{"role": "user", "content": "hi"}], max_attempts=3)


class TestOpenAIClientChatWithUsage:
    def _client(self) -> OpenAIClient:
        return OpenAIClient(api_key="test-key", model="gpt-4o", base_url="https://api.example.com")

    def test_raises_llm_call_error_on_non_retryable_http_error(self) -> None:
        with (
            patch("grobid_metadata_enricher.clients.urllib.request.urlopen", side_effect=_http_error(401)),
            patch("grobid_metadata_enricher.clients.time.sleep"),
        ):
            with pytest.raises(LLMCallError, match="HTTP 401"):
                self._client().chat_with_usage([{"role": "user", "content": "hi"}])

    def test_raises_llm_call_error_after_retryable_http_errors_exhausted(self) -> None:
        with (
            patch("grobid_metadata_enricher.clients.urllib.request.urlopen", side_effect=_http_error(429)),
            patch("grobid_metadata_enricher.clients.time.sleep"),
        ):
            with pytest.raises(LLMCallError, match="after 3 attempts"):
                self._client().chat_with_usage([{"role": "user", "content": "hi"}], max_attempts=3)
