from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from grobid_metadata_enricher.api import app

PDF_BYTES = b"%PDF-1.4 fake pdf content"


@pytest.fixture(name="client")
def _client() -> TestClient:
    return TestClient(app)


@pytest.fixture(name="mock_chat")
def _mock_chat() -> MagicMock:
    return MagicMock(return_value='{"title": "Test"}')


# ---------------------------------------------------------------------------
# GET /
# ---------------------------------------------------------------------------


class TestIndex:
    def test_returns_html(self, client: TestClient) -> None:
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "ScienceBeam" in response.text


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


class TestHealth:
    def test_ok(self, client: TestClient) -> None:
        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# POST /api/grobid
# ---------------------------------------------------------------------------


class TestGrobid:
    def test_rejects_non_pdf(self, client: TestClient) -> None:
        response = client.post(
            "/api/grobid",
            files={"file": ("document.txt", b"hello", "text/plain")},
        )
        assert response.status_code == 400
        assert "PDF" in response.json()["detail"]

    def test_grobid_failure(self, client: TestClient) -> None:
        with patch(
            "grobid_metadata_enricher.api.run_grobid",
            side_effect=RuntimeError("connection refused"),
        ):
            response = client.post(
                "/api/grobid",
                files={"file": ("paper.pdf", PDF_BYTES, "application/pdf")},
            )
        assert response.status_code == 502
        assert "GROBID" in response.json()["detail"]

    def test_success(self, client: TestClient) -> None:
        tei_xml = b"<TEI><teiHeader/></TEI>"
        with patch(
            "grobid_metadata_enricher.api.run_grobid",
            side_effect=lambda pdf, tei, **_: tei.write_bytes(tei_xml),
        ):
            response = client.post(
                "/api/grobid",
                files={"file": ("paper.pdf", PDF_BYTES, "application/pdf")},
            )
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/xml"
        assert response.content == tei_xml


# ---------------------------------------------------------------------------
# POST /transform
# ---------------------------------------------------------------------------


class TestTransform:
    def test_no_llm_backend(self, client: TestClient) -> None:
        with patch("grobid_metadata_enricher.api._chat", None):
            response = client.post(
                "/api/transform",
                files={"file": ("paper.pdf", PDF_BYTES, "application/pdf")},
            )
        assert response.status_code == 503
        assert "LLM" in response.json()["detail"]

    def test_rejects_non_pdf(self, client: TestClient, mock_chat: MagicMock) -> None:
        with patch("grobid_metadata_enricher.api._chat", mock_chat):
            response = client.post(
                "/api/transform",
                files={"file": ("document.txt", b"hello", "text/plain")},
            )
        assert response.status_code == 400
        assert "PDF" in response.json()["detail"]

    def test_accepts_pdf_by_content_type(self, client: TestClient, mock_chat: MagicMock) -> None:
        """A file without a .pdf extension is accepted when content-type is application/pdf."""
        prediction = {"title": "Test Paper"}
        with (
            patch("grobid_metadata_enricher.api._chat", mock_chat),
            patch("grobid_metadata_enricher.api.run_grobid"),
            patch("grobid_metadata_enricher.api.run_pdfalto"),
            patch("grobid_metadata_enricher.api.build_document_context"),
            patch("grobid_metadata_enricher.api.build_prediction", return_value=prediction),
        ):
            response = client.post(
                "/api/transform",
                files={"file": ("upload", PDF_BYTES, "application/pdf")},
            )
        assert response.status_code == 200

    def test_grobid_failure(self, client: TestClient, mock_chat: MagicMock) -> None:
        with (
            patch("grobid_metadata_enricher.api._chat", mock_chat),
            patch(
                "grobid_metadata_enricher.api.run_grobid",
                side_effect=RuntimeError("connection refused"),
            ),
        ):
            response = client.post(
                "/api/transform",
                files={"file": ("paper.pdf", PDF_BYTES, "application/pdf")},
            )
        assert response.status_code == 502
        assert "GROBID" in response.json()["detail"]

    def test_pdfalto_failure(self, client: TestClient, mock_chat: MagicMock) -> None:
        with (
            patch("grobid_metadata_enricher.api._chat", mock_chat),
            patch("grobid_metadata_enricher.api.run_grobid"),
            patch(
                "grobid_metadata_enricher.api.run_pdfalto",
                side_effect=RuntimeError("binary not found"),
            ),
        ):
            response = client.post(
                "/api/transform",
                files={"file": ("paper.pdf", PDF_BYTES, "application/pdf")},
            )
        assert response.status_code == 502
        assert "pdfalto" in response.json()["detail"]

    def test_pipeline_failure(self, client: TestClient, mock_chat: MagicMock) -> None:
        with (
            patch("grobid_metadata_enricher.api._chat", mock_chat),
            patch("grobid_metadata_enricher.api.run_grobid"),
            patch("grobid_metadata_enricher.api.run_pdfalto"),
            patch(
                "grobid_metadata_enricher.api.build_document_context",
                side_effect=RuntimeError("parse error"),
            ),
        ):
            response = client.post(
                "/api/transform",
                files={"file": ("paper.pdf", PDF_BYTES, "application/pdf")},
            )
        assert response.status_code == 500
        assert "Pipeline" in response.json()["detail"]

    def test_success(self, client: TestClient, mock_chat: MagicMock) -> None:
        prediction = {
            "title": "A Great Paper",
            "abstract": "We study things.",
            "authors": ["Alice", "Bob"],
        }
        with (
            patch("grobid_metadata_enricher.api._chat", mock_chat),
            patch("grobid_metadata_enricher.api.run_grobid"),
            patch("grobid_metadata_enricher.api.run_pdfalto"),
            patch("grobid_metadata_enricher.api.build_document_context"),
            patch("grobid_metadata_enricher.api.build_prediction", return_value=prediction),
        ):
            response = client.post(
                "/api/transform",
                files={"file": ("paper.pdf", PDF_BYTES, "application/pdf")},
            )
        assert response.status_code == 200
        assert response.json() == prediction
