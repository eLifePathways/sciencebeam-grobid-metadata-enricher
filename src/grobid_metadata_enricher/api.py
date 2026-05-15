from __future__ import annotations

import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Optional  # noqa: F401

from fastapi import APIRouter, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

from .clients import (
    PREFER_AOAI_POOL,
    DEFAULT_GROBID_TIMEOUT,
    DEFAULT_GROBID_URL,
    DEFAULT_OPENAI_API_KEY,
    DEFAULT_OPENAI_BASE_URL,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_PARSER,
    DEFAULT_PDFALTO_BIN,
    DEFAULT_POOL_PATH,
    AoaiPool,
    OpenAIClient,
    run_grobid,
    run_pdfalto,
)
from .pipeline import DocumentPaths, build_document_context, build_prediction
from .telemetry import init_telemetry


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    init_telemetry()
    yield


_STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(
    title="ScienceBeam V2 API",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=_lifespan,
)
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")
router = APIRouter(prefix="/api")
_grobid_url: str = DEFAULT_GROBID_URL
_grobid_timeout: int = DEFAULT_GROBID_TIMEOUT
_parser: str = DEFAULT_PARSER


def _make_chat() -> Optional[Callable[..., str]]:
    if PREFER_AOAI_POOL and DEFAULT_POOL_PATH.exists():
        return AoaiPool(DEFAULT_POOL_PATH).chat
    if DEFAULT_OPENAI_API_KEY and DEFAULT_OPENAI_MODEL:
        return OpenAIClient(
            api_key=DEFAULT_OPENAI_API_KEY,
            model=DEFAULT_OPENAI_MODEL,
            base_url=DEFAULT_OPENAI_BASE_URL,
        ).chat
    if DEFAULT_POOL_PATH.exists():
        return AoaiPool(DEFAULT_POOL_PATH).chat
    return None


_chat: Optional[Callable[..., str]] = _make_chat()


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index() -> HTMLResponse:
    return HTMLResponse(content=(_STATIC_DIR / "index.html").read_text(encoding="utf-8"))


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/grobid", response_class=Response)
def grobid(
    file: UploadFile = File(..., description="PDF file to process with GROBID"),
) -> Response:
    filename = file.filename or ""
    content_type = file.content_type or ""
    if not filename.lower().endswith(".pdf") and content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Uploaded file must be a PDF.")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        pdf_path = tmp_dir / "input.pdf"
        tei_path = tmp_dir / "output.tei.xml"

        pdf_path.write_bytes(file.file.read())

        try:
            run_grobid(
                pdf_path,
                tei_path,
                grobid_url=_grobid_url,
                timeout=_grobid_timeout,
                parser=_parser,
            )
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"GROBID processing failed: {exc}") from exc

        return Response(content=tei_path.read_text(encoding="utf-8"), media_type="application/xml")


@router.post("/transform")
def transform(
    file: UploadFile = File(..., description="PDF file to transform"),
) -> dict[str, Any]:
    if _chat is None:
        raise HTTPException(
            status_code=503,
            detail="No LLM backend configured. Set OPENAI_API_KEY and OPENAI_MODEL environment variables.",
        )

    filename = file.filename or ""
    content_type = file.content_type or ""
    if not filename.lower().endswith(".pdf") and content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Uploaded file must be a PDF.")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        pdf_path = tmp_dir / "input.pdf"
        tei_path = tmp_dir / "output.tei.xml"
        alto_path = tmp_dir / "output.alto.xml"

        pdf_path.write_bytes(file.file.read())

        try:
            run_grobid(
                pdf_path,
                tei_path,
                grobid_url=_grobid_url,
                timeout=_grobid_timeout,
                parser=_parser,
            )
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"GROBID processing failed: {exc}") from exc

        try:
            run_pdfalto(pdf_path, alto_path, pdfalto_bin=DEFAULT_PDFALTO_BIN)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"pdfalto processing failed: {exc}") from exc

        paths = DocumentPaths(
            record_id="request",
            pdf_path=pdf_path,
            xml_path=Path(""),
            tei_path=tei_path,
            alto_path=alto_path,
            prediction_path=tmp_dir / "prediction.json",
        )

        try:
            context = build_document_context(paths)
            prediction = build_prediction(context, _chat, per_document_llm_workers=5)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Pipeline processing failed: {exc}") from exc

    return prediction


app.include_router(router)
