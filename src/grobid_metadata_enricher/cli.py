from __future__ import annotations

import argparse
from pathlib import Path

from .clients import (
    DEFAULT_OPENAI_API_KEY,
    DEFAULT_OPENAI_BASE_URL,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_PARSER,
    DEFAULT_PDFALTO_BIN,
    DEFAULT_POOL_PATH,
    SUPPORTED_PARSERS,
    resolve_parser_url,
)
from .pipeline import PipelineSettings, run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="grobid-metadata-enricher")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL_PATH)
    parser.add_argument("--openai-api-key", type=str, default=DEFAULT_OPENAI_API_KEY)
    parser.add_argument("--openai-model", type=str, default=DEFAULT_OPENAI_MODEL)
    parser.add_argument("--openai-base-url", type=str, default=DEFAULT_OPENAI_BASE_URL)
    parser.add_argument("--output-dir", type=Path, required=True)
    # default=None lets resolve_parser_url derive the URL from --parser /
    # GROBID_URL / per-parser default, so `--parser sciencebeam` works on
    # its own without an explicit `--grobid-url http://localhost:8071/api`.
    parser.add_argument(
        "--grobid-url",
        type=str,
        default=None,
        help=(
            "Parser endpoint URL. Defaults to localhost:8070 for grobid, "
            "localhost:8071 for sciencebeam (overridable with GROBID_URL env)."
        ),
    )
    parser.add_argument(
        "--parser",
        type=str,
        choices=list(SUPPORTED_PARSERS),
        default=DEFAULT_PARSER,
        help="Upstream PDF parser backend (default: %(default)s).",
    )
    parser.add_argument("--pdfalto", type=Path, default=DEFAULT_PDFALTO_BIN)
    parser.add_argument("--pdfalto-start", type=int, default=1)
    parser.add_argument("--pdfalto-end", type=int, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--per-document-llm-workers", type=int, default=5)
    parser.add_argument("--llm-concurrency", type=int, default=20)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    settings = PipelineSettings(
        manifest_path=args.manifest,
        pool_path=args.pool,
        openai_api_key=args.openai_api_key,
        openai_model=args.openai_model,
        openai_base_url=args.openai_base_url,
        output_dir=args.output_dir,
        grobid_url=resolve_parser_url(args.parser, args.grobid_url),
        parser=args.parser,
        pdfalto_bin=args.pdfalto,
        pdfalto_start_page=args.pdfalto_start,
        pdfalto_end_page=args.pdfalto_end,
        limit=args.limit,
        rerun=args.rerun,
        workers=args.workers,
        per_document_llm_workers=args.per_document_llm_workers,
        llm_concurrency=args.llm_concurrency,
    )
    run_pipeline(settings)
