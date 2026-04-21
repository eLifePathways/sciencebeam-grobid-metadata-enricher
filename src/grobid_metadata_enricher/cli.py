from __future__ import annotations

import argparse
from pathlib import Path

from .clients import (
    DEFAULT_GROBID_URL,
    DEFAULT_OPENAI_API_KEY,
    DEFAULT_OPENAI_BASE_URL,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_PDFALTO_BIN,
    DEFAULT_POOL_PATH,
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
    parser.add_argument("--grobid-url", type=str, default=DEFAULT_GROBID_URL)
    parser.add_argument("--pdfalto", type=Path, default=DEFAULT_PDFALTO_BIN)
    parser.add_argument("--pdfalto-start", type=int, default=1)
    parser.add_argument("--pdfalto-end", type=int, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--per-document-llm-workers", type=int, default=5)
    parser.add_argument("--llm-concurrency", type=int, default=20)
    parser.add_argument("--cache-dir", type=Path, default=None)
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
        grobid_url=args.grobid_url,
        pdfalto_bin=args.pdfalto,
        pdfalto_start_page=args.pdfalto_start,
        pdfalto_end_page=args.pdfalto_end,
        limit=args.limit,
        rerun=args.rerun,
        workers=args.workers,
        per_document_llm_workers=args.per_document_llm_workers,
        llm_concurrency=args.llm_concurrency,
        cache_dir=args.cache_dir,
    )
    run_pipeline(settings)
