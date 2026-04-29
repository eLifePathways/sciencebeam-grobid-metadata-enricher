# grobid-metadata-enricher

A pipeline that takes a scientific paper PDF (plus optional OAI-DC or JATS XML) and produces structured, LLM-enriched metadata — title, authors, affiliations, abstract, keywords, body sections, figures, tables, and references.

## What this pipeline does

**Input**: a PDF + (optionally) an OAI-DC or JATS XML file per paper.

**Steps, in order**:

1. **Grobid + pdfalto** — parse the PDF into structured TEI XML (Grobid) and layout text lines (pdfalto/ALTO). Both run locally; no external service needed when using Docker.
2. **Header re-extraction** (LLM) — extract title, authors, affiliations, language, and identifiers from the layout lines. This catches cases where Grobid's TEI header is wrong or incomplete.
3. **Abstract selection** (LLM) — pick the best abstract from TEI candidates and raw layout text; run an OCR-cleanup pass on noisy text.
4. **Keyword translation** (LLM) — collect keywords from all languages in the TEI and union them into a single deduplicated list.
5. **Content extraction** (LLM, 3 parallel passes) — extract body sections, figure captions, and table captions from the full-document ALTO text. Three non-overlapping windows (head / middle / tail) run concurrently to reduce wall time.
6. **Reference enrichment** (Crossref API, no LLM) — for every reference Grobid found with a title but no DOI, look up the DOI via Crossref (Jaccard-thresholded match, up to 80 lookups, 5 parallel).
7. **SciELO identifiers** — derive DOI and SciELO URL deterministically from the record ID.
8. **Evaluation** — compare predictions against the gold XML and emit per-field recall/match scores. Gated metrics (body section recall, figure/table caption recall, reference recall) only appear when the corresponding gold key exists.

Exposed as a **FastAPI service** (primary usage) and as a **CLI batch processor**.

## Prerequisites
- Python 3.10+ (3.11+ recommended)
- One of:
  - AOAI pool JSON (round-robin backends), or
  - OpenAI API key + model name

> **Note**: when using Docker Compose (`make start`), Grobid and pdfalto are bundled in the image — no separate setup needed. LLM credentials are required regardless of how you run the pipeline.

### AOAI pool JSON
This is a pool of endpoint configs; the runner round‑robins across them.

Example file format:
```json
[
  {
    "id": "backend-1",
    "endpoint": "https://YOUR-RESOURCE.openai.azure.com",
    "deployment": "gpt-4o-mini",
    "apiKey": "YOUR_KEY",
    "apiVersion": "2024-02-15-preview"
  }
]
```

### OpenAI API key + model
```bash
export OPENAI_API_KEY=...
export OPENAI_MODEL=gpt-4o-mini
```

## Quick start (Docker Compose)

Copy `.env` and set your LLM credentials:
```bash
# .env
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-4o-mini
```

Build and start the API (Grobid + pdfalto are bundled in the image):
```bash
make start        # build + start; API at http://localhost:8000
make stop         # stop containers (keeps volumes)
make logs         # tail logs (all services)
make logs-api     # tail logs (API only — less grobid noise)
make clean        # stop and delete volumes
```

The API docs are at http://localhost:8000/api/docs.

## Install (local development)

```bash
make install
# or manually:
uv sync --extra dev --extra bench
```

Common dev commands:
```bash
make lint           # ruff + mypy + pylint
make format         # ruff --fix + ruff format
make test           # pytest (unit + benchmark tests)
make serve          # run the API locally (no Docker)
make serve-reload   # run the API locally with auto-reload
```

## CLI batch mode

The CLI requires a running Grobid server and `pdfalto` on PATH (or via `--pdfalto`).
In Docker, both are bundled automatically. For local use, set them up manually:

### Grobid (Docker — recommended for local CLI use)
```bash
docker run -d --rm --name grobid -p 8070:8070 lfoppiano/grobid:0.9.0-crf
```

### Grobid (local install)
Follow the [official Grobid instructions](https://grobid.readthedocs.io/en/latest/Install-Grobid/). By default it serves at `http://localhost:8070/api`.

### pdfalto
Install [pdfalto](https://github.com/kermitt2/pdfalto) and ensure it is on PATH, or pass its location via `--pdfalto /path/to/pdfalto`.

You can provide either:
1) a manifest CSV with columns:
```
record_id,pdf_path,xml_path
```
2) a parquet file with columns:
```
id (string), pdf (bytes), xml (string)
```

Example:
```
oai_ops.preprints.scielo.org_preprint_123,/path/to/123.pdf,/path/to/123.xml
```

Run (AOAI pool, CSV manifest):
```bash
grobid-metadata-enricher \
  --manifest /path/to/manifest.csv \
  --pool /path/to/aoai_pool.json \
  --output-dir /path/to/output \
  --workers 20 \
  --per-document-llm-workers 5 \
  --llm-concurrency 20
```

Run (OpenAI API key + model, CSV manifest):
```bash
grobid-metadata-enricher \
  --manifest /path/to/manifest.csv \
  --openai-api-key $OPENAI_API_KEY \
  --openai-model gpt-4o-mini \
  --output-dir /path/to/output \
  --workers 20 \
  --per-document-llm-workers 5 \
  --llm-concurrency 20
```

Run (parquet input):
```bash
grobid-metadata-enricher \
  --manifest /path/to/scielo_preprints.parquet \
  --pool /path/to/aoai_pool.json \
  --output-dir /path/to/output \
  --pdfalto /path/to/pdfalto
```

Note: parquet input requires `pyarrow`:
```bash
pip install pyarrow
```

### ScienceBeam Parser backend

To use ScienceBeam Parser instead of GROBID, start its sidecar first:

```bash
make sciencebeam-start   # start the ScienceBeam Parser container (port 8071)
make sciencebeam-stop    # stop it
```

`sciencebeam-start` also applies a one-time patch that fixes a missing `config.json` in the upstream image (`0.1.18`) which causes HTTP 500 errors on every request. The patch is idempotent — re-running is safe.

Then pass `--parser sciencebeam` (or set `PARSER=sciencebeam` for benchmarks):

```bash
grobid-metadata-enricher \
  --parser sciencebeam \
  --manifest /path/to/manifest.csv \
  --openai-api-key $OPENAI_API_KEY \
  --openai-model gpt-4o-mini \
  --output-dir /path/to/output
```

### Key flags
- `--parser`: upstream PDF parser, `grobid` (default) or `sciencebeam`
- `--grobid-url`: parser endpoint URL (defaults to `localhost:8070/api` for grobid, `localhost:8071/api` for sciencebeam; honours `GROBID_URL` env)
- `--workers`: number of docs processed in parallel
- `--per-document-llm-workers`: LLM calls per doc in parallel (after Grobid/pdfalto)
- `--llm-concurrency`: global LLM in-flight cap
- `--pdfalto-start/--pdfalto-end`: page range for layout extraction
- `--rerun`: ignore cached outputs and recompute

## Outputs
`--output-dir` will contain:
- `tei/`: Grobid TEI XML
- `alto/`: pdfalto ALTO layout files
- `predictions/`: JSON per record — includes header fields (title, authors, affiliations, abstract, keywords, language, identifiers) **and** content fields (body_sections, figure_captions, table_captions, reference_dois, reference_titles)
- `per_document.jsonl`: per-record predictions + metrics + LLM token usage breakdown by stage
- `metrics.json`: aggregated metrics
- `root_causes.md`: failure analysis summary
- `errors.json`: errors per record (if any)

## Benchmarking

Benchmarks run via Docker Compose (pdfalto is bundled in the image). Set `HF_TOKEN` in `.env`, then:

```bash
make benchmark                           # smoke run (25 docs/corpus, fast — default)
make benchmark BENCHMARK_MODE=full       # full run (all docs)
make benchmark BENCHMARK_RUN=my-run     # custom output directory name
make benchmark PARSER=sciencebeam       # use ScienceBeam Parser instead of GROBID
```

Outputs land in `benchmarks/runs/<BENCHMARK_RUN>/` (default: `benchmarks/runs/local/`). The aggregated report is written to `report.md` in that directory and printed at the end of the run.

### Cross-parser comparison

Run the smoke benchmark against both GROBID and ScienceBeam Parser back-to-back and diff their `report.md` outputs:

```bash
make benchmark-cross-parser                        # default: smoke mode
make benchmark-cross-parser BENCHMARK_MODE=full    # full run
```

Outputs land in `benchmarks/runs/<BENCHMARK_RUN>-grobid/` and `benchmarks/runs/<BENCHMARK_RUN>-sciencebeam/`.

### Training-set benchmark (local only)

To benchmark on the training split without polluting the validation split used by CI:

```bash
make benchmark-train                     # build + predict + score
make benchmark-train BENCHMARK_MODE=full
```

## LLM observability

All LLM calls are instrumented with OpenTelemetry. By default tracing is off; add one of the backends below to enable it. Choose based on your needs:

| | Arize Phoenix | Langfuse |
|---|---|---|
| **Best for** | Quick local trace inspection | Cost tracking, prompt management, long-term analysis |
| **Stack** | Single container | PostgreSQL + ClickHouse + Redis + MinIO + worker |
| **Start-up** | Seconds | ~1 min (many services) |
| **UI** | http://localhost:6006 | http://localhost:3000 |

### Arize Phoenix — lightweight, trace-first

Pick this if you want to inspect prompt/response pairs and latency with minimal setup overhead.

```bash
make with-phoenix-start    # API + Phoenix (single extra container)
make with-phoenix-stop
make with-phoenix-logs
make with-phoenix-clean    # removes volumes
```

UI at http://localhost:6006 under the **sciencebeam** project.

### Langfuse — full LLM platform

Pick this if you need cost tracking, prompt versioning, user feedback, or evaluations alongside traces.

```bash
make with-langfuse-start   # API + full Langfuse stack
make with-langfuse-stop
make with-langfuse-logs
make with-langfuse-clean   # removes volumes
```

UI at http://localhost:3000 — log in with `admin@local.dev` / `password`. Traces appear under the pre-provisioned **sciencebeam** project.

### Run benchmarks with tracing

Start either backend first, then run the benchmark:

```bash
make with-phoenix-start    # or with-langfuse-start
make benchmark
```

## Notes
- Grobid can return 503 under load. Re-run with `--rerun` or lower `--workers` if that happens.
- Results depend on LLM backend behavior; parallelism can change output order across backends.
- Content extraction (body/figures/tables/references) runs on all supported corpora — all are now JATS. Use `make benchmark-train` locally (see [Training-set benchmark](#training-set-benchmark-local-only)) to avoid polluting the validation split used by CI.
