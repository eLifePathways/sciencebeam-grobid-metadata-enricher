# grobid-metadata-enricher

## What this pipeline does (high level)
- Runs an upstream PDF parser (**Grobid** or **ScienceBeam Parser**) plus pdfalto, then uses an LLM to **re‑extract header metadata** (title/authors/affiliations) from layout lines.
- Uses the LLM to **select the true abstract** from TEI + layout candidates, plus an OCR‑cleanup pass to improve noisy text.
- **Translates and unions keywords** across languages using the LLM.
- Adds **deterministic SciELO identifiers** (DOI + URL) from the record id.
- Produces per‑document metrics and a root‑cause summary against OAI‑DC XML.

A small pipeline that runs an upstream parser + pdfalto + LLM enrichment and evaluates against OAI-DC XML.

## Parser backends

The pipeline supports two upstream parsers, both exposing `/processFulltextDocument` and returning TEI:

| Parser | Image | URL | Notes |
| --- | --- | --- | --- |
| `grobid` (default) | `lfoppiano/grobid:0.9.0-crf` | `http://localhost:8070/api` | linux/amd64 + arm64 |
| `sciencebeam` | `ghcr.io/elifepathways/sciencebeam-parser:0.1.18` | `http://localhost:8071/api` (compose) | linux/amd64 only; slow startup, longer healthcheck `start_period` |

Select the backend with `--parser` (CLI), `PARSER=` (env), or `parser:` in `bench.yaml`. TEI and predictions are namespaced by parser (`tei/<parser>/...`, `predictions/<parser>/...`) so a follow-up run with the other backend does not silently re-use cached outputs.

## Prerequisites
- Python 3.10+ (3.11+ recommended)
- An upstream parser server running and reachable: Grobid at `http://localhost:8070/api`, or ScienceBeam Parser at `http://localhost:8071/api`
- `pdfalto` installed and on PATH (or pass `--pdfalto /path/to/pdfalto`)
- One of:
  - AOAI pool JSON (round-robin backends), or
  - OpenAI API key + model name

### Grobid (Docker)
```bash
docker run -d --rm --name grobid -p 8070:8070 grobid/grobid:0.7.2
```

### ScienceBeam Parser (Docker)
```bash
docker run -d --rm --name sciencebeam-parser \
  --platform linux/amd64 \
  -p 8071:8070 \
  ghcr.io/elifepathways/sciencebeam-parser:0.1.18
```
First-request model loading can take several minutes; the compose healthcheck reflects this. Then point the pipeline at it: `--parser sciencebeam --grobid-url http://localhost:8071/api`.

### Grobid (local install)
If you prefer a local install, follow the official Grobid instructions:
1) Install Java (Grobid requires Java 8+).
2) Clone the Grobid repo and build:
   ```
   git clone https://github.com/kermitt2/grobid.git
   cd grobid
   ./gradlew clean install
   ```
3) Start the service:
   ```
   ./gradlew run
   ```
By default it serves at `http://localhost:8070/api`.

### pdfalto
Install pdfalto and ensure it is on PATH. Example path used in this repo:
```
/Users/leon/bin/pdfalto
```

You can choose which LLM cloud endpoint to use. We plan to support local vLLM deployments later, but for the benchmark here, use OpenAI.

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
Set environment variables or pass CLI flags:
```
export OPENAI_API_KEY=...
export OPENAI_MODEL=gpt-4o-mini
```

Or:
```
--openai-api-key ... --openai-model gpt-4o-mini
```

## Install
From this folder:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

If you plan to use parquet input, install pyarrow in the same environment:
```bash
pip install pyarrow
```

If you do not want to install, you can run with:
```bash
env PYTHONPATH=./src python3 -m grobid_metadata_enricher ...
```

## Run
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
python3 -m grobid_metadata_enricher \
  --manifest /path/to/manifest.csv \
  --pool /path/to/aoai_pool.json \
  --output-dir /path/to/output \
  --pdfalto /path/to/pdfalto \
  --workers 20 \
  --per-document-llm-workers 5 \
  --llm-concurrency 20
```

Run (OpenAI API key + model, CSV manifest):
```bash
python3 -m grobid_metadata_enricher \
  --manifest /path/to/manifest.csv \
  --openai-api-key $OPENAI_API_KEY \
  --openai-model gpt-4o-mini \
  --output-dir /path/to/output \
  --pdfalto /path/to/pdfalto \
  --workers 20 \
  --per-document-llm-workers 5 \
  --llm-concurrency 20
```

Run (parquet input):
```bash
python3 -m grobid_metadata_enricher \
  --manifest /path/to/scielo_preprints.parquet \
  --pool /path/to/aoai_pool.json \
  --output-dir /path/to/output \
  --pdfalto /path/to/pdfalto
```

Note: parquet input requires `pyarrow`:
```bash
pip install pyarrow
```

Run (ScienceBeam Parser instead of Grobid):
```bash
make sciencebeam-start          # starts the sidecar and patches its bundled figure model
python3 -m grobid_metadata_enricher \
  --manifest /path/to/manifest.csv \
  --pool /path/to/aoai_pool.json \
  --output-dir /path/to/output \
  --pdfalto /path/to/pdfalto \
  --parser sciencebeam
```
Same flags as the Grobid runs above plus `--parser sciencebeam`. `--grobid-url` is optional and resolves from the parser: `localhost:8070/api` for grobid, `localhost:8071/api` for sciencebeam. Pass it explicitly only when pointing at a non-default host. `--parser` also accepts the `PARSER` env var. TEI and predictions are namespaced per parser at `<output-dir>/tei/<parser>/...` and `<output-dir>/predictions/<parser>/...`, so a follow-up run with the other backend does not re-use cached outputs. `make sciencebeam-stop` shuts the sidecar down.

Probe a parser directly with a single PDF (no LLM, returns TEI on stdout):
```bash
# Grobid
docker compose up -d --wait grobid
curl -F "input=@paper.pdf" http://localhost:8070/api/processFulltextDocument

# ScienceBeam Parser
make sciencebeam-start
curl -F "input=@paper.pdf" http://localhost:8071/api/processFulltextDocument
```

Key flags:
- `--parser`: upstream PDF parser, `grobid` (default) or `sciencebeam`
- `--grobid-url`: parser endpoint URL (defaults to `localhost:8070/api` for grobid, `localhost:8071/api` for sciencebeam; honour `GROBID_URL` env)
- `--workers`: number of docs processed in parallel
- `--per-document-llm-workers`: LLM calls per doc in parallel (after Grobid/pdfalto)
- `--llm-concurrency`: global LLM in-flight cap
- `--pdfalto-start/--pdfalto-end`: page range for layout extraction
- `--rerun`: ignore cached outputs and recompute

## Outputs
`--output-dir` will contain:
- `tei/`: Grobid TEI
- `alto/`: pdfalto ALTO
- `predictions/`: JSON per record
- `per_document.jsonl`: per-record predictions + metrics
- `metrics.json`: aggregated metrics
- `root_causes.md`: failure analysis summary
- `errors.json`: errors per record (if any)

## Benchmarking
Use a fixed manifest for reproducibility. Example (SciELO 200 sample):
```bash
python3 -m grobid_metadata_enricher \
  --manifest /path/to/manifest_200.csv \
  --pool /path/to/aoai_pool.json \
  --output-dir /path/to/run_200 \
  --pdfalto /path/to/pdfalto \
  --workers 20 --per-document-llm-workers 5 --llm-concurrency 20
```

The aggregated metrics are written to `metrics.json`.

### Cross-parser benchmark

Run the same smoke benchmark through both parser backends and compare:
```bash
make benchmark-cross-parser BENCHMARK_RUN=cross
```
This spins up both `grobid` and `sciencebeam-parser` services, runs predict + score for each, and writes:
- `benchmarks/runs/cross-grobid/report.md`
- `benchmarks/runs/cross-sciencebeam/report.md`

Or run a single backend:
```bash
make benchmark PARSER=sciencebeam BENCHMARK_RUN=sb-smoke
```

## LLM observability with Langfuse

The stack can optionally run [Langfuse](https://langfuse.com) locally to trace all LLM calls via OpenTelemetry.

### Start the full stack (API + Langfuse)

```bash
make with-langfuse-start
```

This brings up the API alongside Langfuse and its dependencies (ClickHouse, PostgreSQL, Redis, MinIO). Once ready:
- API: http://localhost:8000
- Langfuse UI: http://localhost:3000 — log in with `admin@local.dev` / `password`

The API and benchmark services automatically send OTEL traces to Langfuse when started this way.

### Other targets

```bash
make with-langfuse-stop    # stop all containers (keeps volumes)
make with-langfuse-logs    # tail logs for the full stack
make with-langfuse-clean   # stop and delete all volumes
```

### Run benchmarks with tracing

```bash
make with-langfuse-start
make benchmark
```

Traces appear in the Langfuse UI under the pre-provisioned **sciencebeam** project.

## Notes
- Grobid can return 503 under load. Re-run with `--rerun` or lower `--workers` if that happens.
- Results depend on LLM backend behavior; parallelism can change output order across backends.
