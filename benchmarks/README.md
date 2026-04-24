# Benchmarks

Two-stage benchmarking harness for the Grobid + LLM metadata extraction pipeline.

## Layout

- `bench.yaml`: frozen benchmark contract (dataset revision, metrics, bootstrap params, sampling rules).
- `manifest.py`: pulls the parquet shard(s) from the HF dataset
  `elifepathways/sciencebeam-v2-benchmarking` and materialises `{pdf,xml}` files to disk.
- `predict.py`: runs Grobid, pdfalto, and the LLM on each record; writes `per_document.jsonl`,
  `run_record.json`, `errors.json`. LLM predictions are cached on disk per record, so
  re-running with the same model is free at the LLM layer.
- `score.py`: consumes `per_document.jsonl`; computes overall and per-corpus means, 95%
  bootstrap CIs (`scipy.stats.bootstrap`, percentile), paired Wilcoxon p-values (LLM vs
  Grobid on the same run, and LLM vs a named baseline run on matching records). Emits
  `metrics.json` and a Markdown `report.md`.
- `tests/`: unit tests for `score.py`.

## Local invocation

```bash
uv sync --extra bench --extra dev   # one-time: install deps

# Smoke (50 records, ~5 min wall-clock when Grobid + pdfalto are on PATH)
export HF_TOKEN=...
export AOAI_POOL_PATH=/path/to/aoai_pool.json
export PDFALTO_BIN=/path/to/pdfalto           # or rely on the Dockerfile-extracted binary
uv run python -m benchmarks.predict --config benchmarks/bench.yaml --mode smoke --out benchmarks/runs/$(git rev-parse --short HEAD)
uv run python -m benchmarks.score   --config benchmarks/bench.yaml --run   benchmarks/runs/$(git rev-parse --short HEAD)
```

Add `--baseline benchmarks/runs/<other-sha>` to get paired deltas and Wilcoxon p-values.

## CI

`.github/workflows/benchmark.yml` runs on:
- `workflow_dispatch` (choose smoke/full)
- PR labeled `benchmark:smoke` (smoke only; avoids LLM cost on every PR)
- `schedule` nightly (full)
- `push` to `main` (full, publishes baseline)

Required GitHub Actions secrets:
- `HF_TOKEN`: read access to the benchmark dataset.
- `AOAI_POOL_JSON`: contents of the AoaiPool JSON file.

Grobid runs as a services container (`lfoppiano/grobid:0.9.0-crf`, same tag as
`compose.yml` and `Dockerfile`). `pdfalto` is extracted from that image at runtime via
`docker cp`; no separate install recipe needed.

### Grobid output caching

TEI and ALTO files produced by Grobid and pdfalto are cached in GitHub Actions across
runs. The cache key is tied to the hash of `benchmark.yml`, which contains the Grobid
image tag — so bumping the image tag automatically invalidates the cache and forces a
fresh parse of all documents. The Grobid request timeout in CI is set to 360 s
(`GROBID_TIMEOUT=360`) to accommodate cold-start JVM/model load on the first few requests.


## Extending to more corpora

Add a new parquet to the HF dataset (e.g. `ore.parquet`), then in `bench.yaml`:

```yaml
dataset:
  files:
    scielo_preprints: scielo_preprints.parquet
    ore: ore.parquet
corpora: [scielo_preprints, ore]
sampling:
  full:  {scielo_preprints: 1000, ore: 199}
  smoke: {scielo_preprints: 50,   ore: 10}
```

If the new corpus's gold format isn't OAI-DC, extend `predict.process_one`'s gold-parsing
branch (currently hardcoded to `extract_oai_dc`) to dispatch on `row["corpus"]`.
