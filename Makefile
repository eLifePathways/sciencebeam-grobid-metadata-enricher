.PHONY: install lint format test serve serve-reload build start stop logs clean \
        with-langfuse-start with-langfuse-stop with-langfuse-logs with-langfuse-clean \
        with-phoenix-start with-phoenix-stop with-phoenix-logs with-phoenix-clean \
        benchmark-build benchmark benchmark-train-predict benchmark-train-score benchmark-train \
        sciencebeam-start sciencebeam-stop sciencebeam-patch-figure-model benchmark-cross-parser

-include .env
export

VENV := .venv
HOST ?= 127.0.0.1
PORT ?= 8000

BENCHMARK_MODE ?= smoke
BENCHMARK_RUN  ?= local
PARSER         ?= grobid

install:
	uv sync --extra dev --extra bench

lint:
	$(VENV)/bin/ruff check \
		src/ \
		tests/ \
		benchmarks/
	$(VENV)/bin/mypy src/
	$(VENV)/bin/pylint src/

format:
	$(VENV)/bin/ruff check --fix \
		src/ \
		tests/ \
		benchmarks/
	$(VENV)/bin/ruff format \
		src/ \
		tests/ \
		benchmarks/

test:
	$(VENV)/bin/pytest \
		tests/ \
		benchmarks/tests/

serve:
	$(VENV)/bin/uvicorn grobid_metadata_enricher.api:app --host $(HOST) --port $(PORT)

serve-reload:
	$(VENV)/bin/uvicorn grobid_metadata_enricher.api:app --host $(HOST) --port $(PORT) --reload

build:
	docker compose build

start:
	docker compose up -d --wait
	@echo "Ready. API available at http://localhost:8000"

stop:
	docker compose down

logs:
	docker compose logs -f

clean:
	docker compose down -v


COMPOSE_LANGFUSE := docker compose -f compose.yml -f compose.langfuse.yml

with-langfuse-start:
	$(COMPOSE_LANGFUSE) up -d --wait
	@echo "Ready. API at http://localhost:8000, Langfuse at http://localhost:3000 (admin@local.dev / password)"

with-langfuse-stop:
	$(COMPOSE_LANGFUSE) down

with-langfuse-logs:
	$(COMPOSE_LANGFUSE) logs -f

with-langfuse-clean:
	$(COMPOSE_LANGFUSE) down -v


COMPOSE_PHOENIX := docker compose -f compose.yml -f compose.phoenix.yml

with-phoenix-start:
	$(COMPOSE_PHOENIX) up -d --wait
	@echo "Ready. API at http://localhost:8000, Phoenix at http://localhost:6006"

with-phoenix-stop:
	$(COMPOSE_PHOENIX) down

with-phoenix-logs:
	$(COMPOSE_PHOENIX) logs -f

with-phoenix-clean:
	$(COMPOSE_PHOENIX) down -v


benchmark-build:
	docker compose --profile benchmark build benchmark

# Run benchmark tests via docker compose (Dockerfile.bench bundles pdfalto).
# Modes: smoke (25 docs/corpus, fast) or full (all docs).
# Override with: make benchmark BENCHMARK_MODE=full
# Override run dir with: make benchmark BENCHMARK_RUN=my-run
benchmark: benchmark-build
	@if [ "$(PARSER)" = "sciencebeam" ]; then \
		docker compose --profile sciencebeam up -d --wait sciencebeam-parser; \
	else \
		docker compose up -d --wait grobid; \
	fi
	docker compose --profile benchmark run --rm \
		-e PARSER=$(PARSER) \
		-e GROBID_URL=$$( [ "$(PARSER)" = "sciencebeam" ] && echo http://sciencebeam-parser:8070/api || echo http://grobid:8070/api ) \
		benchmark \
		python -m benchmarks.predict \
			--config benchmarks/bench.yaml \
			--mode   $(BENCHMARK_MODE) \
			--parser $(PARSER) \
			--out    benchmarks/runs/$(BENCHMARK_RUN)
	docker compose --profile benchmark run --rm --no-deps benchmark \
		python -m benchmarks.score \
			--run    benchmarks/runs/$(BENCHMARK_RUN) \
			--config benchmarks/bench.yaml \
			--out    benchmarks/runs/$(BENCHMARK_RUN)/report.md
	@cat benchmarks/runs/$(BENCHMARK_RUN)/report.md


# Spin up the ScienceBeam Parser sidecar (off by default; profile-gated in
# compose.yml). Patches the missing figure-model config.json that ships
# broken in 0.1.18 (see sciencebeam-patch-figure-model below) so biorxiv
# requests don't 500 on first contact. Use before `make benchmark
# PARSER=sciencebeam` so the benchmark container can reach it.
sciencebeam-start: sciencebeam-patch-figure-model
	@echo "ScienceBeam Parser ready at http://localhost:8071/api"

sciencebeam-stop:
	docker compose --profile sciencebeam stop sciencebeam-parser

# Repair the upstream image bug: ghcr.io/elifepathways/sciencebeam-parser:0.1.18
# ships the figure-model directory at /root/.cache/sciencebeam-parser/downloads/
# without its config.json, so any request that triggers figure extraction (i.e.
# every biorxiv preprint) fails with HTTP 500 + 'FileNotFoundError: ... config.json'.
# This target downloads the upstream tarball and copies its files into the
# running container's cache volume. Idempotent: if config.json is already
# present the docker cp is a no-op overwrite.
SB_FIG_URL := https://github.com/eLifePathways/sciencebeam-models/releases/download/biorxiv-grobid/2021-05-11-delft-grobid-figure-biorxiv-10k-auto-v0.0.18-train-1865-e219.tar.gz
SB_FIG_CACHE_HASH := e3172c0f10622fe2015a65127d5c0fd9-2021-05-11-delft-grobid-figure-biorxiv-10k-auto-v0.0.18-train-1865-e219
SB_FIG_CONTAINER := sciencebeam-grobid-metadata-enricher-sciencebeam-parser-1
sciencebeam-patch-figure-model:
	docker compose --profile sciencebeam up -d --wait sciencebeam-parser
	@if docker exec $(SB_FIG_CONTAINER) test -f /root/.cache/sciencebeam-parser/downloads/$(SB_FIG_CACHE_HASH)/config.json; then \
		echo "ScienceBeam figure model already patched."; \
	else \
		echo "Patching missing figure-model config.json from upstream tarball..."; \
		tmpdir=$$(mktemp -d); \
		curl -sL --max-time 120 "$(SB_FIG_URL)" -o $$tmpdir/figure-model.tar.gz; \
		mkdir -p $$tmpdir/extract && tar -xzf $$tmpdir/figure-model.tar.gz -C $$tmpdir/extract; \
		docker cp $$tmpdir/extract/. $(SB_FIG_CONTAINER):/root/.cache/sciencebeam-parser/downloads/$(SB_FIG_CACHE_HASH)/; \
		rm -rf $$tmpdir; \
		echo "Patched. config.json now present in container cache."; \
	fi


# Cross-parser benchmark: runs the smoke benchmark twice (grobid then
# sciencebeam) into sibling run dirs so report.md outputs can be diffed
# directly. Outputs land in benchmarks/runs/$(BENCHMARK_RUN)-grobid and
# benchmarks/runs/$(BENCHMARK_RUN)-sciencebeam.
benchmark-cross-parser: benchmark-build sciencebeam-patch-figure-model
	docker compose up -d --wait grobid
	docker compose --profile benchmark run --rm \
		-e PARSER=grobid \
		-e GROBID_URL=http://grobid:8070/api \
		benchmark \
		python -m benchmarks.predict \
			--config benchmarks/bench.yaml \
			--mode   $(BENCHMARK_MODE) \
			--parser grobid \
			--out    benchmarks/runs/$(BENCHMARK_RUN)-grobid
	docker compose --profile benchmark run --rm --no-deps \
		benchmark \
		python -m benchmarks.score \
			--run    benchmarks/runs/$(BENCHMARK_RUN)-grobid \
			--config benchmarks/bench.yaml \
			--out    benchmarks/runs/$(BENCHMARK_RUN)-grobid/report.md
	docker compose --profile benchmark run --rm \
		-e PARSER=sciencebeam \
		-e GROBID_URL=http://sciencebeam-parser:8070/api \
		benchmark \
		python -m benchmarks.predict \
			--config benchmarks/bench.yaml \
			--mode   $(BENCHMARK_MODE) \
			--parser sciencebeam \
			--out    benchmarks/runs/$(BENCHMARK_RUN)-sciencebeam
	docker compose --profile benchmark run --rm --no-deps \
		benchmark \
		python -m benchmarks.score \
			--run    benchmarks/runs/$(BENCHMARK_RUN)-sciencebeam \
			--config benchmarks/bench.yaml \
			--out    benchmarks/runs/$(BENCHMARK_RUN)-sciencebeam/report.md
	@echo "=== GROBID benchmark ==="
	@cat benchmarks/runs/$(BENCHMARK_RUN)-grobid/report.md
	@echo "=== ScienceBeam Parser benchmark ==="
	@cat benchmarks/runs/$(BENCHMARK_RUN)-sciencebeam/report.md


benchmark-train-predict:
	docker compose --profile benchmark run --rm benchmark \
		python -m benchmarks.predict \
			--config benchmarks/bench-train.yaml \
			--mode   $(BENCHMARK_MODE) \
			--out    benchmarks/runs/train/$(BENCHMARK_RUN)

benchmark-train-score:
	docker compose --profile benchmark run --rm --no-deps benchmark \
		python -m benchmarks.score \
			--run    benchmarks/runs/train/$(BENCHMARK_RUN) \
			--config benchmarks/bench-train.yaml \
			--out    benchmarks/runs/train/$(BENCHMARK_RUN)/report.md
	@cat benchmarks/runs/train/$(BENCHMARK_RUN)/report.md

benchmark-train: benchmark-build benchmark-train-predict benchmark-train-score
