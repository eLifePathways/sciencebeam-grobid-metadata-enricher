.PHONY: install lint format test serve serve-reload build start stop logs logs-api clean \
        with-langfuse-start with-langfuse-stop with-langfuse-logs with-langfuse-clean \
        with-phoenix-start with-phoenix-stop with-phoenix-logs with-phoenix-clean \
        benchmark-build benchmark benchmark-train-predict benchmark-train-score benchmark-train \
        sciencebeam-start sciencebeam-stop sciencebeam-patch-figure-model benchmark-cross-parser \
        show-regressions show-improvements \
        benchmark-train-predict-grobid benchmark-train-score-grobid benchmark-train-grobid benchmark-train-rescore-grobid

-include .env
export

VENV := .venv
HOST ?= 127.0.0.1
PORT ?= 8000

BENCHMARK_MODE ?= smoke
BENCHMARK_RUN  ?= local
PARSER         ?= grobid

SHOW_RUN    ?= train/local-grobid
SHOW_CORPUS ?=
SHOW_METRIC ?=


.require-%:
	@if [ -z "$($(*))" ]; then \
		echo "Error: $* is required. Usage: make $(@:.require-%=%) $*=<value>"; \
		exit 1; \
	fi


install:
	uv sync --extra dev --extra bench

lint:
	$(VENV)/bin/ruff check \
		src/ \
		tests/ \
		benchmarks/
	$(VENV)/bin/mypy \
		src/ \
		tests/ \
		benchmarks/
	$(VENV)/bin/pylint \
		src/ \
		tests/ \
		benchmarks/

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

logs-api:
	docker compose logs -f api

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


grobid-start:
	docker compose up -d --wait grobid


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


sciencebeam-start:
	docker compose --profile sciencebeam up -d --wait sciencebeam-parser
	@echo "ScienceBeam Parser ready at http://localhost:8071/api"

sciencebeam-stop:
	docker compose --profile sciencebeam stop sciencebeam-parser


# Cross-parser benchmark: runs the smoke benchmark twice (grobid then
# sciencebeam) into sibling run dirs so report.md outputs can be diffed
# directly. Outputs land in benchmarks/runs/$(BENCHMARK_RUN)-grobid and
# benchmarks/runs/$(BENCHMARK_RUN)-sciencebeam.
benchmark-cross-parser: grobid-start benchmark-build sciencebeam-start
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


benchmark-train-predict-grobid: grobid-start
	docker compose --profile benchmark run --rm benchmark \
		python -m benchmarks.predict \
			--config benchmarks/bench-train.yaml \
			--mode   $(BENCHMARK_MODE) \
			--out    benchmarks/runs/train/$(BENCHMARK_RUN)-grobid

benchmark-train-score-grobid:
	docker compose --profile benchmark run --rm --no-deps benchmark \
		python -m benchmarks.score \
			--run    benchmarks/runs/train/$(BENCHMARK_RUN)-grobid \
			--config benchmarks/bench-train.yaml \
			--out    benchmarks/runs/train/$(BENCHMARK_RUN)-grobid/report.md
	@cat benchmarks/runs/train/$(BENCHMARK_RUN)-grobid/report.md

benchmark-train-grobid: benchmark-build benchmark-train-predict-grobid benchmark-train-score-grobid

# Force a full re-predict + re-score (use when adding new score fields requires
# fresh per-document rows, e.g. after adding title_edit_sim).
benchmark-train-rescore-grobid:
	rm -f benchmarks/runs/train/$(BENCHMARK_RUN)-grobid/per_document.jsonl
	$(MAKE) benchmark-train-grobid


# Find and export regression/improvement cases for a given metric and corpus.
# Example: make show-regressions SHOW_METRIC=abstract_edit_sim SHOW_CORPUS=ore
# Override run dir: make show-regressions SHOW_RUN=train/local SHOW_METRIC=title_match SHOW_CORPUS=biorxiv
show-regressions: .require-SHOW_METRIC .require-SHOW_CORPUS
	$(VENV)/bin/python -m benchmarks.show_cases \
		--run    benchmarks/runs/$(SHOW_RUN) \
		--metric $(SHOW_METRIC) \
		--corpus $(SHOW_CORPUS) \
		--mode   regression

show-improvements: .require-SHOW_METRIC .require-SHOW_CORPUS
	$(VENV)/bin/python -m benchmarks.show_cases \
		--run    benchmarks/runs/$(SHOW_RUN) \
		--metric $(SHOW_METRIC) \
		--corpus $(SHOW_CORPUS) \
		--mode   improvement
