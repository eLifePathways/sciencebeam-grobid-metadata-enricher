.PHONY: install lint format test test-bench check serve serve-reload build start stop logs logs-api clean shell \
        with-langfuse-start with-langfuse-stop with-langfuse-logs with-langfuse-clean \
        with-phoenix-start with-phoenix-stop with-phoenix-logs with-phoenix-clean \
        benchmark-build benchmark benchmark-score benchmark-compare \
        benchmark-train-predict benchmark-train-score benchmark-train

-include .env
export

VENV := .venv
HOST ?= 127.0.0.1
PORT ?= 8000

BENCHMARK_MODE ?= smoke
BENCHMARK_RUN  ?= local

install:
	uv sync --extra dev

lint:
	$(VENV)/bin/ruff check src/ tests/
	$(VENV)/bin/mypy src/
	$(VENV)/bin/pylint src/

format:
	$(VENV)/bin/ruff check --fix src/ tests/
	$(VENV)/bin/ruff format src/ tests/

test:
	$(VENV)/bin/pytest tests/

test-bench:
	$(VENV)/bin/pytest benchmarks/tests/

check: lint test test-bench

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

shell:
	docker compose exec api bash

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
	docker compose --profile benchmark run --rm benchmark \
		python -m benchmarks.predict \
			--config benchmarks/bench.yaml \
			--mode   $(BENCHMARK_MODE) \
			--out    benchmarks/runs/$(BENCHMARK_RUN)
	docker compose --profile benchmark run --rm --no-deps benchmark \
		python -m benchmarks.score \
			--run    benchmarks/runs/$(BENCHMARK_RUN) \
			--config benchmarks/bench.yaml \
			--out    benchmarks/runs/$(BENCHMARK_RUN)/report.md
	@cat benchmarks/runs/$(BENCHMARK_RUN)/report.md


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
