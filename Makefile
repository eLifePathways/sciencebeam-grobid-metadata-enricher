.PHONY: install lint format test serve serve-reload build start stop logs benchmark

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

# Run benchmark tests via docker compose (Dockerfile.bench bundles pdfalto).
# Modes: smoke (25 docs/corpus, fast) or full (all docs).
# Override with: make benchmark BENCHMARK_MODE=full
# Override run dir with: make benchmark BENCHMARK_RUN=my-run
benchmark:
	docker compose --profile benchmark build benchmark
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
