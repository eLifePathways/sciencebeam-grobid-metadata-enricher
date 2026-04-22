.PHONY: install lint format test serve serve-reload build start stop logs benchmark benchmark-cached langfuse-start langfuse-stop langfuse-logs

-include .env
export

VENV := .venv
HOST ?= 127.0.0.1
PORT ?= 8000

BENCHMARK_MODE ?= smoke
BENCHMARK_RUN  ?= local
CACHE_DIR      ?= .llm_cache

install:
	uv sync --extra dev --extra observe --extra cache

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


# Like benchmark but with a persistent LLM response cache mounted at CACHE_DIR.
# First run warms the cache; subsequent runs skip LLM calls entirely.
benchmark-cached: benchmark-build
	mkdir -p $(CACHE_DIR)
	docker compose --profile benchmark run --rm \
		--volume $(PWD)/$(CACHE_DIR):/llm_cache \
		benchmark \
		python -m benchmarks.predict \
			--config benchmarks/bench.yaml \
			--mode   $(BENCHMARK_MODE) \
			--out    benchmarks/runs/$(BENCHMARK_RUN) \
			--cache-dir /llm_cache
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

# Local Langfuse instance (observability UI, no cloud account needed, unlimited).
# Pre-provisioned keys — add these to your .env:
#   LANGFUSE_HOST=http://langfuse:3000   (Docker-to-Docker, same network)
#   LANGFUSE_PUBLIC_KEY=pk-lf-local
#   LANGFUSE_SECRET_KEY=sk-lf-local
# UI: http://localhost:3000  login: admin@local.dev / password
langfuse-start:
	docker compose -f compose.langfuse.yml up -d --wait
	@echo "Langfuse ready at http://localhost:3000  (admin@local.dev / password)"

langfuse-stop:
	docker compose -f compose.langfuse.yml down

langfuse-logs:
	docker compose -f compose.langfuse.yml logs -f
