.PHONY: install lint format test serve serve-reload build start stop logs logs-api clean \
        with-langfuse-start with-langfuse-stop with-langfuse-logs with-langfuse-clean \
        with-phoenix-start with-phoenix-stop with-phoenix-logs with-phoenix-clean \
        with-vllm-download with-vllm-start with-vllm-stop with-vllm-logs with-vllm-clean \
        with-vllm-phoenix-start with-vllm-phoenix-stop with-vllm-phoenix-logs with-vllm-phoenix-clean \
        benchmark-build benchmark benchmark-train-predict benchmark-train-score benchmark-train \
        sciencebeam-start sciencebeam-stop sciencebeam-patch-figure-model benchmark-cross-parser \
        show-regressions show-improvements \
        benchmark-train-grobid benchmark-validation-grobid benchmark-train-rescore-grobid \
        benchmark-train-lora benchmark-validation-lora

-include .env
export

VENV := .venv
HOST ?= 127.0.0.1
PORT ?= 8000

BENCHMARK_MODE ?= smoke
BENCHMARK_RUN  ?= local
PARSER         ?= grobid
BENCHMARK_SPLIT ?= train
TRAIN_BENCHMARK_CONFIG ?= benchmarks/bench-train.yaml
VALIDATION_BENCHMARK_CONFIG ?= benchmarks/bench.yaml
BENCHMARK_CONFIG ?= $(if $(filter $(BENCHMARK_SPLIT),train),$(TRAIN_BENCHMARK_CONFIG),$(VALIDATION_BENCHMARK_CONFIG))
unexport BENCHMARK_CONFIG

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
COMPOSE_VLLM         := docker compose -f compose.yml -f compose.vllm.yml
COMPOSE_VLLM_PHOENIX := docker compose -f compose.yml -f compose.phoenix.yml -f compose.vllm.yml

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


with-vllm-download:
	$(VENV)/bin/hf download elifepathways/Qwen3-8B-ft \
		--local-dir .local/qwen3-8b-ft

with-vllm-start:
	$(COMPOSE_VLLM) up -d --wait vllm
	@echo "vLLM ready at http://localhost:18000 (base: Qwen/Qwen3-8B-AWQ + 7 LoRA adapters)"

with-vllm-stop:
	$(COMPOSE_VLLM) stop vllm

with-vllm-logs:
	$(COMPOSE_VLLM) logs -f vllm

with-vllm-clean:
	$(COMPOSE_VLLM) down -v --remove-orphans


with-vllm-phoenix-start:
	$(COMPOSE_VLLM_PHOENIX) up -d --wait
	@echo "Ready. API at http://localhost:8000, Phoenix at http://localhost:6006, vLLM at http://localhost:18000"

with-vllm-phoenix-stop:
	$(COMPOSE_VLLM_PHOENIX) down

with-vllm-phoenix-logs:
	$(COMPOSE_VLLM_PHOENIX) logs -f

with-vllm-phoenix-clean:
	$(COMPOSE_VLLM_PHOENIX) down -v


grobid-start:
	docker compose up -d --wait grobid


sciencebeam-start:
	docker compose --profile sciencebeam up -d --wait sciencebeam-parser
	@echo "ScienceBeam Parser ready at http://localhost:8071/api"

sciencebeam-stop:
	docker compose --profile sciencebeam stop sciencebeam-parser


benchmark-build:
	docker compose --profile benchmark build benchmark


.benchmark-predict:
	@if [ "$(PARSER)" = "sciencebeam" ]; then \
		$(MAKE) sciencebeam-start; \
	else \
		$(MAKE) grobid-start; \
	fi
	docker compose --profile benchmark run --rm \
		-e PARSER=$(PARSER) \
		-e GROBID_URL=$$( [ "$(PARSER)" = "sciencebeam" ] && echo http://sciencebeam-parser:8070/api || echo http://grobid:8070/api ) \
		benchmark \
		python -m benchmarks.predict \
			--config "$(BENCHMARK_CONFIG)" \
			--mode   $(BENCHMARK_MODE) \
			--parser $(PARSER) \
			--out    "benchmarks/runs/$(BENCHMARK_SPLIT)/$(BENCHMARK_RUN)-$(PARSER)"

.benchmark-score:
	docker compose --profile benchmark run --rm --no-deps benchmark \
		python -m benchmarks.score \
			--run    "benchmarks/runs/$(BENCHMARK_SPLIT)/$(BENCHMARK_RUN)-$(PARSER)" \
			--config "$(BENCHMARK_CONFIG)" \
			--out    "benchmarks/runs/$(BENCHMARK_SPLIT)/$(BENCHMARK_RUN)-$(PARSER)/report.md"
	@echo "=== $(PARSER) benchmark ==="
	@cat "benchmarks/runs/$(BENCHMARK_SPLIT)/$(BENCHMARK_RUN)-$(PARSER)/report.md"


benchmark-train-grobid: benchmark-build
	$(MAKE) PARSER=grobid BENCHMARK_SPLIT=train .benchmark-predict .benchmark-score

benchmark-validation-grobid: benchmark-build
	$(MAKE) PARSER=grobid BENCHMARK_SPLIT=validation .benchmark-predict .benchmark-score

# Force a full re-predict + re-score (use when adding new score fields requires
# fresh per-document rows, e.g. after adding title_edit_sim).
benchmark-train-rescore-grobid:
	rm -f "benchmarks/runs/train/$(BENCHMARK_RUN)-grobid/per_document.jsonl"
	$(MAKE) benchmark-train-grobid


benchmark-train-lora: benchmark-build
	$(MAKE) grobid-start
	$(COMPOSE_VLLM) up -d --wait vllm
	$(COMPOSE_VLLM) --profile benchmark run --rm \
		benchmark \
		python -m benchmarks.predict \
			--config "$(BENCHMARK_CONFIG)" \
			--mode   $(BENCHMARK_MODE) \
			--out    "benchmarks/runs/$(BENCHMARK_SPLIT)/$(BENCHMARK_RUN)-lora"
	$(MAKE) PARSER=lora BENCHMARK_SPLIT=$(BENCHMARK_SPLIT) .benchmark-score

benchmark-validation-lora: benchmark-build
	$(MAKE) BENCHMARK_SPLIT=validation benchmark-train-lora


benchmark-train-sciencebeam-parser: benchmark-build
	$(MAKE) PARSER=sciencebeam BENCHMARK_SPLIT=train .benchmark-predict .benchmark-score

benchmark-validation-sciencebeam-parser: benchmark-build
	$(MAKE) PARSER=sciencebeam BENCHMARK_SPLIT=validation .benchmark-predict .benchmark-score


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
