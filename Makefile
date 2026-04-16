.PHONY: install lint format test serve serve-reload build start stop logs

-include .env
export

VENV := .venv
HOST ?= 127.0.0.1
PORT ?= 8000

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
