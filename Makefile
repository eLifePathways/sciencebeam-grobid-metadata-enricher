.PHONY: install lint

VENV := .venv

install:
	uv sync --extra dev

lint:
	$(VENV)/bin/ruff check src/
	$(VENV)/bin/mypy src/
	$(VENV)/bin/pylint src/
