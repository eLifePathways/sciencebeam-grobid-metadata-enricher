FROM lfoppiano/grobid:0.9.0-crf AS grobid

FROM python:3.11-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libfontconfig1 \
    libfreetype6 \
    libexpat1 \
    libbz2-1.0 \
    libpng16-16 \
    libbrotli1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=grobid /opt/grobid/grobid-home/pdfalto/ /opt/pdfalto/
RUN chmod +x /opt/pdfalto/lin-64/pdfalto

ENV PDFALTO_BIN=/opt/pdfalto/lin-64/pdfalto
ENV PYTHONUNBUFFERED=1

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV VENV=/app/.venv
ENV VIRTUAL_ENV=${VENV} PYTHONUSERBASE=${VENV} PATH=${VENV}/bin:$PATH

WORKDIR /app


FROM base AS benchmark

COPY pyproject.toml uv.lock ./
RUN uv sync --extra bench --extra dev --extra observe --extra cache --no-install-project

COPY src/ src/
COPY benchmarks/ benchmarks/
RUN uv sync --extra bench --extra dev --extra observe --extra cache


FROM base AS runtime

COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --extra observe --no-install-project

COPY src/ src/
RUN uv sync --no-dev --extra observe

EXPOSE 8000

CMD [".venv/bin/uvicorn", "grobid_metadata_enricher.api:app", "--host", "0.0.0.0", "--port", "8000"]
