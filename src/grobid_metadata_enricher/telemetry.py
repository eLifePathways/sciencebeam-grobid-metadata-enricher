from __future__ import annotations

import atexit
import logging
import os
from typing import Callable, TypeVar

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

_T = TypeVar("_T")
_logger = logging.getLogger(__name__)


def init_telemetry() -> None:
    """Configure the global OTEL TracerProvider. Call once from each entry point (API startup, CLI)."""
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")
    if not endpoint:
        _logger.info("OTEL tracing disabled (OTEL_EXPORTER_OTLP_ENDPOINT not set)")
        return
    if isinstance(trace.get_tracer_provider(), TracerProvider):
        _logger.debug("OTEL tracing already initialised")
        return
    resource_attrs: dict = {SERVICE_NAME: "grobid-metadata-enricher"}
    project_name = os.getenv("PHOENIX_PROJECT_NAME")
    if project_name:
        resource_attrs["openinference.project.name"] = project_name
    provider = TracerProvider(resource=Resource(resource_attrs))
    exporter = OTLPSpanExporter()  # reads OTEL_EXPORTER_OTLP_ENDPOINT from env
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    atexit.register(provider.shutdown)
    _logger.info("OTEL tracing enabled, exporting to %s", endpoint)


def get_tracer() -> trace.Tracer:
    return trace.get_tracer("grobid-metadata-enricher")


def with_otel_context(fn: Callable[[], _T]) -> Callable[[], _T]:
    """Capture the current OTEL context so it can be restored in a worker thread."""
    ctx = otel_context.get_current()

    def wrapped() -> _T:
        token = otel_context.attach(ctx)
        try:
            return fn()
        finally:
            otel_context.detach(token)

    return wrapped
