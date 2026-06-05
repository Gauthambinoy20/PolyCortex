"""OpenTelemetry tracing scaffold."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_tracer_provider: Any | None = None


def setup_tracing(exporter_type: str = "console", endpoint: str | None = None) -> Any:
    """Configure an OpenTelemetry ``TracerProvider``.

    Args:
        exporter_type: One of ``console``, ``otlp``, or ``memory``.
        endpoint: OTLP endpoint URL (only used when ``exporter_type='otlp'``).

    Returns:
        The ``TracerProvider`` instance.
    """
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

    provider = TracerProvider()

    if exporter_type == "console":
        exporter: Any = ConsoleSpanExporter()
    elif exporter_type == "otlp":
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        except ImportError as exc:  # pragma: no cover
            raise ImportError("opentelemetry-exporter-otlp-proto-grpc required for OTLP export") from exc
        exporter = OTLPSpanExporter(endpoint=endpoint) if endpoint else OTLPSpanExporter()
    elif exporter_type == "memory":
        try:
            from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
                InMemorySpanExporter,
            )
        except ImportError:  # pragma: no cover
            # Older versions expose it differently
            # Older aiohttp/otel versions expose this symbol here; current stubs
            # only declare it in the in_memory_span_exporter submodule above.
            from opentelemetry.sdk.trace.export import (  # type: ignore[no-redef,attr-defined]
                InMemorySpanExporter,
            )
        exporter = InMemorySpanExporter()
    else:
        raise ValueError(f"Unknown exporter_type: {exporter_type}")

    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    global _tracer_provider
    _tracer_provider = provider
    logger.info("OpenTelemetry tracing configured with exporter=%s", exporter_type)
    return provider


def get_tracer(name: str = "polymarket_agent") -> Any:
    """Return a tracer, lazily configuring a console provider if needed."""
    from opentelemetry import trace

    return trace.get_tracer(name)
