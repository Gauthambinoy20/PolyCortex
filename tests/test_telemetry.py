"""Tests for infra.telemetry."""

from __future__ import annotations

import pytest

# OpenTelemetry is an optional ("observability") extra. Skip the whole module
# when it is not installed so a plain ``pip install -e .[dev]`` stays green; CI
# installs the extra so these still run there. We probe the SDK submodule
# specifically: the ``opentelemetry`` namespace package can linger via other
# otel-* distributions even when the SDK itself is absent.
pytest.importorskip("opentelemetry.sdk.trace")

from polymarket_agent.infra import telemetry


def test_setup_console():
    provider = telemetry.setup_tracing("console")
    assert provider is not None


def test_setup_memory():
    provider = telemetry.setup_tracing("memory")
    assert provider is not None


def test_setup_unknown_raises():
    with pytest.raises(ValueError):
        telemetry.setup_tracing("bogus")


def test_get_tracer():
    tracer = telemetry.get_tracer("test")
    assert tracer is not None
    with tracer.start_as_current_span("test-span") as span:
        assert span is not None
