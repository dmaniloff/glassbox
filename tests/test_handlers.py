import json
import logging
from unittest.mock import MagicMock

import pytest

from glassbox.config import GlassboxConfig
from glassbox.handlers import (
    JsonlHandler,
    LoggingHandler,
    OtelHandler,
    create_handlers_from_config,
)
from glassbox.results import SelfAttnFeatures, SpectralFeatures, SVDSnapshot

# ── Test helpers ─────────────────────────────────────────────────────────


def _make_snapshot(**overrides) -> SVDSnapshot:
    defaults = {
        "signal": "spectral",
        "request_id": 0,
        "layer": "model.layers.0.self_attn",
        "layer_idx": 0,
        "head": 0,
        "step": 32,
        "L": 128,
        "singular_values": [10.0, 5.0, 2.0],
        "features": SpectralFeatures(singular_values=[10.0, 5.0, 2.0]),
    }
    defaults.update(overrides)
    return SVDSnapshot(**defaults)


def _make_snapshot_selfattn(**overrides) -> SVDSnapshot:
    defaults = {
        "signal": "selfattn",
        "request_id": 0,
        "layer": "model.layers.0.self_attn",
        "layer_idx": 0,
        "head": 0,
        "step": 32,
        "L": 128,
        "tier": "materialized",
        "features": SelfAttnFeatures(attn_diag_logmean=-3.2),
    }
    defaults.update(overrides)
    return SVDSnapshot(**defaults)


# ── JsonlHandler ─────────────────────────────────────────────────────────


class TestJsonlHandler:
    def test_writes_jsonl(self, tmp_path):
        path = str(tmp_path / "out.jsonl")
        handler = JsonlHandler(path)
        handler.handle(_make_snapshot())
        handler.close()
        with open(path) as f:
            lines = f.readlines()
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["signal"] == "spectral"
        assert data["features"]["sv1"] == 10.0

    def test_appends_multiple(self, tmp_path):
        path = str(tmp_path / "out.jsonl")
        handler = JsonlHandler(path)
        handler.handle(_make_snapshot(step=1))
        handler.handle(_make_snapshot(step=2))
        handler.close()
        with open(path) as f:
            lines = f.readlines()
        assert len(lines) == 2

    def test_close_idempotent(self, tmp_path):
        path = str(tmp_path / "out.jsonl")
        handler = JsonlHandler(path)
        handler.handle(_make_snapshot())
        handler.close()
        handler.close()  # should not raise


# ── LoggingHandler ───────────────────────────────────────────────────────


class TestLoggingHandler:
    def test_logs_spectral_repr(self, caplog):
        handler = LoggingHandler()
        with caplog.at_level(logging.INFO, logger="glassbox.handlers"):
            handler.handle(_make_snapshot())
        assert "[spectral]" in caplog.text
        assert "sv1=" in caplog.text

    def test_logs_selfattn_repr(self, caplog):
        handler = LoggingHandler()
        with caplog.at_level(logging.INFO, logger="glassbox.handlers"):
            handler.handle(_make_snapshot_selfattn())
        assert "[selfattn]" in caplog.text
        assert "attn_diag_logmean=" in caplog.text

    def test_close_is_noop(self):
        handler = LoggingHandler()
        handler.close()  # should not raise


# ── OtelHandler ──────────────────────────────────────────────────────────


class TestOtelHandler:
    @staticmethod
    def _make_handler():
        """Create an OtelHandler with a mock tracer and trace module."""
        mock_trace_mod = MagicMock()
        mock_tracer = MagicMock()

        # Parent spans returned by start_span (not context-managed)
        parent_span = MagicMock()
        mock_tracer.start_span.return_value = parent_span

        # Child spans returned by start_as_current_span (context-managed)
        child_span = MagicMock()
        child_span.__enter__ = MagicMock(return_value=child_span)
        child_span.__exit__ = MagicMock(return_value=False)
        mock_tracer.start_as_current_span.return_value = child_span

        handler = OtelHandler()
        handler._tracer = mock_tracer
        handler._trace_mod = mock_trace_mod
        return handler, mock_tracer, parent_span, child_span, mock_trace_mod

    def test_creates_parent_and_child_spans(self):
        handler, mock_tracer, parent_span, child_span, mock_trace_mod = self._make_handler()

        handler.handle(_make_snapshot())

        # Parent span created for (request_id=0, step=32)
        mock_tracer.start_span.assert_called_once_with("glassbox.step")
        parent_span.set_attribute.assert_any_call("glassbox.request_id", 0)
        parent_span.set_attribute.assert_any_call("glassbox.step", 32)

        # Child span created under parent context
        ctx = mock_trace_mod.set_span_in_context.return_value
        mock_tracer.start_as_current_span.assert_called_once_with(
            "glassbox.spectral",
            context=ctx,
        )
        attr_calls = {c[0][0]: c[0][1] for c in child_span.set_attribute.call_args_list}
        assert attr_calls["glassbox.signal"] == "spectral"
        assert attr_calls["glassbox.head"] == 0
        assert attr_calls["glassbox.L"] == 128
        assert attr_calls["glassbox.sv1"] == 10.0
        assert attr_calls["glassbox.sv_ratio"] == pytest.approx(2.0)

    def test_same_step_reuses_parent_span(self):
        handler, mock_tracer, parent_span, _, _ = self._make_handler()

        handler.handle(_make_snapshot(layer="model.layers.0.self_attn", layer_idx=0))
        handler.handle(_make_snapshot(layer="model.layers.1.self_attn", layer_idx=1))

        # Only one parent span created
        mock_tracer.start_span.assert_called_once()
        # Two child spans
        assert mock_tracer.start_as_current_span.call_count == 2

    def test_new_step_ends_previous_parent(self):
        handler, mock_tracer, parent_span, _, _ = self._make_handler()

        handler.handle(_make_snapshot(step=32))

        # New step → new parent span; start_span returns a fresh mock each time
        parent_span_2 = MagicMock()
        mock_tracer.start_span.return_value = parent_span_2

        handler.handle(_make_snapshot(step=64))

        # First parent was ended
        parent_span.end.assert_called_once()
        # Second parent started
        assert mock_tracer.start_span.call_count == 2

    def test_close_ends_active_parent(self):
        handler, _, parent_span, _, _ = self._make_handler()

        handler.handle(_make_snapshot())
        handler.close()

        parent_span.end.assert_called_once()

    def test_close_idempotent(self):
        handler, _, parent_span, _, _ = self._make_handler()

        handler.handle(_make_snapshot())
        handler.close()
        handler.close()  # should not raise or double-end

        parent_span.end.assert_called_once()

    def test_noops_when_otel_unavailable(self):
        handler = OtelHandler()
        handler._tracer = None
        handler.handle(_make_snapshot())  # should not raise


# ── OtelHandler integration (real TracerProvider) ────────────────────────

otel_sdk = pytest.importorskip("opentelemetry.sdk")


@pytest.fixture()
def otel_spans():
    """Set up a real TracerProvider with InMemorySpanExporter.

    Yields (make_handler, get_spans) — call make_handler() to create an
    OtelHandler wired to the test provider, and get_spans() to retrieve
    finished spans.
    """
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    def make_handler():
        handler = OtelHandler()
        handler._tracer = provider.get_tracer("glassbox")
        handler._trace_mod = __import__("opentelemetry").trace
        return handler

    yield make_handler, exporter.get_finished_spans
    provider.shutdown()


class TestOtelHandlerIntegration:
    """End-to-end tests with a real TracerProvider and InMemorySpanExporter."""

    def test_single_snapshot_creates_parent_and_child(self, otel_spans):
        make_handler, get_spans = otel_spans
        handler = make_handler()
        handler.handle(_make_snapshot())
        handler.close()

        spans = get_spans()
        assert len(spans) == 2

        child, parent = spans  # child finishes first
        assert parent.name == "glassbox.step"
        assert child.name == "glassbox.spectral"

        # Child is nested under parent
        assert child.parent.span_id == parent.context.span_id

        # Parent attributes
        parent_attrs = dict(parent.attributes)
        assert parent_attrs["glassbox.request_id"] == 0
        assert parent_attrs["glassbox.step"] == 32

        # Child attributes
        child_attrs = dict(child.attributes)
        assert child_attrs["glassbox.signal"] == "spectral"
        assert child_attrs["glassbox.layer"] == "model.layers.0.self_attn"
        assert child_attrs["glassbox.head"] == 0
        assert child_attrs["glassbox.L"] == 128
        assert child_attrs["glassbox.sv1"] == 10.0
        assert child_attrs["glassbox.sv_ratio"] == pytest.approx(2.0)

    def test_same_step_shares_parent(self, otel_spans):
        make_handler, get_spans = otel_spans
        handler = make_handler()
        handler.handle(_make_snapshot(layer="model.layers.0.self_attn", layer_idx=0))
        handler.handle(_make_snapshot(layer="model.layers.1.self_attn", layer_idx=1))
        handler.close()

        spans = get_spans()
        assert len(spans) == 3  # 2 children + 1 parent

        parent = [s for s in spans if s.name == "glassbox.step"]
        children = [s for s in spans if s.name == "glassbox.spectral"]
        assert len(parent) == 1
        assert len(children) == 2

        # Both children share the same parent
        assert children[0].parent.span_id == parent[0].context.span_id
        assert children[1].parent.span_id == parent[0].context.span_id

    def test_new_step_creates_new_parent(self, otel_spans):
        make_handler, get_spans = otel_spans
        handler = make_handler()
        handler.handle(_make_snapshot(step=32))
        handler.handle(_make_snapshot(step=64))
        handler.close()

        spans = get_spans()
        parents = [s for s in spans if s.name == "glassbox.step"]
        children = [s for s in spans if s.name == "glassbox.spectral"]
        assert len(parents) == 2
        assert len(children) == 2

        # Each child has a different parent
        assert children[0].parent.span_id != children[1].parent.span_id

        # Parent step attributes are correct
        parent_steps = {dict(p.attributes)["glassbox.step"] for p in parents}
        assert parent_steps == {32, 64}

    def test_selfattn_snapshot_attributes(self, otel_spans):
        make_handler, get_spans = otel_spans
        handler = make_handler()
        handler.handle(_make_snapshot_selfattn())
        handler.close()

        spans = get_spans()
        child = [s for s in spans if s.name == "glassbox.selfattn"][0]
        attrs = dict(child.attributes)
        assert attrs["glassbox.signal"] == "selfattn"
        assert attrs["glassbox.tier"] == "materialized"
        assert attrs["glassbox.attn_diag_logmean"] == pytest.approx(-3.2)


# ── create_handlers_from_config ──────────────────────────────────────────


class TestCreateHandlersFromConfig:
    def test_output_creates_jsonl(self):
        config = GlassboxConfig(output={"path": "/tmp/test.jsonl"})
        handlers = create_handlers_from_config(config)
        assert len(handlers) == 1
        assert isinstance(handlers[0], JsonlHandler)

    def test_otel_creates_otel_handler(self):
        config = GlassboxConfig(emit={"otel": True})
        handlers = create_handlers_from_config(config)
        assert len(handlers) == 1
        assert isinstance(handlers[0], OtelHandler)

    def test_both_creates_two_handlers(self):
        config = GlassboxConfig(output={"path": "/tmp/test.jsonl"}, emit={"otel": True})
        handlers = create_handlers_from_config(config)
        assert len(handlers) == 2
        types = {type(h) for h in handlers}
        assert JsonlHandler in types
        assert OtelHandler in types

    def test_neither_creates_logging(self):
        config = GlassboxConfig()
        handlers = create_handlers_from_config(config)
        assert len(handlers) == 1
        assert isinstance(handlers[0], LoggingHandler)
