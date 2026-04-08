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
    def test_logs_with_singular_values(self, caplog):
        handler = LoggingHandler()
        with caplog.at_level(logging.INFO, logger="glassbox.handlers"):
            handler.handle(_make_snapshot())
        assert "singular values" in caplog.text

    def test_logs_without_singular_values(self, caplog):
        handler = LoggingHandler()
        with caplog.at_level(logging.INFO, logger="glassbox.handlers"):
            handler.handle(_make_snapshot_selfattn())
        assert "selfattn" in caplog.text

    def test_close_is_noop(self):
        handler = LoggingHandler()
        handler.close()  # should not raise



# ── OtelHandler ──────────────────────────────────────────────────────────


class TestOtelHandler:
    def test_creates_span_with_attributes(self):
        mock_span = MagicMock()
        mock_span.__enter__ = MagicMock(return_value=mock_span)
        mock_span.__exit__ = MagicMock(return_value=False)

        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.return_value = mock_span

        handler = OtelHandler()
        handler._tracer = mock_tracer

        handler.handle(_make_snapshot())

        mock_tracer.start_as_current_span.assert_called_once_with("glassbox.spectral")
        attr_calls = {c[0][0]: c[0][1] for c in mock_span.set_attribute.call_args_list}
        assert attr_calls["glassbox.signal"] == "spectral"
        assert attr_calls["glassbox.head"] == 0
        assert attr_calls["glassbox.L"] == 128
        assert attr_calls["glassbox.sv1"] == 10.0
        assert attr_calls["glassbox.sv_ratio"] == pytest.approx(2.0)

    def test_noops_when_otel_unavailable(self):
        handler = OtelHandler()
        handler._tracer = None
        handler.handle(_make_snapshot())  # should not raise



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
