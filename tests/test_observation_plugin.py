"""Tests for GlassboxObservationPlugin."""

from __future__ import annotations

import pytest

from glassbox.observation_plugin import (
    GlassboxObservationPlugin,
    ObservationAction,
    ObservationResult,
    RequestContext,
    _HAS_OBSERVATION_API,
)
from glassbox.verdict import VerdictStore


@pytest.fixture(autouse=True)
def _clean_store():
    VerdictStore.reset()
    yield
    VerdictStore.reset()


def _make_ctx(request_id: str = "req-0", **kwargs) -> RequestContext:
    if _HAS_OBSERVATION_API:
        return RequestContext(request_id=request_id, **kwargs)
    ctx = RequestContext()
    ctx.request_id = request_id
    return ctx


class TestGlassboxObservationPlugin:
    def test_get_observation_layers_empty(self):
        plugin = GlassboxObservationPlugin()
        assert plugin.get_observation_layers() == []

    def test_observe_decode_true(self):
        plugin = GlassboxObservationPlugin()
        assert plugin.observe_decode() is True

    def test_on_request_start_maps_id(self):
        plugin = GlassboxObservationPlugin()
        plugin.on_request_start("req-abc")

        # Verify mapping was established
        VerdictStore.report(0, 0.9, "abort")
        verdict = VerdictStore.consume("req-abc")
        assert verdict is not None
        assert verdict.probability == pytest.approx(0.9)

    def test_on_request_start_increments_id(self):
        plugin = GlassboxObservationPlugin()
        plugin.on_request_start("req-0")
        plugin.on_request_start("req-1")

        VerdictStore.report(0, 0.3, "continue")
        VerdictStore.report(1, 0.9, "abort")

        v0 = VerdictStore.consume("req-0")
        v1 = VerdictStore.consume("req-1")
        assert v0.action == "continue"
        assert v1.action == "abort"

    def test_on_step_batch_returns_continue_no_verdict(self):
        plugin = GlassboxObservationPlugin()
        plugin.on_request_start("req-0")

        ctx = _make_ctx("req-0")
        results = plugin.on_step_batch({}, [ctx])

        assert len(results) == 1
        assert results[0].action == ObservationAction.CONTINUE

    def test_on_step_batch_returns_abort(self):
        plugin = GlassboxObservationPlugin()
        plugin.on_request_start("req-0")

        VerdictStore.report(0, 0.9, "abort")

        ctx = _make_ctx("req-0")
        results = plugin.on_step_batch({}, [ctx])

        assert len(results) == 1
        assert results[0].action == ObservationAction.ABORT
        assert results[0].metadata["glassbox.probability"] == pytest.approx(0.9)

    def test_on_step_batch_returns_preempt(self):
        plugin = GlassboxObservationPlugin()
        plugin.on_request_start("req-0")

        VerdictStore.report(0, 0.8, "preempt")

        ctx = _make_ctx("req-0")
        results = plugin.on_step_batch({}, [ctx])

        assert len(results) == 1
        assert results[0].action == ObservationAction.PREEMPT

    def test_on_step_batch_continue_below_threshold(self):
        plugin = GlassboxObservationPlugin()
        plugin.on_request_start("req-0")

        VerdictStore.report(0, 0.3, "continue")

        ctx = _make_ctx("req-0")
        results = plugin.on_step_batch({}, [ctx])

        assert len(results) == 1
        assert results[0].action == ObservationAction.CONTINUE

    def test_on_step_batch_multiple_requests(self):
        plugin = GlassboxObservationPlugin()
        plugin.on_request_start("req-0")
        plugin.on_request_start("req-1")

        VerdictStore.report(0, 0.3, "continue")
        VerdictStore.report(1, 0.9, "abort")

        contexts = [_make_ctx("req-0"), _make_ctx("req-1")]
        results = plugin.on_step_batch({}, contexts)

        assert len(results) == 2
        assert results[0].action == ObservationAction.CONTINUE
        assert results[1].action == ObservationAction.ABORT

    def test_on_request_complete_cleans_up(self):
        plugin = GlassboxObservationPlugin()
        plugin.on_request_start("req-0")
        VerdictStore.report(0, 0.9, "abort")

        plugin.on_request_complete("req-0")

        # Verdict should be cleaned up
        assert VerdictStore.consume("req-0") is None

    def test_consume_is_one_shot(self):
        """Verdict is consumed on first on_step_batch, not available on second."""
        plugin = GlassboxObservationPlugin()
        plugin.on_request_start("req-0")
        VerdictStore.report(0, 0.9, "abort")

        ctx = _make_ctx("req-0")
        results1 = plugin.on_step_batch({}, [ctx])
        results2 = plugin.on_step_batch({}, [ctx])

        assert results1[0].action == ObservationAction.ABORT
        assert results2[0].action == ObservationAction.CONTINUE


class TestRegistration:
    def test_register_observation_plugin(self):
        from glassbox.observation_plugin import register_observation_plugin

        qualname = register_observation_plugin()
        assert qualname == "glassbox.observation_plugin:GlassboxObservationPlugin"
