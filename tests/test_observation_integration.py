"""End-to-end integration test: snapshot → ClassifierHandler → VerdictStore → ObservationPlugin → ABORT."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from glassbox.handlers import ClassifierHandler
from glassbox.observation_plugin import (
    GlassboxObservationPlugin,
    ObservationAction,
    _HAS_OBSERVATION_API,
    RequestContext,
)
from glassbox.results import LaplacianFeatures, SVDSnapshot
from glassbox.verdict import VerdictStore


class _FakeClassifier:
    """Module-level fake so it can be pickled by joblib."""

    def __init__(self, prob: float = 0.85):
        self._prob = prob

    def predict_proba(self, X):
        n = X.shape[0]
        return np.array([[1 - self._prob, self._prob]] * n)


@pytest.fixture(autouse=True)
def _clean_store():
    VerdictStore.reset()
    yield
    VerdictStore.reset()


@pytest.fixture()
def classifier_model(tmp_path):
    """Create a fake trained model."""
    import joblib

    feat_cols = [
        f"laplacian_lap_eigval_{ei}_L{li}_H0"
        for li in range(2)
        for ei in range(3)
    ]
    model_dict = {
        "model": _FakeClassifier(prob=0.85),
        "pca": None,
        "feature_columns": feat_cols,
        "signal": "laplacian",
        "threshold": 0.5,
        "metadata": {},
    }
    path = str(tmp_path / "model.joblib")
    joblib.dump(model_dict, path)
    return path


def _make_ctx(request_id: str) -> RequestContext:
    if _HAS_OBSERVATION_API:
        return RequestContext(request_id=request_id)
    ctx = RequestContext()
    ctx.request_id = request_id
    return ctx


class TestFullFlow:
    """Test the complete path: snapshot → handler → verdict → plugin → action."""

    def test_abort_flow(self, classifier_model, caplog):
        # 1. Set up ObservationPlugin and map request IDs
        plugin = GlassboxObservationPlugin()
        plugin.on_request_start("vllm-req-0")

        # 2. Set up ClassifierHandler with action=abort
        handler = ClassifierHandler(
            model_path=classifier_model,
            threshold=0.5,
            signal="laplacian",
            action="abort",
        )

        # 3. Send snapshots (simulating what SVDTritonAttentionImpl does)
        for layer_idx in range(2):
            snap = SVDSnapshot(
                signal="laplacian",
                request_id=0,  # glassbox integer ID (mapped to "vllm-req-0")
                layer=f"model.layers.{layer_idx}.self_attn",
                layer_idx=layer_idx,
                head=0,
                step=1,
                L=128,
                features=LaplacianFeatures(eigvals=[0.9, 0.7, 0.5]),
            )
            handler.handle(snap)

        # 4. Trigger classification (step boundary — send a snapshot for step 2)
        with caplog.at_level(logging.WARNING, logger="glassbox.handlers"):
            trigger = SVDSnapshot(
                signal="laplacian",
                request_id=0,
                layer="model.layers.0.self_attn",
                layer_idx=0,
                head=0,
                step=2,
                L=128,
                features=LaplacianFeatures(eigvals=[0.9, 0.7, 0.5]),
            )
            handler.handle(trigger)

        assert "Hallucination detected" in caplog.text
        assert "action=abort" in caplog.text

        # 5. ObservationPlugin reads the verdict
        ctx = _make_ctx("vllm-req-0")
        results = plugin.on_step_batch({}, [ctx])

        assert len(results) == 1
        assert results[0].action == ObservationAction.ABORT
        assert results[0].metadata["glassbox.probability"] == pytest.approx(0.85)

        handler.close()

    def test_log_only_flow(self, classifier_model, caplog):
        """action=log_only should NOT trigger ABORT even when probability is high."""
        plugin = GlassboxObservationPlugin()
        plugin.on_request_start("vllm-req-0")

        handler = ClassifierHandler(
            model_path=classifier_model,
            threshold=0.5,
            signal="laplacian",
            action="log_only",  # safe default
        )

        for layer_idx in range(2):
            snap = SVDSnapshot(
                signal="laplacian",
                request_id=0,
                layer=f"model.layers.{layer_idx}.self_attn",
                layer_idx=layer_idx,
                head=0,
                step=1,
                L=128,
                features=LaplacianFeatures(eigvals=[0.9, 0.7, 0.5]),
            )
            handler.handle(snap)

        # Trigger classification
        with caplog.at_level(logging.WARNING, logger="glassbox.handlers"):
            handler.handle(SVDSnapshot(
                signal="laplacian",
                request_id=0,
                layer="model.layers.0.self_attn",
                layer_idx=0,
                head=0,
                step=2,
                L=128,
                features=LaplacianFeatures(eigvals=[0.9, 0.7, 0.5]),
            ))

        # Warning is still logged
        assert "Hallucination detected" in caplog.text

        # But ObservationPlugin should NOT abort (action=log_only)
        ctx = _make_ctx("vllm-req-0")
        results = plugin.on_step_batch({}, [ctx])

        assert len(results) == 1
        assert results[0].action == ObservationAction.CONTINUE

        handler.close()

    def test_below_threshold_continues(self, tmp_path):
        """Probability below threshold → CONTINUE."""
        import joblib

        feat_cols = [f"laplacian_lap_eigval_{ei}_L0_H0" for ei in range(3)]
        model_dict = {
            "model": _FakeClassifier(prob=0.2),  # below threshold
            "pca": None,
            "feature_columns": feat_cols,
            "signal": "laplacian",
            "metadata": {},
        }
        path = str(tmp_path / "low.joblib")
        joblib.dump(model_dict, path)

        plugin = GlassboxObservationPlugin()
        plugin.on_request_start("vllm-req-0")

        handler = ClassifierHandler(
            model_path=path, threshold=0.5, action="abort"
        )

        handler.handle(SVDSnapshot(
            signal="laplacian",
            request_id=0,
            layer="model.layers.0.self_attn",
            layer_idx=0,
            head=0,
            step=1,
            L=128,
            features=LaplacianFeatures(eigvals=[0.9, 0.7, 0.5]),
        ))
        # Trigger via close
        handler.close()

        ctx = _make_ctx("vllm-req-0")
        results = plugin.on_step_batch({}, [ctx])

        assert results[0].action == ObservationAction.CONTINUE
