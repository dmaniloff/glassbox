"""Tests for glassbox-serve REST endpoint."""

from __future__ import annotations

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient


class _FakeClassifier:
    """Module-level fake so it can be pickled by joblib."""

    def __init__(self, prob: float = 0.75):
        self._prob = prob

    def predict_proba(self, X):
        n = X.shape[0]
        return np.array([[1 - self._prob, self._prob]] * n)


@pytest.fixture()
def serve_app(tmp_path):
    """Build a test app with a fake model."""
    feat_cols = [
        f"laplacian_lap_eigval_{ei}_L{li}_H0"
        for li in range(2)
        for ei in range(3)
    ]
    model_dict = {
        "model": _FakeClassifier(prob=0.75),
        "pca": None,
        "feature_columns": feat_cols,
        "signal": "laplacian",
        "threshold": 0.5,
        "train_auroc": 0.9,
        "test_auroc": 0.85,
        "metadata": {},
    }
    model_path = str(tmp_path / "model.joblib")
    joblib.dump(model_dict, model_path)

    from glassbox.cli.serve import _build_app

    app = _build_app(model_path)
    return TestClient(app), feat_cols


class TestHealth:
    def test_health_returns_ok(self, serve_app):
        client, _ = serve_app
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["signal"] == "laplacian"
        assert data["n_features"] == 6
        assert data["train_auroc"] == pytest.approx(0.9)
        assert data["test_auroc"] == pytest.approx(0.85)


class TestClassify:
    def test_classify_returns_probability(self, serve_app):
        client, feat_cols = serve_app
        features = {col: 0.5 for col in feat_cols}
        resp = client.post("/classify", json={"features": features})
        assert resp.status_code == 200
        data = resp.json()
        assert data["hallucination_probability"] == pytest.approx(0.75)
        assert data["is_hallucination"] is True
        assert data["threshold"] == 0.5

    def test_classify_with_custom_threshold(self, serve_app):
        client, feat_cols = serve_app
        features = {col: 0.5 for col in feat_cols}
        resp = client.post(
            "/classify", json={"features": features, "threshold": 0.9}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["is_hallucination"] is False  # 0.75 < 0.9
        assert data["threshold"] == 0.9

    def test_classify_missing_features_default_to_zero(self, serve_app):
        client, _ = serve_app
        # Send partial features — missing ones default to 0
        resp = client.post(
            "/classify", json={"features": {"laplacian_lap_eigval_0_L0_H0": 0.5}}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "hallucination_probability" in data

    def test_classify_empty_features(self, serve_app):
        client, _ = serve_app
        resp = client.post("/classify", json={"features": {}})
        assert resp.status_code == 200


class TestThresholdOverride:
    def test_app_level_threshold_override(self, tmp_path):
        feat_cols = ["laplacian_lap_eigval_0_L0_H0"]
        model_dict = {
            "model": _FakeClassifier(prob=0.6),
            "pca": None,
            "feature_columns": feat_cols,
            "signal": "laplacian",
            "threshold": 0.5,
            "metadata": {},
        }
        model_path = str(tmp_path / "model.joblib")
        joblib.dump(model_dict, model_path)

        from glassbox.cli.serve import _build_app

        # Override threshold to 0.8
        app = _build_app(model_path, threshold_override=0.8)
        client = TestClient(app)

        resp = client.post(
            "/classify", json={"features": {"laplacian_lap_eigval_0_L0_H0": 1.0}}
        )
        data = resp.json()
        assert data["threshold"] == 0.8
        assert data["is_hallucination"] is False  # 0.6 < 0.8
