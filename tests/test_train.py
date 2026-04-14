"""Tests for glassbox-train CLI and training logic."""

from __future__ import annotations

import numpy as np
import pytest

from glassbox.cli.train import _find_feature_columns, _train_probe
from glassbox.results import parse_feature_column


# ── parse_feature_column ────────────────────────────────────────────────


class TestParseFeatureColumn:
    def test_laplacian_column(self):
        result = parse_feature_column("laplacian_lap_eigval_3_L2_H0")
        assert result == ("laplacian", 3, 2, 0)

    def test_selfattn_column(self):
        result = parse_feature_column("selfattn_ad_eigval_0_L0_H1")
        assert result == ("selfattn", 0, 0, 1)

    def test_non_feature_column_returns_none(self):
        assert parse_feature_column("label") is None
        assert parse_feature_column("phase") is None
        assert parse_feature_column("request_id") is None

    def test_other_feature_column_returns_none(self):
        # Scalar features (non-eigval) should not match
        assert parse_feature_column("spectral_sv_ratio_L0_H0") is None

    def test_multi_digit_indices(self):
        result = parse_feature_column("laplacian_lap_eigval_12_L31_H15")
        assert result == ("laplacian", 12, 31, 15)


# ── _find_feature_columns ──────────────────────────────────────────────


class TestFindFeatureColumns:
    @pytest.fixture()
    def sample_columns(self):
        """Column names mimicking a 2-layer, 1-head, 3-eigval Parquet."""
        meta = ["request_id", "label", "phase", "sample_id", "length", "source"]
        feats = []
        for li in range(2):
            for ei in range(3):
                feats.append(f"laplacian_lap_eigval_{ei}_L{li}_H0")
        return meta + feats

    def test_finds_all_laplacian(self, sample_columns):
        cols = _find_feature_columns(sample_columns, "laplacian", layer=None)
        assert len(cols) == 6  # 2 layers x 3 eigvals

    def test_filters_by_layer(self, sample_columns):
        cols = _find_feature_columns(sample_columns, "laplacian", layer=1)
        assert len(cols) == 3
        assert all("_L1_" in c for c in cols)

    def test_wrong_signal_returns_empty(self, sample_columns):
        cols = _find_feature_columns(sample_columns, "spectral", layer=None)
        assert cols == []

    def test_nonexistent_layer_returns_empty(self, sample_columns):
        cols = _find_feature_columns(sample_columns, "laplacian", layer=99)
        assert cols == []


# ── _train_probe (end-to-end on synthetic data) ────────────────────────


def _make_parquet(tmp_path, n_samples=60, n_layers=2, n_heads=1, n_eigvals=3, nan_rows=0):
    """Create a synthetic Parquet file matching glassbox-extract output."""
    import pandas as pd

    rng = np.random.RandomState(42)

    data = {
        "request_id": list(range(n_samples)),
        "label": rng.randint(0, 2, size=n_samples).tolist(),
        "phase": ["full"] * n_samples,
        "sample_id": list(range(n_samples)),
        "length": [128] * n_samples,
    }

    feat_cols = []
    for li in range(n_layers):
        for hi in range(n_heads):
            for ei in range(n_eigvals):
                col = f"laplacian_lap_eigval_{ei}_L{li}_H{hi}"
                feat_cols.append(col)
                data[col] = rng.randn(n_samples).tolist()

    # Inject NaN rows if requested
    for i in range(nan_rows):
        for col in feat_cols:
            data[col][i] = float("nan")

    df = pd.DataFrame(data)
    path = tmp_path / "features.parquet"
    df.to_parquet(path)
    return str(path), feat_cols


class TestTrainProbe:
    def test_basic_training(self, tmp_path):
        parquet_path, _ = _make_parquet(tmp_path)
        output = str(tmp_path / "model.joblib")
        result = _train_probe(parquet_path, output, "laplacian", pca=0, test_size=0.3, layer=None, seed=42)

        assert result["signal"] == "laplacian"
        assert 0.0 <= result["train_auroc"] <= 1.0
        assert 0.0 <= result["test_auroc"] <= 1.0
        assert len(result["feature_columns"]) == 6  # 2 layers x 1 head x 3 eigvals
        assert result["pca"] is None

    def test_training_with_pca(self, tmp_path):
        parquet_path, _ = _make_parquet(tmp_path)
        output = str(tmp_path / "model.joblib")
        result = _train_probe(parquet_path, output, "laplacian", pca=3, test_size=0.3, layer=None, seed=42)

        assert result["pca"] is not None
        assert result["metadata"]["n_features_pca"] <= 3

    def test_single_layer(self, tmp_path):
        parquet_path, _ = _make_parquet(tmp_path)
        output = str(tmp_path / "model.joblib")
        result = _train_probe(parquet_path, output, "laplacian", pca=0, test_size=0.3, layer=0, seed=42)

        assert len(result["feature_columns"]) == 3  # 1 layer x 1 head x 3 eigvals
        assert all("_L0_" in c for c in result["feature_columns"])

    def test_nan_rows_dropped(self, tmp_path):
        parquet_path, _ = _make_parquet(tmp_path, n_samples=60, nan_rows=5)
        output = str(tmp_path / "model.joblib")
        result = _train_probe(parquet_path, output, "laplacian", pca=0, test_size=0.3, layer=None, seed=42)

        total = result["metadata"]["n_train"] + result["metadata"]["n_test"]
        assert total == 55  # 60 - 5 NaN rows

    def test_model_save_load_roundtrip(self, tmp_path):
        import joblib

        parquet_path, feat_cols = _make_parquet(tmp_path)
        output = str(tmp_path / "model.joblib")
        _train_probe(parquet_path, output, "laplacian", pca=0, test_size=0.3, layer=None, seed=42)

        loaded = joblib.load(output)
        assert "model" in loaded
        assert "feature_columns" in loaded
        assert loaded["signal"] == "laplacian"
        # Model can predict
        X_fake = np.random.randn(5, len(loaded["feature_columns"]))
        proba = loaded["model"].predict_proba(X_fake)
        assert proba.shape == (5, 2)

    def test_filters_to_full_phase(self, tmp_path):
        """Ensure 'question' phase rows are excluded."""
        import pandas as pd

        rng = np.random.RandomState(42)
        n = 40
        data = {
            "request_id": list(range(n)),
            "label": rng.randint(0, 2, size=n).tolist(),
            "phase": (["question", "full"] * (n // 2)),
            "sample_id": list(range(n)),
            "length": [128] * n,
        }
        for li in range(2):
            col = f"laplacian_lap_eigval_0_L{li}_H0"
            data[col] = rng.randn(n).tolist()

        path = tmp_path / "features.parquet"
        pd.DataFrame(data).to_parquet(path)

        output = str(tmp_path / "model.joblib")
        result = _train_probe(str(path), output, "laplacian", pca=0, test_size=0.3, layer=None, seed=42)

        total = result["metadata"]["n_train"] + result["metadata"]["n_test"]
        assert total == 20  # only the 20 "full" rows

    def test_no_feature_columns_raises(self, tmp_path):
        import pandas as pd

        df = pd.DataFrame({"label": [0, 1], "phase": ["full", "full"]})
        path = tmp_path / "empty.parquet"
        df.to_parquet(path)

        with pytest.raises(Exception, match="No feature columns"):
            _train_probe(str(path), str(tmp_path / "m.joblib"), "laplacian", pca=0, test_size=0.3, layer=None, seed=42)
