import pytest

from glassbox.results import (
    SPECTRAL_FEATURE_NAMES,
    LaplacianFeatures,
    RoutingFeatures,
    SelfAttnFeatures,
    SpectralFeatures,
    SVDSnapshot,
    TrackerFeatures,
    _spectral_from_svs,
)

# ── _spectral_from_svs ────────────────────────────────────────────────────


class TestSpectralFromSvs:
    def test_normal(self):
        result = _spectral_from_svs([10.0, 5.0, 2.0])
        assert result["sv1"] == 10.0
        assert result["sv_ratio"] == pytest.approx(2.0)
        assert result["sv_entropy"] is not None
        assert result["sv_entropy"] > 0

    def test_single_value(self):
        result = _spectral_from_svs([7.0])
        assert result["sv1"] == 7.0
        assert result["sv_ratio"] is None
        # entropy of single value: -1.0 * log(1.0 + 1e-12) ≈ 0
        assert result["sv_entropy"] == pytest.approx(0.0, abs=1e-6)

    def test_empty(self):
        result = _spectral_from_svs([])
        assert result == {"sv1": None, "sv_ratio": None, "sv_entropy": None}

    def test_all_zeros(self):
        result = _spectral_from_svs([0.0, 0.0])
        assert result["sv1"] == 0.0
        assert result["sv_ratio"] is None  # division by zero guard
        assert result["sv_entropy"] is None  # total == 0

    def test_second_sv_zero(self):
        result = _spectral_from_svs([5.0, 0.0])
        assert result["sv1"] == 5.0
        assert result["sv_ratio"] is None


# ── SpectralFeatures ─────────────────────────────────────────────────────


class TestSpectralFeatures:
    def test_derives_spectral(self):
        f = SpectralFeatures(singular_values=[429.6, 59.0, 41.9])
        assert f.sv1 == 429.6
        assert f.sv_ratio == pytest.approx(429.6 / 59.0)
        assert f.sv_entropy is not None


# ── RoutingFeatures ──────────────────────────────────────────────────────


class TestRoutingFeatures:
    def test_derives_spectral(self):
        f = RoutingFeatures(
            singular_values=[1.0, 0.5],
            phi_hat=0.31,
            G=0.15,
            curl_ratio=0.42,
        )
        assert f.sv1 == 1.0
        assert f.sv_ratio == pytest.approx(2.0)
        assert f.phi_hat == 0.31
        assert f.G == 0.15
        assert f.curl_ratio == 0.42
        assert f.sigma2 is None  # not passed


# ── TrackerFeatures ──────────────────────────────────────────────────────


class TestTrackerFeatures:
    def test_derives_spectral(self):
        f = TrackerFeatures(
            singular_values=[1.0, 0.5, 0.1],
            sigma2=0.5,
            sigma2_asym=0.02,
            commutator_norm=0.03,
        )
        assert f.sv1 == 1.0
        assert f.sv_ratio == pytest.approx(2.0)
        assert f.sigma2 == 0.5
        assert f.sigma2_asym == 0.02
        assert f.commutator_norm == 0.03


# ── SVDSnapshot ───────────────────────────────────────────────────────────


class TestSVDSnapshot:
    def _make_snapshot(self, **overrides):
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

    def test_construction(self):
        snap = self._make_snapshot()
        assert snap.signal == "spectral"
        assert snap.features.sv1 == 10.0

    def test_model_dump_excludes_none(self):
        snap = self._make_snapshot()
        d = snap.model_dump(exclude_none=True)
        assert "tier" not in d
        assert d["signal"] == "spectral"
        assert d["features"]["sv1"] == 10.0

    def test_round_trip(self):
        snap = self._make_snapshot()
        d = snap.model_dump(exclude_none=True)
        restored = SVDSnapshot.from_jsonl_row(d)
        assert restored.features.sv1 == snap.features.sv1
        assert restored.features.sv_ratio == snap.features.sv_ratio

    def test_tracker_round_trip(self):
        features = TrackerFeatures(
            singular_values=[1.0, 0.5],
            sigma2=0.5,
            sigma2_asym=0.02,
            commutator_norm=0.03,
        )
        snap = self._make_snapshot(
            signal="tracker",
            tier="materialized",
            features=features,
        )
        d = snap.model_dump(exclude_none=True)
        assert d["tier"] == "materialized"
        assert d["features"]["sigma2"] == 0.5
        restored = SVDSnapshot.from_jsonl_row(d)
        assert restored.features.sigma2 == 0.5
        assert restored.features.sigma2_asym == 0.02
        assert restored.features.sv_ratio == pytest.approx(2.0)

    def test_routing_round_trip(self):
        features = RoutingFeatures(
            singular_values=[1.0, 0.5],
            phi_hat=0.3,
            G=0.15,
        )
        snap = self._make_snapshot(
            signal="routing",
            tier="materialized",
            features=features,
        )
        d = snap.model_dump(exclude_none=True)
        assert d["tier"] == "materialized"
        assert d["features"]["phi_hat"] == 0.3
        restored = SVDSnapshot.from_jsonl_row(d)
        assert restored.features.phi_hat == 0.3
        assert restored.features.sv_ratio == pytest.approx(2.0)


# ── SelfAttnFeatures ────────────────────────────────────────────────────


class TestSelfAttnFeatures:
    def test_construction(self):
        f = SelfAttnFeatures(attn_diag_logmean=-2.5)
        assert f.attn_diag_logmean == -2.5


class TestSelfAttnSnapshot:
    def test_round_trip(self):
        features = SelfAttnFeatures(attn_diag_logmean=-3.2)
        snap = SVDSnapshot(
            signal="selfattn",
            request_id=0,
            layer="model.layers.0.self_attn",
            layer_idx=0,
            head=0,
            step=32,
            L=128,
            tier="materialized",
            features=features,
        )
        d = snap.model_dump(exclude_none=True)
        assert d["singular_values"] == []
        assert d["features"]["attn_diag_logmean"] == -3.2
        restored = SVDSnapshot.from_jsonl_row(d)
        assert isinstance(restored.features, SelfAttnFeatures)
        assert restored.features.attn_diag_logmean == -3.2


# ── LaplacianFeatures ──────────────────────────────────────────────────


class TestLaplacianFeatures:
    def test_construction(self):
        f = LaplacianFeatures(eigvals=[0.9, 0.8, 0.7])
        assert f.eigvals == [0.9, 0.8, 0.7]


class TestLaplacianSnapshot:
    def test_round_trip(self):
        features = LaplacianFeatures(eigvals=[0.95, 0.82, 0.71])
        snap = SVDSnapshot(
            signal="laplacian",
            request_id=0,
            layer="model.layers.0.self_attn",
            layer_idx=0,
            head=0,
            step=32,
            L=128,
            tier="materialized",
            features=features,
        )
        d = snap.model_dump(exclude_none=True)
        assert d["singular_values"] == []
        assert d["features"]["eigvals"] == [0.95, 0.82, 0.71]
        restored = SVDSnapshot.from_jsonl_row(d)
        assert isinstance(restored.features, LaplacianFeatures)
        assert restored.features.eigvals == [0.95, 0.82, 0.71]


# ── SPECTRAL_FEATURE_NAMES ────────────────────────────────────────────────


def test_spectral_feature_names():
    assert SPECTRAL_FEATURE_NAMES == ["sv_ratio", "sv1", "sv_entropy"]
