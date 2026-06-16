"""Tests for the Diagnostic protocol and concrete implementations."""

import pytest
import torch

from glassbox.config import GlassboxConfig
from glassbox.diagnostic import Diagnostic
from glassbox.diagnostics import (
    DIAGNOSTIC_REGISTRY,
    LaplacianDiagnostic,
    RoutingDiagnostic,
    SelfAttnDiagnostic,
    SpectralDiagnostic,
    TrackerDiagnostic,
)
from glassbox.results import SpectralFeatures, SVDSnapshot

L = 16
D = 8


@pytest.fixture
def qk():
    torch.manual_seed(42)
    Q = torch.randn(L, D)
    K = torch.randn(L, D)
    return Q, K


class TestRegistry:
    def test_all_signals_registered(self):
        assert set(DIAGNOSTIC_REGISTRY.keys()) == {
            "spectral",
            "routing",
            "tracker",
            "selfattn",
            "laplacian",
        }

    def test_protocol_conformance(self):
        for name, cls in DIAGNOSTIC_REGISTRY.items():
            instance = cls()
            assert isinstance(instance, Diagnostic), f"{name} does not satisfy Diagnostic protocol"


class TestSpectralDiagnostic:
    def test_reduce_returns_features(self, qk):
        Q, K = qk
        diag = SpectralDiagnostic(rank=2, method="randomized")
        result = diag.reduce(Q, K, L)
        assert "features" in result
        assert "singular_values" in result
        assert len(result["singular_values"]) == 2

    def test_witness_not_implemented(self, qk):
        Q, K = qk
        diag = SpectralDiagnostic()
        with pytest.raises(NotImplementedError):
            diag.witness(Q, K, L)

    def test_accumulate_returns_local(self):
        diag = SpectralDiagnostic()
        local = {"features": "test"}
        assert diag.accumulate(local, None) is local
        assert diag.accumulate(local, {"old": True}) is local


class TestRoutingDiagnostic:
    def test_reduce_materialized(self, qk):
        Q, K = qk
        diag = RoutingDiagnostic(rank=2, threshold=1024)
        result = diag.reduce(Q, K, L)
        assert "features" in result
        assert result["tier"] == "materialized"
        assert "singular_values" in result

    def test_reduce_matrix_free(self, qk):
        Q, K = qk
        diag = RoutingDiagnostic(rank=2, threshold=4, block_size=4)
        result = diag.reduce(Q, K, L)
        assert "features" in result
        assert result["tier"] == "matrix_free"

    def test_signal_name(self):
        assert RoutingDiagnostic.signal_name == "routing"


class TestTrackerDiagnostic:
    def test_reduce_materialized(self, qk):
        Q, K = qk
        diag = TrackerDiagnostic(rank=2, threshold=1024)
        result = diag.reduce(Q, K, L)
        assert "features" in result
        assert result["tier"] == "materialized"

    def test_reduce_matrix_free(self, qk):
        Q, K = qk
        diag = TrackerDiagnostic(rank=2, threshold=4, block_size=4)
        result = diag.reduce(Q, K, L)
        assert result["tier"] == "matrix_free"


class TestSelfAttnDiagnostic:
    def test_reduce_materialized(self, qk):
        Q, K = qk
        diag = SelfAttnDiagnostic(top_k=5, threshold=1024)
        result = diag.reduce(Q, K, L)
        assert "features" in result
        assert result["tier"] == "materialized"

    def test_reduce_matrix_free(self, qk):
        Q, K = qk
        diag = SelfAttnDiagnostic(top_k=5, threshold=4, block_size=4)
        result = diag.reduce(Q, K, L)
        assert result["tier"] == "matrix_free"


class TestLaplacianDiagnostic:
    def test_reduce_materialized(self, qk):
        Q, K = qk
        diag = LaplacianDiagnostic(top_k=5, threshold=1024)
        result = diag.reduce(Q, K, L)
        assert "features" in result
        assert result["tier"] == "materialized"

    def test_reduce_matrix_free(self, qk):
        Q, K = qk
        diag = LaplacianDiagnostic(top_k=5, threshold=4, block_size=4)
        result = diag.reduce(Q, K, L)
        assert result["tier"] == "matrix_free"


class TestSVDSnapshotWitness:
    def test_witness_field_default_none(self):
        features = SpectralFeatures(singular_values=[1.0, 0.5])
        snap = SVDSnapshot(
            signal="spectral",
            request_id=0,
            layer="model.layers.0.self_attn",
            layer_idx=0,
            head=0,
            step=32,
            L=16,
            singular_values=[1.0, 0.5],
            features=features,
        )
        assert snap.witness is None

    def test_witness_field_with_values(self):
        features = SpectralFeatures(singular_values=[1.0, 0.5])
        witness = [0.1] * 16
        snap = SVDSnapshot(
            signal="spectral",
            request_id=0,
            layer="model.layers.0.self_attn",
            layer_idx=0,
            head=0,
            step=32,
            L=16,
            singular_values=[1.0, 0.5],
            witness=witness,
            features=features,
        )
        assert snap.witness == witness
        assert len(snap.witness) == 16


class TestEmitWitnessConfig:
    def test_default_false(self):
        cfg = GlassboxConfig()
        assert cfg.emit_witness is False

    def test_set_true(self):
        cfg = GlassboxConfig(emit_witness=True)
        assert cfg.emit_witness is True
