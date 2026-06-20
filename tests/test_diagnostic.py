"""Tests for the Diagnostic protocol and concrete implementations."""

import json

import pytest
import torch

from glassbox.config import GlassboxConfig
from glassbox.diagnostic import Diagnostic
from glassbox.diagnostics import (
    DIAGNOSTIC_REGISTRY,
    AsymmetryDiagnostic,
    CheegerDiagnostic,
    LaplacianDiagnostic,
    RoutingDiagnostic,
    SelfAttnDiagnostic,
    SpectralDiagnostic,
    TrackerDiagnostic,
)
from glassbox.results import AsymmetryFeatures, SpectralFeatures, SVDSnapshot

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
            "asymmetry",
            "tracker",
            "selfattn",
            "laplacian",
            "cheeger",
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


class TestAsymmetryDiagnostic:
    def test_signal_name(self):
        assert AsymmetryDiagnostic.signal_name == "asymmetry"

    def test_reduce_materialized_matches_oracle(self, qk):
        # Operator is P (row-stochastic attention), NOT the degree-normalized M.
        Q, K = qk
        scale = 1.0 / (D**0.5)
        P = torch.softmax(Q @ K.T * scale, dim=-1)
        G_ref = (((P - P.T) / 2).norm() / P.norm()).item()

        diag = AsymmetryDiagnostic(threshold=1024, causal=False)
        result = diag.reduce(Q, K, L)
        assert result["tier"] == "materialized"
        assert abs(result["features"].G - G_ref) < 1e-6  # exact below threshold

    def test_reduce_matrix_free_matches_oracle(self, qk):
        Q, K = qk
        scale = 1.0 / (D**0.5)
        P = torch.softmax(Q @ K.T * scale, dim=-1)
        G_ref = (((P - P.T) / 2).norm() / P.norm()).item()

        diag = AsymmetryDiagnostic(
            threshold=4, block_size=4, n_hutchinson=128, seed=0, causal=False
        )
        result = diag.reduce(Q, K, L)
        assert result["tier"] == "matrix_free"
        assert abs(result["features"].G - G_ref) < 0.05

    def test_witness_shape_and_consistency(self, qk):
        Q, K = qk
        diag = AsymmetryDiagnostic(threshold=4, block_size=4, n_hutchinson=64, seed=0, causal=False)
        w = diag.witness(Q, K, L)
        assert w.shape == (L,)
        # ||witness||_2 == G from the same seeded probes (up to float)
        G = diag.reduce(Q, K, L)["features"].G
        assert abs(w.norm().item() - G) < 1e-5

    def test_witness_materialized_norm_equals_G(self, qk):
        Q, K = qk
        diag = AsymmetryDiagnostic(threshold=1024, causal=False)
        w = diag.witness(Q, K, L)
        G = diag.reduce(Q, K, L)["features"].G
        assert w.shape == (L,)
        assert abs(w.norm().item() - G) < 1e-6

    def test_seeded_determinism(self, qk):
        Q, K = qk
        diag = AsymmetryDiagnostic(
            threshold=4, block_size=4, n_hutchinson=32, seed=11, causal=False
        )
        a = diag.reduce(Q, K, L)["features"].G
        b = diag.reduce(Q, K, L)["features"].G
        assert a == b

    def test_accumulate_latest_only_when_not_streaming(self):
        diag = AsymmetryDiagnostic()
        local = {"features": "x", "partials": {"S_asym": 1.0, "S_den": 2.0, "n_windows": 1}}
        assert diag.accumulate(local, None) is local

    def test_streaming_accumulate_is_additive_global(self, qk):
        # Two disjoint windows -> global G = sqrt(sum S_asym / sum S_den).
        torch.manual_seed(7)
        Q1, K1 = torch.randn(L, D), torch.randn(L, D)
        Q2, K2 = torch.randn(L, D), torch.randn(L, D)

        diag = AsymmetryDiagnostic(threshold=1024, causal=False, streaming=True)
        r1 = diag.reduce(Q1, K1, L, prior_state=None)
        st = diag.accumulate(r1, None)
        r2 = diag.reduce(Q2, K2, L, prior_state=st)

        p1, p2 = r1["partials"], r2["partials"]
        assert p1["n_windows"] == 1 and p2["n_windows"] == 2

        # additivity of the sufficient statistics
        solo2 = AsymmetryDiagnostic(threshold=1024, causal=False).reduce(Q2, K2, L)["partials"]
        assert abs(p2["S_asym"] - (p1["S_asym"] + solo2["S_asym"])) < 1e-6
        assert abs(p2["S_den"] - (p1["S_den"] + solo2["S_den"])) < 1e-6

        # reported global G equals the ratio of accumulated statistics
        g_global = (p2["S_asym"] ** 0.5) / (p2["S_den"] ** 0.5 + 1e-10)
        assert abs(r2["features"].G - g_global) < 1e-9

    def test_batch_reduce_is_per_window(self, qk):
        # Without streaming, prior_state is ignored: each window reports its own G.
        Q, K = qk
        diag = AsymmetryDiagnostic(threshold=1024, causal=False, streaming=False)
        prior = {"partials": {"S_asym": 99.0, "S_den": 1.0, "n_windows": 5}}
        r = diag.reduce(Q, K, L, prior_state=prior)
        assert r["partials"]["n_windows"] == 1

    def test_causal_materialized_matches_oracle(self, qk):
        # causal=True (the default) materialized path on the attention operator P:
        # check reduce G + witness against the exact P-based oracle.
        Q, K = qk
        scale = 1.0 / (D**0.5)
        scores = Q @ K.T * scale
        scores = scores.masked_fill(~torch.tril(torch.ones(L, L, dtype=torch.bool)), float("-inf"))
        P = torch.softmax(scores, dim=-1)
        G_ref = (((P - P.T) / 2).norm() / P.norm()).item()

        diag = AsymmetryDiagnostic(threshold=1024, causal=True)
        r = diag.reduce(Q, K, L)
        w = diag.witness(Q, K, L)
        assert r["tier"] == "materialized"
        assert abs(r["features"].G - G_ref) < 1e-6
        assert w.shape == (L,)
        assert abs(w.norm().item() - G_ref) < 1e-6

    def test_causal_matrix_free_matches_oracle(self, qk):
        Q, K = qk
        scale = 1.0 / (D**0.5)
        scores = Q @ K.T * scale
        scores = scores.masked_fill(~torch.tril(torch.ones(L, L, dtype=torch.bool)), float("-inf"))
        P = torch.softmax(scores, dim=-1)
        G_ref = (((P - P.T) / 2).norm() / P.norm()).item()

        diag = AsymmetryDiagnostic(threshold=4, block_size=4, n_hutchinson=256, seed=0, causal=True)
        r = diag.reduce(Q, K, L)
        assert r["tier"] == "matrix_free"
        assert abs(r["features"].G - G_ref) < 0.05


class TestAsymmetrySerialization:
    def test_jsonl_round_trip(self):
        # SVDSnapshot with AsymmetryFeatures + witness survives JSONL ser/de,
        # and from_jsonl_row discriminates the 'asymmetry' signal correctly.
        snap = SVDSnapshot(
            signal="asymmetry",
            request_id=0,
            layer="model.layers.0.self_attn",
            layer_idx=0,
            head=0,
            step=32,
            L=16,
            tier="matrix_free",
            witness=[0.1] * 16,
            features=AsymmetryFeatures(G=0.42),
        )
        back = SVDSnapshot.from_jsonl_row(json.loads(snap.model_dump_json()))
        assert isinstance(back.features, AsymmetryFeatures)
        assert back.features.G == 0.42
        assert back.tier == "matrix_free"
        assert back.witness == [0.1] * 16
        assert "G=0.42" in repr(back)  # __repr__ renders the feature


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


class TestCheegerDiagnostic:
    def test_reduce_materialized(self, qk):
        Q, K = qk
        diag = CheegerDiagnostic(rank=2, threshold=1024)
        result = diag.reduce(Q, K, L)
        assert "features" in result
        assert result["tier"] == "materialized"
        assert 0.0 <= result["features"].phi_star <= 1.0

    def test_reduce_matrix_free(self, qk):
        Q, K = qk
        diag = CheegerDiagnostic(rank=2, threshold=4, block_size=4)
        result = diag.reduce(Q, K, L)
        assert "features" in result
        assert result["tier"] == "matrix_free"
        assert 0.0 <= result["features"].phi_star <= 1.0

    def test_witness_materialized(self, qk):
        Q, K = qk
        diag = CheegerDiagnostic(rank=2, threshold=1024)
        w = diag.witness(Q, K, L)
        assert w.shape == (L,)
        assert set(w.tolist()).issubset({-1, 1})

    def test_witness_matrix_free(self, qk):
        Q, K = qk
        diag = CheegerDiagnostic(rank=2, threshold=4, block_size=4)
        w = diag.witness(Q, K, L)
        assert w.shape == (L,)
        assert set(w.tolist()).issubset({-1, 1})

    def test_signal_name(self):
        assert CheegerDiagnostic.signal_name == "cheeger"

    def test_accumulate_returns_local(self):
        diag = CheegerDiagnostic()
        local = {"features": "test"}
        assert diag.accumulate(local, None) is local
        assert diag.accumulate(local, {"old": True}) is local

    def test_cheeger_bounds(self, qk):
        Q, K = qk
        diag = CheegerDiagnostic(rank=2, threshold=1024)
        result = diag.reduce(Q, K, L)
        f = result["features"]
        assert f.cheeger_lower - 1e-6 <= f.phi_star <= f.cheeger_upper + 1e-6


class TestEmitWitnessConfig:
    def test_default_false(self):
        cfg = GlassboxConfig()
        assert cfg.emit_witness is False

    def test_set_true(self):
        cfg = GlassboxConfig(emit_witness=True)
        assert cfg.emit_witness is True
