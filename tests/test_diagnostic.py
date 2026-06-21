"""Tests for the Diagnostic protocol and concrete implementations."""

import itertools
import json

import pytest
import torch

from glassbox.config import GlassboxConfig
from glassbox.diagnostic import Diagnostic
from glassbox.diagnostics import (
    DIAGNOSTIC_REGISTRY,
    CyclicTrianglesDiagnostic,
    LaplacianDiagnostic,
    RoutingDiagnostic,
    SelfAttnDiagnostic,
    SpectralDiagnostic,
    TrackerDiagnostic,
)
from glassbox.results import CyclicTrianglesFeatures, SpectralFeatures, SVDSnapshot

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
            "cyclic",
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


class TestCyclicTriangles:
    """|T_cyc| on the pre-softmax sign tournament ω(QKᵀ) — exact integer count."""

    def _brute(self, Q, K, n):
        # O(n^3) reference: a triple is cyclic iff each vertex has out-degree 1 within it.
        S = Q[:n] @ K[:n].T
        Dm = S - S.T  # Δ_ij = q_i·k_j − q_j·k_i

        def beats(a, b):
            return bool(Dm[a, b] > 0) or (float(Dm[a, b]) == 0.0 and a < b)

        c = 0
        for i, j, k in itertools.combinations(range(n), 3):
            outs = sorted(sum(beats(a, b) for b in (i, j, k) if b != a) for a in (i, j, k))
            if outs == [1, 1, 1]:
                c += 1
        return c

    def test_signal_name(self):
        assert CyclicTrianglesDiagnostic.signal_name == "cyclic"

    def test_batch_matches_brute_force(self, qk):
        Q, K = qk
        r = CyclicTrianglesDiagnostic().reduce(Q, K, L)
        assert r["tier"] == "materialized"
        assert r["features"].T_cyc == self._brute(Q, K, L)

    def test_incremental_matches_batch_at_each_fire(self):
        torch.manual_seed(1)
        N = 40
        Q, K = torch.randn(N, D), torch.randn(N, D)
        diag = CyclicTrianglesDiagnostic(incremental=True)
        state = None
        for fire_L in (8, 17, N):
            res = diag.reduce(Q[:fire_L], K[:fire_L], fire_L, prior_state=state)
            state = diag.accumulate(res, state)
            assert res["tier"] == "incremental"
            assert res["partials"]["tcyc"]["n"] == fire_L
            assert res["features"].T_cyc == self._brute(Q, K, fire_L)

    def test_witness_sums_to_count(self, qk):
        Q, K = qk
        diag = CyclicTrianglesDiagnostic()
        w = diag.witness(Q, K, L)
        assert w.shape == (L,)
        assert int(w.sum().item()) == diag.reduce(Q, K, L)["features"].T_cyc

    def test_transitive_tournament_is_zero(self):
        # q_i=[a_i,1], k_j=[1,0] ⇒ q_i·k_j = a_i ⇒ Δ_ij = a_i − a_j ⇒ total order ⇒ transitive.
        a = torch.arange(L, dtype=torch.float32)
        Q = torch.stack([a, torch.ones(L)], dim=1)
        K = torch.stack([torch.ones(L), torch.zeros(L)], dim=1)
        assert CyclicTrianglesDiagnostic().reduce(Q, K, L)["features"].T_cyc == 0

    def test_ties_oriented_by_index(self):
        # identical tokens ⇒ Δ ≡ 0 ⇒ every pair oriented by index ⇒ transitive ⇒ |T_cyc| = 0.
        Q = torch.ones(L, D)
        K = torch.ones(L, D)
        assert CyclicTrianglesDiagnostic().reduce(Q, K, L)["features"].T_cyc == 0

    def test_serialization_round_trip(self):
        snap = SVDSnapshot(
            signal="cyclic",
            request_id=0,
            layer="model.layers.0.self_attn",
            layer_idx=0,
            head=0,
            step=8,
            L=16,
            tier="materialized",
            features=CyclicTrianglesFeatures(T_cyc=142),
        )
        back = SVDSnapshot.from_jsonl_row(json.loads(snap.model_dump_json()))
        assert isinstance(back.features, CyclicTrianglesFeatures)
        assert back.features.T_cyc == 142
