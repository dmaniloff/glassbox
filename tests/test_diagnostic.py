"""Tests for the Diagnostic protocol and concrete implementations."""

import itertools
import json

import pytest
import torch

from glassbox.config import GlassboxConfig
from glassbox.diagnostic import Diagnostic
from glassbox.diagnostics import (
    DIAGNOSTIC_REGISTRY,
    AsymmetryDiagnostic,
    CyclicTrianglesDiagnostic,
    LaplacianDiagnostic,
    MagneticDiagnostic,
    RoutingDiagnostic,
    SelfAttnDiagnostic,
    SpectralDiagnostic,
    TrackerDiagnostic,
)
from glassbox.results import (
    AsymmetryFeatures,
    CyclicTrianglesFeatures,
    MagneticFeatures,
    SpectralFeatures,
    SVDSnapshot,
)

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
            "cyclic",
            "magnetic",
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

    def test_one_token_streaming_matches_batch(self):
        # Mode 1: global statistic, one token at a time (interval=1). Cold-start at t=1, then
        # the O(ΔE) per-token update; the running |T_cyc| matches batch at every step.
        torch.manual_seed(3)
        N = 24
        Q, K = torch.randn(N, D), torch.randn(N, D)
        diag = CyclicTrianglesDiagnostic(incremental=True)
        state = None
        for t in range(1, N + 1):
            res = diag.reduce(Q[:t], K[:t], t, prior_state=state)
            state = diag.accumulate(res, state)
            assert res["features"].T_cyc == self._brute(Q, K, t)

    def test_block_local_not_additive(self):
        # Mode 2: block-local per-window counts. |T_cyc| is NOT additive across disjoint
        # windows (cyclic triangles span boundaries), so per-window counts must NOT be summed
        # into a global — only the full-sequence count (mode 1) is the true global |T_cyc|.
        torch.manual_seed(7)
        N = 30
        Q, K = torch.randn(N, D), torch.randn(N, D)
        diag = CyclicTrianglesDiagnostic()  # batch = block-local
        full = diag.reduce(Q, K, N)["features"].T_cyc
        half = N // 2
        w1 = diag.reduce(Q[:half], K[:half], half)["features"].T_cyc
        w2 = diag.reduce(Q[half:], K[half:], N - half)["features"].T_cyc
        assert full == self._brute(Q, K, N)
        assert w1 + w2 < full  # block sum undercounts: cross-window triangles are dropped


class TestMagnetic:
    """Magnetic-Laplacian frustration λ₁ on the pre-softmax tournament (unmasked S=QKᵀ)."""

    def _dense_lambda1(self, Q, K):
        # Independent dense oracle: L_φ = D − A⊙e^{iθ}, λ₁ = smallest eigenvalue (clamped ≥0).
        scale = 1.0 / (D**0.5)
        S = Q @ K.T * scale
        St = S.T
        W = (S.abs() + St.abs()) / 2.0
        W = W - torch.diag(torch.diagonal(W))
        denom = S + St
        safe = torch.where(denom != 0, denom, torch.ones_like(denom))
        theta = torch.where(denom != 0, torch.atan((S - St) / safe), torch.zeros_like(denom))
        A = torch.complex(W * torch.cos(theta), W * torch.sin(theta))
        A = A - torch.diag(torch.diagonal(A))
        Lphi = torch.diag(W.sum(1)).to(A.dtype) - A
        return max(0.0, float(torch.linalg.eigvalsh(Lphi)[0].item()))

    def test_signal_name(self):
        assert MagneticDiagnostic.signal_name == "magnetic"

    def test_psd_and_frustrated(self, qk):
        Q, K = qk
        f = MagneticDiagnostic(threshold=1024).reduce(Q, K, L)["features"]
        assert f.frustration >= 0.0  # L_φ is PSD
        assert f.frustration > 1e-4  # generic asymmetric scores are frustrated

    def test_matches_dense_oracle(self, qk):
        Q, K = qk
        r = MagneticDiagnostic(threshold=1024).reduce(Q, K, L)
        assert r["tier"] == "materialized"
        assert abs(r["features"].frustration - self._dense_lambda1(Q, K)) < 1e-5

    def test_balanced_symmetric_is_zero(self):
        # Q=K -> S=Q@Qᵀ symmetric -> θ=0 -> L_φ is the real graph Laplacian -> λ₁=0 (balanced).
        torch.manual_seed(1)
        Q = torch.randn(L, D)
        f = MagneticDiagnostic(threshold=1024).reduce(Q, Q, L)["features"]
        assert f.frustration < 1e-4

    def test_matrix_free_matches_materialized(self):
        torch.manual_seed(2)
        Q, K = torch.randn(L, D), torch.randn(L, D)
        mat = MagneticDiagnostic(threshold=1024).reduce(Q, K, L)
        mf = MagneticDiagnostic(threshold=4, block_size=8).reduce(Q, K, L)
        assert mf["tier"] == "matrix_free"
        assert abs(mat["features"].frustration - mf["features"].frustration) < 1e-3

    def test_witness_is_bottom_eigenvector_magnitude(self, qk):
        Q, K = qk
        w = MagneticDiagnostic(threshold=1024).witness(Q, K, L)
        assert w.shape == (L,)
        assert bool(torch.isfinite(w).all()) and float(w.min()) >= 0.0

    def test_nonfinite_scrubbed(self):
        assert MagneticFeatures(frustration=float("nan")).frustration is None
        assert MagneticFeatures(frustration=2.5).frustration == 2.5

    def test_serialization_round_trip(self):
        snap = SVDSnapshot(
            signal="magnetic",
            request_id=0,
            layer="model.layers.0.self_attn",
            layer_idx=0,
            head=0,
            step=8,
            L=16,
            tier="materialized",
            features=MagneticFeatures(frustration=0.73),
        )
        back = SVDSnapshot.from_jsonl_row(json.loads(snap.model_dump_json()))
        assert isinstance(back.features, MagneticFeatures)
        assert back.features.frustration == 0.73


class TestAsymmetryDiagnostic:
    """G = ||P_asym||_F / ||P||_F on the row-stochastic attention P (NOT degree-normalized M)."""

    def _g_on_p(self, Q, K, causal=False):
        scale = 1.0 / (D**0.5)
        scores = Q @ K.T * scale
        if causal:
            mask = ~torch.tril(torch.ones(L, L, dtype=torch.bool))
            scores = scores.masked_fill(mask, float("-inf"))
        P = torch.softmax(scores, dim=-1)
        return (((P - P.T) / 2).norm() / P.norm()).item()

    def test_signal_name(self):
        assert AsymmetryDiagnostic.signal_name == "asymmetry"

    def test_reduce_materialized_matches_oracle(self, qk):
        Q, K = qk
        diag = AsymmetryDiagnostic(threshold=1024, causal=False)
        result = diag.reduce(Q, K, L)
        assert result["tier"] == "materialized"
        assert abs(result["features"].G - self._g_on_p(Q, K)) < 1e-6

    def test_reduce_matrix_free_matches_oracle(self, qk):
        Q, K = qk
        diag = AsymmetryDiagnostic(
            threshold=4, block_size=4, n_hutchinson=128, seed=0, causal=False
        )
        result = diag.reduce(Q, K, L)
        assert result["tier"] == "matrix_free"
        assert abs(result["features"].G - self._g_on_p(Q, K)) < 0.05

    def test_causal_materialized_matches_oracle(self, qk):
        Q, K = qk
        diag = AsymmetryDiagnostic(threshold=1024, causal=True)
        r = diag.reduce(Q, K, L)
        w = diag.witness(Q, K, L)
        g = self._g_on_p(Q, K, causal=True)
        assert r["tier"] == "materialized"
        assert abs(r["features"].G - g) < 1e-6
        assert w.shape == (L,) and abs(w.norm().item() - g) < 1e-6

    def test_witness_consistency(self, qk):
        Q, K = qk
        diag = AsymmetryDiagnostic(threshold=4, block_size=4, n_hutchinson=64, seed=0, causal=False)
        w = diag.witness(Q, K, L)
        assert w.shape == (L,)
        assert abs(w.norm().item() - diag.reduce(Q, K, L)["features"].G) < 1e-5

    def test_seeded_determinism(self, qk):
        Q, K = qk
        diag = AsymmetryDiagnostic(
            threshold=4, block_size=4, n_hutchinson=32, seed=11, causal=False
        )
        assert diag.reduce(Q, K, L)["features"].G == diag.reduce(Q, K, L)["features"].G

    def test_accumulate_latest_only(self):
        diag = AsymmetryDiagnostic()
        local = {"features": "x", "partials": {"S_asym": 1.0, "S_den": 2.0, "n_windows": 1}}
        assert diag.accumulate(local, None) is local

    def test_streaming_accumulate_is_additive_global(self):
        torch.manual_seed(7)
        Q1, K1 = torch.randn(L, D), torch.randn(L, D)
        Q2, K2 = torch.randn(L, D), torch.randn(L, D)
        diag = AsymmetryDiagnostic(threshold=1024, causal=False, streaming=True)
        r1 = diag.reduce(Q1, K1, L, prior_state=None)
        st = diag.accumulate(r1, None)
        r2 = diag.reduce(Q2, K2, L, prior_state=st)
        p1, p2 = r1["partials"], r2["partials"]
        assert p1["n_windows"] == 1 and p2["n_windows"] == 2
        solo2 = AsymmetryDiagnostic(threshold=1024, causal=False).reduce(Q2, K2, L)["partials"]
        assert abs(p2["S_asym"] - (p1["S_asym"] + solo2["S_asym"])) < 1e-6
        assert abs(p2["S_den"] - (p1["S_den"] + solo2["S_den"])) < 1e-6
        g_global = (p2["S_asym"] ** 0.5) / (p2["S_den"] ** 0.5 + 1e-10)
        assert abs(r2["features"].G - g_global) < 1e-9


class TestAsymmetrySerialization:
    def test_jsonl_round_trip(self):
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
        assert back.features.G == 0.42 and back.tier == "matrix_free" and back.witness == [0.1] * 16
