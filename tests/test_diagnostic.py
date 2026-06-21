"""Tests for the Diagnostic protocol and concrete implementations."""

import json

import pydantic
import pytest
import torch

from glassbox.config import AsymmetryConfig, GlassboxConfig
from glassbox.diagnostic import Diagnostic
from glassbox.diagnostics import (
    DIAGNOSTIC_REGISTRY,
    AsymmetryDiagnostic,
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


class TestAsymmetryIncremental:
    """Incremental exact-full G: fold delta tokens per fire; matches batch full-operator G."""

    def _batch_g_causal(self, Q, K, n):
        scale = 1.0 / (D**0.5)
        scores = Q[:n] @ K[:n].T * scale
        scores = scores.masked_fill(~torch.tril(torch.ones(n, n, dtype=torch.bool)), float("-inf"))
        P = torch.softmax(scores, dim=-1)
        return (((P - P.T) / 2).norm() / P.norm()).item()

    def test_incremental_matches_batch_at_each_fire(self):
        torch.manual_seed(0)
        N = 24
        Q, K = torch.randn(N, D), torch.randn(N, D)
        diag = AsymmetryDiagnostic(causal=True, incremental=True)
        state = None
        for fire_L in (4, 9, 16, N):  # growing full-sequence buffer
            r = diag.reduce(Q[:fire_L], K[:fire_L], fire_L, prior_state=state)
            state = diag.accumulate(r, state)
            assert r["tier"] == "incremental"
            assert r["partials"]["incr"]["n"] == fire_L  # folded up to current length
            assert abs(r["features"].G - self._batch_g_causal(Q, K, fire_L)) < 1e-6

    def test_delta_fold_equals_single_shot(self):
        # folding in chunks == folding the whole sequence at once (delta composition)
        torch.manual_seed(1)
        N = 20
        Q, K = torch.randn(N, D), torch.randn(N, D)
        chunked = AsymmetryDiagnostic(causal=True, incremental=True)
        state = None
        for fire_L in (5, 11, N):
            r = chunked.reduce(Q[:fire_L], K[:fire_L], fire_L, prior_state=state)
            state = chunked.accumulate(r, state)
        fresh = AsymmetryDiagnostic(causal=True, incremental=True)
        one_shot = fresh.reduce(Q, K, N, prior_state=None)
        # fp32 block reductions sum in different orders chunked vs single-shot (~1e-8)
        assert abs(r["features"].G - one_shot["features"].G) < 1e-6


class TestAsymmetryCurlSplit:
    """Hodge gradient/curl split G^2 = Gamma^2 + C^2 on causal P (Gamma exact via row-sum)."""

    def _oracle(self, Q, K, n):
        # Independent Hodge decomposition: potential phi = A.sum(1)/n, A_grad[i,j]=phi_i-phi_j.
        scale = 1.0 / (D**0.5)
        scores = Q[:n] @ K[:n].T * scale
        scores = scores.masked_fill(~torch.tril(torch.ones(n, n, dtype=torch.bool)), float("-inf"))
        P = torch.softmax(scores, dim=-1)
        A = (P - P.T) / 2
        phi = A.sum(dim=1) / n
        A_grad = phi[:, None] - phi[None, :]
        pn = P.norm().item()
        return A.norm().item() / pn, A_grad.norm().item() / pn, (A - A_grad).norm().item() / pn

    def test_materialized_matches_hodge_oracle(self, qk):
        Q, K = qk
        f = AsymmetryDiagnostic(threshold=1024, causal=True).reduce(Q, K, L)["features"]
        g, gamma, c = self._oracle(Q, K, L)
        assert abs(f.G - g) < 1e-6
        assert abs(f.Gamma - gamma) < 1e-6
        assert abs(f.C - c) < 1e-6

    def test_pythagorean_identity(self, qk):
        Q, K = qk
        f = AsymmetryDiagnostic(threshold=1024, causal=True).reduce(Q, K, L)["features"]
        assert abs(f.G**2 - (f.Gamma**2 + f.C**2)) < 1e-9
        assert f.Gamma >= 0.0 and f.C >= 0.0

    def test_matrix_free_gamma_is_exact(self, qk):
        # Gamma uses r = A_asym @ 1 (one exact matvec), so it is exact even matrix-free;
        # only G and C carry Hutchinson noise from S_asym.
        Q, K = qk
        diag = AsymmetryDiagnostic(threshold=4, block_size=4, n_hutchinson=128, seed=0, causal=True)
        f = diag.reduce(Q, K, L)["features"]
        g, gamma, c = self._oracle(Q, K, L)
        assert f.G > 0 and diag.reduce(Q, K, L)["tier"] == "matrix_free"
        assert abs(f.Gamma - gamma) < 1e-5
        assert abs(f.G - g) < 0.05

    def test_incremental_split_matches_batch_at_each_fire(self):
        torch.manual_seed(0)
        N = 24
        Q, K = torch.randn(N, D), torch.randn(N, D)
        diag = AsymmetryDiagnostic(causal=True, incremental=True)
        state = None
        for fire_L in (4, 9, 16, N):
            res = diag.reduce(Q[:fire_L], K[:fire_L], fire_L, prior_state=state)
            state = diag.accumulate(res, state)
            f = res["features"]
            g, gamma, c = self._oracle(Q, K, fire_L)
            assert abs(f.G - g) < 1e-6
            assert abs(f.Gamma - gamma) < 1e-6
            assert abs(f.C - c) < 1e-6
            assert abs(f.G**2 - (f.Gamma**2 + f.C**2)) < 1e-9

    def test_split_serialization_round_trip(self):
        snap = SVDSnapshot(
            signal="asymmetry",
            request_id=0,
            layer="model.layers.0.self_attn",
            layer_idx=0,
            head=0,
            step=8,
            L=8,
            tier="incremental",
            features=AsymmetryFeatures(G=0.5, Gamma=0.3, C=0.4),
        )
        back = SVDSnapshot.from_jsonl_row(json.loads(snap.model_dump_json()))
        assert isinstance(back.features, AsymmetryFeatures)
        assert back.features.Gamma == 0.3 and back.features.C == 0.4


class TestAsymmetryRobustness:
    """Guards for the security/robustness + faithfulness fixes (audit P0/P2)."""

    @pytest.mark.parametrize(
        "bad", [{"n_hutchinson": 0}, {"n_hutchinson": -1}, {"block_size": 0}, {"threshold": -1}]
    )
    def test_config_rejects_out_of_bounds(self, bad):
        # block_size=0 crashes range(); n_hutchinson=0 divides by zero -> NaN.
        with pytest.raises(pydantic.ValidationError):
            AsymmetryConfig(**bad)

    def test_features_scrub_nonfinite(self):
        # max(x, 0.0) does NOT scrub NaN, so the model validator is the choke point.
        f = AsymmetryFeatures(G=float("nan"), Gamma=float("inf"), C=float("-inf"))
        assert f.G is None and f.Gamma is None and f.C is None
        ok = AsymmetryFeatures(G=0.5, Gamma=0.3, C=0.4)
        assert ok.G == 0.5 and ok.Gamma == 0.3 and ok.C == 0.4

    def test_reduce_clears_stale_cache(self, qk):
        # Simulate a data_ptr address reused by a freed tensor of the same shape: a poisoned
        # entry whose KEY matches this input. The per-reduce clear must discard it.
        Q, K = qk
        diag = AsymmetryDiagnostic(threshold=1024, causal=True)
        real = diag.reduce(Q, K, L)["features"].G
        key = (Q.data_ptr(), K.data_ptr(), tuple(Q.shape), tuple(K.shape), L, diag.n_hutchinson)
        diag._cache = (key, (999.0, 1.0, 0.0, torch.zeros(L), "materialized"))
        assert abs(diag.reduce(Q, K, L)["features"].G - real) < 1e-6

    def test_streaming_witness_reuses_reduce_probes(self):
        # Past the first window, reduce() uses seed self.seed+n_windows but witness() uses
        # self.seed. With seed dropped from the cache key, witness reuses reduce's exact
        # probes, so ||witness||_2 == the per-window G. (Old code recomputed -> mismatch.)
        torch.manual_seed(5)
        Q1, K1 = torch.randn(L, D), torch.randn(L, D)
        Q2, K2 = torch.randn(L, D), torch.randn(L, D)
        diag = AsymmetryDiagnostic(
            threshold=4, block_size=4, n_hutchinson=64, seed=1, causal=True, streaming=True
        )
        r1 = diag.reduce(Q1, K1, L, prior_state=None)
        st = diag.accumulate(r1, None)
        r2 = diag.reduce(Q2, K2, L, prior_state=st)  # window 2: seed = 1 + 1
        w2 = diag.witness(Q2, K2, L)  # must reuse r2's cached pass (seed 2 probes)
        win_asym = r2["partials"]["S_asym"] - r1["partials"]["S_asym"]
        win_den = r2["partials"]["S_den"] - r1["partials"]["S_den"]
        expected = (win_asym**0.5) / (win_den**0.5)
        assert abs(w2.norm().item() - expected) < 1e-4

    def test_incremental_cold_state_recompute_is_correct(self):
        # A dropped prior_state restarts at n_prev=0 (full recompute) but stays correct.
        torch.manual_seed(2)
        Q, K = torch.randn(20, D), torch.randn(20, D)
        diag = AsymmetryDiagnostic(causal=True, incremental=True)
        warm = diag.reduce(Q, K, 20, prior_state=None)["features"].G
        cold = diag.reduce(Q, K, 20, prior_state=None)["features"].G  # no threaded state
        assert abs(warm - cold) < 1e-6
