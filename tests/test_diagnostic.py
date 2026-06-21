"""Tests for the Diagnostic protocol and concrete implementations."""

import json

import pytest
import torch

from glassbox.config import GlassboxConfig
from glassbox.diagnostic import Diagnostic
from glassbox.diagnostics import (
    DIAGNOSTIC_REGISTRY,
    LaplacianDiagnostic,
    MagneticDiagnostic,
    RoutingDiagnostic,
    SelfAttnDiagnostic,
    SpectralDiagnostic,
    TrackerDiagnostic,
)
from glassbox.results import MagneticFeatures, SpectralFeatures, SVDSnapshot

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


class TestMagneticStreaming:
    """Streamable frustration: phase-field Hodge curl energy (eigensolver-free, issue #68)."""

    def _phase_theta_W(self, Q, K):
        scale = 1.0 / (D**0.5)
        S = Q @ K.T * scale
        denom = S + S.T
        safe = torch.where(denom != 0, denom, torch.ones_like(denom))
        theta = torch.where(denom != 0, torch.atan((S - S.T) / safe), torch.zeros_like(denom))
        W = (S.abs() + S.T.abs()) / 2.0
        W = W - torch.diag(torch.diagonal(W))
        return theta, W

    def _explicit_curlE(self, Q, K):  # unweighted Hodge curl oracle
        theta, _ = self._phase_theta_W(Q, K)
        phi = theta.sum(1) / Q.shape[0]
        grad = phi[:, None] - phi[None, :]
        return float(((theta - grad) ** 2).sum())

    def _explicit_curlE_w(self, Q, K):  # Jacobi W-weighted curl oracle
        theta, W = self._phase_theta_W(Q, K)
        Wth = W * theta
        b = Wth.sum(1)
        d = W.sum(1)
        return max(0.0, float((Wth * theta).sum()) - 2 * float((b * b / (d + 1e-10)).sum()))

    def test_batch_reports_lambda1_and_both_curls(self, qk):
        Q, K = qk
        f = MagneticDiagnostic(threshold=1024).reduce(Q, K, L)["features"]
        assert f.frustration is not None and f.frustration >= 0.0
        assert abs(f.phase_curl - self._explicit_curlE(Q, K)) < 1e-2
        assert abs(f.phase_curl_w - self._explicit_curlE_w(Q, K)) < 1e-2

    def test_incremental_matches_batch(self):
        torch.manual_seed(3)
        N = 28
        Q, K = torch.randn(N, D), torch.randn(N, D)
        b = MagneticDiagnostic(threshold=1024).reduce(Q, K, N)["features"]
        diag = MagneticDiagnostic(incremental=True)
        state = None
        for t in range(1, N + 1):  # one token at a time
            r = diag.reduce(Q[:t], K[:t], t, prior_state=state)
            state = diag.accumulate(r, state)
        assert r["tier"] == "incremental"
        assert r["features"].frustration is None  # eigensolver-free
        assert abs(r["features"].phase_curl - b.phase_curl) < 1e-2
        assert abs(r["features"].phase_curl_w - b.phase_curl_w) < 1e-2  # weighted streams too

    def test_balanced_curls_are_zero(self):
        # Q=K -> symmetric S -> theta=0 -> both curls = 0 (balanced, matches lambda1=0).
        torch.manual_seed(1)
        Q = torch.randn(L, D)
        f = MagneticDiagnostic(incremental=True).reduce(Q, Q, L)["features"]
        assert f.phase_curl < 1e-5 and f.phase_curl_w < 1e-5

    def test_phase_curl_w_tracks_lambda1_better(self):
        # The whole point: the W-weighted curl is a tighter lambda1 proxy than the unweighted.
        lam, pc, pcw = [], [], []
        for seed in range(40):
            torch.manual_seed(seed)
            Q, K = torch.randn(16, D), torch.randn(16, D)
            f = MagneticDiagnostic(threshold=1024).reduce(Q, K, 16)["features"]
            lam.append(f.frustration)
            pc.append(f.phase_curl)
            pcw.append(f.phase_curl_w)

        def spearman(a, b):
            ra = torch.tensor(a).argsort().argsort().float()
            rb = torch.tensor(b).argsort().argsort().float()
            ra -= ra.mean()
            rb -= rb.mean()
            return float((ra @ rb) / (ra.norm() * rb.norm()))

        assert spearman(lam, pcw) > spearman(lam, pc)
