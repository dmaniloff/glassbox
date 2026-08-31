"""Comprehensive test suite for matrix-free Hodge decomposition.

Organized into 12 groups that together establish mathematical faithfulness
of the matrix-free implementation against materialized references and
Hodge-theoretic identities.
"""

import math

import pytest
import torch
from conftest import make_M

from glassbox.hodge import (
    compute_G_materialized,
    compute_G_matrix_free,
    compute_routing_features_materialized,
    compute_routing_features_matrix_free,
    compute_sigma2_asym_matrix_free,
    estimate_commutator_norm_matrix_free,
)
from glassbox.svd import (
    compute_dk_blocked,
    compute_M_fro_norm_blocked,
    matvec_commutator_blocked,
    matvec_M_blocked,
    matvec_Masym_blocked,
    matvec_Msym_blocked,
)

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


# ===========================================================================
# Group 4: Asymmetry Coefficient G
# ===========================================================================
#
# Note: the Hodge gradient/curl split (Gamma, C) is NOT computed on M — degree
# normalization leaks the symmetric routing into the antisymmetric channel entry-wise,
# so the split lives only in the asymmetry signal on P (see
# tests/test_diagnostic.py::TestAsymmetryCurlSplit and docs/operator-choice.md). On M we
# keep only the scalar asymmetry index asym_index = ||M_asym||_F / ||M||_F, tested here
# via the low-level compute_G_* funcs.


class TestAsymmetryG:
    def test_matrix_free_matches_materialized(self):
        Q, K, scale, A, M, d_k_inv_sqrt = make_M(16, 4)
        G_ref, fro_ref = compute_G_materialized(M)
        _, d_k_mf = compute_dk_blocked(Q, K, scale)
        G_mf, fro_mf = compute_G_matrix_free(Q, K, d_k_mf, scale, block_size=4)
        assert abs(G_ref - G_mf) < 0.01, f"G: ref={G_ref}, mf={G_mf}"
        assert abs(fro_ref - fro_mf) < 0.01, f"Fro: ref={fro_ref}, mf={fro_mf}"

    def test_symmetric_is_small(self):
        """Q=K gives nearly-symmetric M (softmax(QQ^T) is symmetric, but
        degree normalization D_Q^{-1/2} A D_K^{-1/2} may not be exactly
        symmetric since the matrix-free path uses D_Q = I)."""
        torch.manual_seed(42)
        Q = torch.randn(10, 4)
        K = Q.clone()
        scale = 1.0 / math.sqrt(4)
        _, d_k_mf = compute_dk_blocked(Q, K, scale)
        G, _ = compute_G_matrix_free(Q, K, d_k_mf, scale, block_size=4)
        assert G < 0.25  # small but not exactly zero due to normalization

    def test_algebraic_identity(self):
        """||M_asym||²_F = (||M||²_F - <M,M^T>_F) / 2"""
        Q, K, scale, A, M, d_k_inv_sqrt = make_M(10, 4)
        M_asym = (M - M.T) / 2.0
        lhs = torch.linalg.norm(M_asym, "fro").square()
        M_fro_sq = torch.linalg.norm(M, "fro").square()
        inner = (M * M.T).sum()
        rhs = (M_fro_sq - inner) / 2.0
        assert abs(lhs.item() - rhs.item()) < 1e-6


# ===========================================================================
# Group 5: Spectral Features — σ₂(M) and φ̂
# ===========================================================================


class TestSpectral:
    def test_sigma2_matches_full_svd(self):
        Q, K, scale, A, M, d_k_inv_sqrt = make_M(16, 4)
        sigma_ref = torch.linalg.svdvals(M)
        s2_ref = sigma_ref[1].item()
        _, d_k_mf = compute_dk_blocked(Q, K, scale)
        f = compute_routing_features_matrix_free(Q, K, d_k_mf, scale, rank=4)
        assert abs(s2_ref - f.sigma2) < 0.05

    def test_phi_hat_range(self):
        for seed in range(5):
            Q, K, scale, A, M, d_k_inv_sqrt = make_M(16, 4, seed=seed)
            _, d_k_mf = compute_dk_blocked(Q, K, scale)
            f = compute_routing_features_matrix_free(Q, K, d_k_mf, scale, rank=2)
            assert 0.0 <= f.phi_hat <= 1.0


# ===========================================================================
# Group 7: σ₂(M_asym) — Matrix-Free
# ===========================================================================


class TestSigma2Asym:
    def test_matches_materialized(self):
        Q, K, scale, A, M, d_k_inv_sqrt = make_M(16, 4)
        M_asym = (M - M.T) / 2.0
        sigma_ref = torch.linalg.svdvals(M_asym)
        s2_ref = sigma_ref[1].item() if len(sigma_ref) > 1 else 0.0
        _, d_k_mf = compute_dk_blocked(Q, K, scale)
        s2_mf = compute_sigma2_asym_matrix_free(Q, K, d_k_mf, scale, block_size=4)
        assert abs(s2_ref - s2_mf) < 0.05, f"ref={s2_ref}, mf={s2_mf}"

    def test_symmetric_is_small(self):
        """Q=K gives nearly-symmetric M; sigma2_asym should be small."""
        torch.manual_seed(42)
        Q = torch.randn(10, 4)
        K = Q.clone()
        scale = 1.0 / math.sqrt(4)
        _, d_k_mf = compute_dk_blocked(Q, K, scale)
        s2 = compute_sigma2_asym_matrix_free(Q, K, d_k_mf, scale, block_size=4)
        assert s2 < 0.25

    def test_antisymmetric_property(self):
        """<M_asym·v, w> = -<v, M_asym·w> for random v, w."""
        Q, K, scale, A, M, d_k_inv_sqrt = make_M(10, 4)
        _, d_k_mf = compute_dk_blocked(Q, K, scale)
        torch.manual_seed(0)
        v = torch.randn(10)
        w = torch.randn(10)
        Av = matvec_Masym_blocked(Q, K, v, d_k_mf, scale, block_size=4)
        Aw = matvec_Masym_blocked(Q, K, w, d_k_mf, scale, block_size=4)
        lhs = Av.dot(w)
        rhs = v.dot(Aw)
        assert abs(lhs.item() + rhs.item()) < 1e-4, f"<Av,w>={lhs.item()}, <v,Aw>={rhs.item()}"

    def test_multiple_seeds(self):
        for seed in range(5):
            Q, K, scale, A, M, d_k_inv_sqrt = make_M(12, 4, seed=seed)
            M_asym = (M - M.T) / 2.0
            sigma_ref = torch.linalg.svdvals(M_asym)
            s2_ref = sigma_ref[1].item() if len(sigma_ref) > 1 else 0.0
            _, d_k_mf = compute_dk_blocked(Q, K, scale)
            s2_mf = compute_sigma2_asym_matrix_free(Q, K, d_k_mf, scale, block_size=4)
            assert abs(s2_ref - s2_mf) < 0.05, f"seed={seed}: ref={s2_ref}, mf={s2_mf}"


# ===========================================================================
# Group 8: Commutator Norm — Hutchinson Trace Estimation
# ===========================================================================


class TestCommutatorNorm:
    def test_matches_materialized(self):
        Q, K, scale, A, M, d_k_inv_sqrt = make_M(16, 4)
        M_sym = (M + M.T) / 2.0
        M_asym = (M - M.T) / 2.0
        comm = M_sym @ M_asym - M_asym @ M_sym
        ref = torch.linalg.norm(comm, "fro").item() / torch.linalg.norm(M, "fro").item()
        _, d_k_mf = compute_dk_blocked(Q, K, scale)
        M_fro = compute_M_fro_norm_blocked(Q, K, d_k_mf, scale, block_size=4).item()
        mf = estimate_commutator_norm_matrix_free(
            Q,
            K,
            d_k_mf,
            scale,
            M_fro,
            block_size=4,
            n_hutchinson=30,
        )
        assert abs(ref - mf) < 0.1, f"ref={ref}, mf={mf}"

    def test_symmetric_is_small(self):
        """Q=K gives nearly-symmetric M; commutator should be small."""
        torch.manual_seed(42)
        Q = torch.randn(10, 4)
        K = Q.clone()
        scale = 1.0 / math.sqrt(4)
        _, d_k_mf = compute_dk_blocked(Q, K, scale)
        M_fro = compute_M_fro_norm_blocked(Q, K, d_k_mf, scale).item()
        cn = estimate_commutator_norm_matrix_free(
            Q,
            K,
            d_k_mf,
            scale,
            M_fro,
            n_hutchinson=20,
        )
        assert cn < 0.25

    def test_nonneg(self):
        for seed in range(5):
            Q, K, scale, A, M, d_k_inv_sqrt = make_M(10, 4, seed=seed)
            _, d_k_mf = compute_dk_blocked(Q, K, scale)
            M_fro = compute_M_fro_norm_blocked(Q, K, d_k_mf, scale).item()
            cn = estimate_commutator_norm_matrix_free(Q, K, d_k_mf, scale, M_fro)
            assert cn >= 0.0

    def test_matvec_correctness(self):
        Q, K, scale, A, M, d_k_inv_sqrt = make_M(10, 4)
        M_sym = (M + M.T) / 2.0
        M_asym = (M - M.T) / 2.0
        comm = M_sym @ M_asym - M_asym @ M_sym
        _, d_k_mf = compute_dk_blocked(Q, K, scale)
        torch.manual_seed(0)
        v = torch.randn(10)
        ref = comm @ v
        mf = matvec_commutator_blocked(Q, K, v, d_k_mf, scale, block_size=4)
        assert torch.allclose(ref, mf, atol=1e-4), f"max diff={torch.max(torch.abs(ref - mf))}"


# ===========================================================================
# Group 9: Matvec Helpers — Algebraic Correctness
# ===========================================================================


class TestMatvecHelpers:
    def test_Masym_matches_materialized(self):
        Q, K, scale, A, M, d_k_inv_sqrt = make_M(10, 4)
        _, d_k_mf = compute_dk_blocked(Q, K, scale)
        M_asym = (M - M.T) / 2.0
        torch.manual_seed(0)
        v = torch.randn(10)
        ref = M_asym @ v
        mf = matvec_Masym_blocked(Q, K, v, d_k_mf, scale, block_size=4)
        rel = torch.linalg.norm(ref - mf) / torch.linalg.norm(ref).clamp(min=1e-8)
        assert rel < 1e-4

    def test_Msym_matches_materialized(self):
        Q, K, scale, A, M, d_k_inv_sqrt = make_M(10, 4)
        _, d_k_mf = compute_dk_blocked(Q, K, scale)
        M_sym = (M + M.T) / 2.0
        torch.manual_seed(0)
        v = torch.randn(10)
        ref = M_sym @ v
        mf = matvec_Msym_blocked(Q, K, v, d_k_mf, scale, block_size=4)
        rel = torch.linalg.norm(ref - mf) / torch.linalg.norm(ref).clamp(min=1e-8)
        assert rel < 1e-4

    def test_Masym_antisymmetry(self):
        Q, K, scale, A, M, d_k_inv_sqrt = make_M(10, 4)
        _, d_k_mf = compute_dk_blocked(Q, K, scale)
        torch.manual_seed(0)
        v = torch.randn(10)
        w = torch.randn(10)
        Av = matvec_Masym_blocked(Q, K, v, d_k_mf, scale, block_size=4)
        Aw = matvec_Masym_blocked(Q, K, w, d_k_mf, scale, block_size=4)
        # <Av, w> + <v, Aw> = 0
        assert abs(Av.dot(w).item() + v.dot(Aw).item()) < 1e-4

    def test_Msym_symmetry(self):
        Q, K, scale, A, M, d_k_inv_sqrt = make_M(10, 4)
        _, d_k_mf = compute_dk_blocked(Q, K, scale)
        torch.manual_seed(0)
        v = torch.randn(10)
        w = torch.randn(10)
        Sv = matvec_Msym_blocked(Q, K, v, d_k_mf, scale, block_size=4)
        Sw = matvec_Msym_blocked(Q, K, w, d_k_mf, scale, block_size=4)
        # <Sv, w> = <v, Sw>
        assert abs(Sv.dot(w).item() - v.dot(Sw).item()) < 1e-4

    def test_decomposition_M_eq_Msym_plus_Masym(self):
        Q, K, scale, A, M, d_k_inv_sqrt = make_M(10, 4)
        _, d_k_mf = compute_dk_blocked(Q, K, scale)
        torch.manual_seed(0)
        v = torch.randn(10)
        Mv = matvec_M_blocked(Q, K, v, d_k_mf, scale, block_size=4)
        Sv = matvec_Msym_blocked(Q, K, v, d_k_mf, scale, block_size=4)
        Av = matvec_Masym_blocked(Q, K, v, d_k_mf, scale, block_size=4)
        assert torch.allclose(Mv, Sv + Av, atol=1e-6)


# ===========================================================================
# Group 10: Full Integration — compute_routing_features
# ===========================================================================


class TestRoutingFeatures:
    def test_returns_typed_features(self):
        from glassbox.results import RoutingFeatures

        Q, K, scale, A, M, d_k_inv_sqrt = make_M(16, 4)
        _, d_k_mf = compute_dk_blocked(Q, K, scale)
        f = compute_routing_features_matrix_free(Q, K, d_k_mf, scale, rank=4)
        assert isinstance(f, RoutingFeatures)
        assert len(f.singular_values) > 0
        # All hodge fields populated
        assert f.phi_hat is not None
        assert f.sigma2 is not None
        assert f.asym_index is not None
        # Spectral fields populated
        assert f.sv1 is not None
        assert f.sv_ratio is not None

    def test_value_ranges(self):
        Q, K, scale, A, M, d_k_inv_sqrt = make_M(16, 4)
        _, d_k_mf = compute_dk_blocked(Q, K, scale)
        f = compute_routing_features_matrix_free(Q, K, d_k_mf, scale, rank=4)
        assert 0.0 <= f.sigma2 <= 1.0
        assert 0.0 <= f.phi_hat <= 1.0
        assert f.asym_index >= 0.0
        assert len(f.singular_values) > 0

    def test_singular_values_descending(self):
        Q, K, scale, A, M, d_k_inv_sqrt = make_M(16, 4)
        _, d_k_mf = compute_dk_blocked(Q, K, scale)
        f = compute_routing_features_matrix_free(Q, K, d_k_mf, scale, rank=4)
        for i in range(len(f.singular_values) - 1):
            assert f.singular_values[i] >= f.singular_values[i + 1] - 1e-6


# ===========================================================================
# Group 12: Edge Cases and Robustness
# ===========================================================================


class TestEdgeCases:
    def test_very_small_sequence_L3(self):
        """L=3 should not crash. Only 1 triangle."""
        Q, K, scale, A, M, d_k_inv_sqrt = make_M(3, 4)
        _, d_k_mf = compute_dk_blocked(Q, K, scale)
        f = compute_routing_features_matrix_free(Q, K, d_k_mf, scale, rank=2)
        assert f.asym_index >= 0.0
        assert math.isfinite(f.asym_index)

    def test_numerical_stability_float32(self):
        Q, K, scale, A, M, d_k_inv_sqrt = make_M(16, 4)
        Q, K = Q.float(), K.float()
        _, d_k_mf = compute_dk_blocked(Q, K, scale)
        f = compute_routing_features_matrix_free(Q, K, d_k_mf, scale, rank=2)
        for key, val in f.model_dump().items():
            if isinstance(val, float):
                assert math.isfinite(val), f"{key} is not finite: {val}"


# ===========================================================================
# Group 13: Materialized Path
# ===========================================================================


class TestMaterializedPath:
    def test_returns_typed_features(self):
        from glassbox.results import RoutingFeatures

        Q, K, scale, A, M, d_k_inv_sqrt = make_M(16, 4)
        f = compute_routing_features_materialized(M, rank=4)
        assert isinstance(f, RoutingFeatures)
        assert len(f.singular_values) > 0
        assert f.phi_hat is not None
        assert f.asym_index is not None

    def test_value_ranges(self):
        Q, K, scale, A, M, d_k_inv_sqrt = make_M(16, 4)
        f = compute_routing_features_materialized(M, rank=4)
        assert 0.0 <= f.sigma2 <= 1.0
        assert 0.0 <= f.phi_hat <= 1.0
        assert f.asym_index >= 0.0

    def test_symmetric_near_zero(self):
        torch.manual_seed(77)
        X = torch.randn(12, 12)
        M = X @ X.T
        M = M / M.sum(dim=1, keepdim=True)
        M = (M + M.T) / 2.0
        f = compute_routing_features_materialized(M, rank=4)
        assert f.asym_index < 0.01

    def test_singular_values_match_torch(self):
        Q, K, scale, A, M, d_k_inv_sqrt = make_M(16, 4)
        f = compute_routing_features_materialized(M, rank=4)
        sigma_ref = torch.linalg.svdvals(M)[:4].tolist()
        for a, b in zip(f.singular_values, sigma_ref):
            assert abs(a - b) < 1e-5


# ===========================================================================
# Group 14: Cross-Validation — Materialized vs Matrix-Free
# ===========================================================================


class TestMaterializedVsMatrixFree:
    def test_G_agreement(self):
        Q, K, scale, A, M, d_k_inv_sqrt = make_M(16, 4)
        _, d_k_mf = compute_dk_blocked(Q, K, scale)
        f_mat = compute_routing_features_materialized(M, rank=4)
        f_mf = compute_routing_features_matrix_free(
            Q,
            K,
            d_k_mf,
            scale,
            rank=4,
        )
        assert abs(f_mat.asym_index - f_mf.asym_index) < 0.02

    def test_sigma2_agreement(self):
        Q, K, scale, A, M, d_k_inv_sqrt = make_M(16, 4)
        _, d_k_mf = compute_dk_blocked(Q, K, scale)
        f_mat = compute_routing_features_materialized(M, rank=4)
        f_mf = compute_routing_features_matrix_free(
            Q,
            K,
            d_k_mf,
            scale,
            rank=4,
        )
        assert abs(f_mat.sigma2 - f_mf.sigma2) < 0.05

    def test_all_features_close(self):
        """All routing features should agree between materialized and matrix-free."""
        for seed in range(5):
            Q, K, scale, A, M, d_k_inv_sqrt = make_M(12, 4, seed=seed)
            _, d_k_mf = compute_dk_blocked(Q, K, scale)
            f_mat = compute_routing_features_materialized(M, rank=4)
            f_mf = compute_routing_features_matrix_free(
                Q,
                K,
                d_k_mf,
                scale,
                rank=4,
            )
            for key in ["asym_index"]:
                assert abs(getattr(f_mat, key) - getattr(f_mf, key)) < 0.05, (
                    f"seed={seed}, {key}: mat={getattr(f_mat, key)}, mf={getattr(f_mf, key)}"
                )
            assert abs(f_mat.sigma2 - f_mf.sigma2) < 0.1


# ===========================================================================
# Group 15: Half-precision dtype propagation (fp16 / bf16)
# ===========================================================================

HALF_DTYPES = [torch.float16, torch.bfloat16]
DTYPE_IDS = {torch.float16: "fp16", torch.bfloat16: "bf16"}


def _make_M_half(L, D, dtype, seed=42):
    """Generate half-precision Q, K and derived quantities."""
    torch.manual_seed(seed)
    Q = torch.randn(L, D).to(dtype)
    K = torch.randn(L, D).to(dtype)
    scale = 1.0 / math.sqrt(D)
    d_k, d_k_mf = compute_dk_blocked(Q, K, scale)
    return Q, K, scale, d_k_mf


class TestHalfPrecisionDtype:
    @pytest.mark.parametrize("dtype", HALF_DTYPES, ids=lambda d: DTYPE_IDS[d])
    def test_routing_features_matrix_free_half(self, dtype):
        """compute_routing_features_matrix_free should not crash with half Q/K."""
        Q, K, scale, d_k_mf = _make_M_half(16, 4, dtype)
        f = compute_routing_features_matrix_free(
            Q,
            K,
            d_k_mf,
            scale,
            rank=2,
        )
        assert f.sigma2 is not None
        assert f.asym_index is not None
        assert len(f.singular_values) == 2
        assert all(sv > 0 for sv in f.singular_values)

    @pytest.mark.parametrize("dtype", HALF_DTYPES, ids=lambda d: DTYPE_IDS[d])
    def test_sigma2_asym_half(self, dtype):
        """compute_sigma2_asym_matrix_free should not crash with half Q/K."""
        Q, K, scale, d_k_mf = _make_M_half(16, 4, dtype)
        result = compute_sigma2_asym_matrix_free(Q, K, d_k_mf, scale, block_size=256)
        assert isinstance(result, float)
        assert math.isfinite(result)

    @pytest.mark.parametrize("dtype", HALF_DTYPES, ids=lambda d: DTYPE_IDS[d])
    def test_G_matrix_free_half(self, dtype):
        """compute_G_matrix_free should not crash with half Q/K."""
        Q, K, scale, d_k_mf = _make_M_half(16, 4, dtype)
        G, fro = compute_G_matrix_free(Q, K, d_k_mf, scale)
        assert math.isfinite(G)
        assert fro > 0

    @pytest.mark.parametrize("dtype", HALF_DTYPES, ids=lambda d: DTYPE_IDS[d])
    def test_commutator_norm_half(self, dtype):
        """estimate_commutator_norm_matrix_free should not crash with half Q/K."""
        Q, K, scale, d_k_mf = _make_M_half(16, 4, dtype)
        fro = compute_M_fro_norm_blocked(Q, K, d_k_mf, scale)
        cn = estimate_commutator_norm_matrix_free(
            Q,
            K,
            d_k_mf,
            scale,
            fro.item(),
            n_hutchinson=5,
            seed=42,
        )
        assert math.isfinite(cn)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_routing_materialized_half_precision_no_crash(dtype):
    """fp16/bf16 M must not crash the dense SVD / svdvals path (#57)."""
    torch.manual_seed(0)
    M = torch.softmax(torch.randn(16, 16), dim=-1).to(dtype)
    feats = compute_routing_features_materialized(M, rank=3)
    for v in (feats.sigma2, feats.asym_index, feats.phi_hat):
        assert v is not None and math.isfinite(v)
