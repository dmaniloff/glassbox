"""Comprehensive test suite for matrix-free Hodge decomposition.

Organized into 12 groups that together establish mathematical faithfulness
of the matrix-free implementation against materialized references and
Hodge-theoretic identities.
"""

import math
from itertools import combinations

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


def _build_B1(n):
    """Build incidence matrix B1 for complete graph K_n."""
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    m = len(edges)
    B1 = torch.zeros(n, m, dtype=torch.float64)
    for e_idx, (i, j) in enumerate(edges):
        B1[i, e_idx] = -1
        B1[j, e_idx] = +1
    return B1, edges


def _build_B2(n, edges):
    """Build boundary matrix B2 for complete graph K_n."""
    edge_to_idx = {e: i for i, e in enumerate(edges)}
    triangles = list(combinations(range(n), 3))
    m = len(edges)
    t = len(triangles)
    B2 = torch.zeros(m, t, dtype=torch.float64)
    for tri_idx, (i, j, k) in enumerate(triangles):
        B2[edge_to_idx[(i, j)], tri_idx] = +1
        B2[edge_to_idx[(j, k)], tri_idx] = +1
        B2[edge_to_idx[(i, k)], tri_idx] = -1
    return B2


def _hodge_decompose(f, B1, B2):
    """Inline exact Hodge decomposition (for test cross-validation)."""
    rcond = 1e-10
    L0 = B1 @ B1.T
    L0_pinv = torch.linalg.pinv(L0, rcond=rcond)
    phi = L0_pinv @ B1 @ f
    f_grad = B1.T @ phi
    if B2.shape[1] > 0:
        L2 = B2.T @ B2
        L2_pinv = torch.linalg.pinv(L2, rcond=rcond)
        psi = L2_pinv @ B2.T @ f
        f_curl = B2 @ psi
    else:
        f_curl = torch.zeros_like(f)
    return f_grad, f_curl


def _matrix_to_edge_flow(M, edges):
    """Extract edge flow from matrix."""
    M.shape[0]
    f = torch.zeros(len(edges), dtype=M.dtype)
    for e_idx, (i, j) in enumerate(edges):
        f[e_idx] = M[i, j] - M[j, i]
    return f


# ===========================================================================
# Curl coefficient C — exact Hodge gradient/curl split (row-sum identity, #55)
# ===========================================================================


class TestCurl:
    """Exact gradient/curl split of M_asym via the row-sum identity (replaces triangle-RMS)."""

    def _hodge_oracle(self, M):
        # independent split: potential phi = (M_asym @ 1)/n, A_grad[i,j] = phi_i - phi_j
        A = (M - M.T) / 2.0
        n = M.shape[0]
        phi = A.sum(dim=1) / n
        A_grad = phi[:, None] - phi[None, :]
        fro = torch.linalg.norm(M, "fro")
        return (
            (A.norm() / fro).item(),
            (A_grad.norm() / fro).item(),
            ((A - A_grad).norm() / fro).item(),
        )

    def test_materialized_matches_hodge_oracle(self):
        _, _, _, _, M, _ = make_M(16, 4)
        f = compute_routing_features_materialized(M, rank=4)
        g, gamma, c = self._hodge_oracle(M)
        assert abs(f.G - g) < 1e-6
        assert abs(f.Gamma - gamma) < 1e-6
        assert abs(f.C - c) < 1e-6
        assert abs(f.G**2 - (f.Gamma**2 + f.C**2)) < 1e-9  # genuine split, not a tautology

    def test_matrix_free_matches_materialized(self):
        Q, K, scale, A, M, d_k_inv_sqrt = make_M(16, 4)
        _, d_k_mf = compute_dk_blocked(Q, K, scale)
        f_mat = compute_routing_features_materialized(M, rank=4)
        f_mf = compute_routing_features_matrix_free(Q, K, d_k_mf, scale, rank=4, block_size=8)
        assert abs(f_mat.C - f_mf.C) < 1e-4
        assert abs(f_mat.Gamma - f_mf.Gamma) < 1e-4

    def test_pure_gradient_has_zero_curl(self):
        # M_asym built as a pure gradient field (phi_i - phi_j) -> curl C = 0, Gamma = G.
        # float64: the exact-zero curl is the difference of two ~equal energies, so it needs
        # the precision (in float32 the residual is ~3e-4).
        torch.manual_seed(0)
        n = 12
        phi = torch.randn(n, dtype=torch.float64)
        A_grad = phi[:, None] - phi[None, :]  # antisymmetric, pure gradient (no curl)
        M = torch.eye(n, dtype=torch.float64) + A_grad  # symmetric part I + pure-gradient asym
        f = compute_routing_features_materialized(M, rank=4)
        assert f.C < 1e-6  # no circulatory component
        assert abs(f.Gamma - f.G) < 1e-6

    def test_scales_with_antisymmetry(self):
        _, _, _, _, M, _ = make_M(12, 4, seed=77)
        M_sym = (M + M.T) / 2.0
        M_asym = (M - M.T) / 2.0
        curls = [
            compute_routing_features_materialized(M_sym + a * M_asym, rank=4).C
            for a in [0.0, 0.5, 1.0, 2.0]
        ]
        for i in range(len(curls) - 1):
            assert curls[i] <= curls[i + 1] + 1e-6  # monotone (now exact)


# ===========================================================================
# Group 4: Asymmetry Coefficient G
# ===========================================================================


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
# Group 5: Pythagorean Identity G² = Γ² + C²
# ===========================================================================


class TestPythagorean:
    def test_basic(self):
        Q, K, scale, A, M, d_k_inv_sqrt = make_M(20, 4, seed=99)
        _, d_k_mf = compute_dk_blocked(Q, K, scale)
        f = compute_routing_features_matrix_free(Q, K, d_k_mf, scale, rank=4)
        residual = abs(f.G**2 - f.Gamma**2 - f.C**2)
        assert residual < 0.01, f"Pythagorean: G={f.G}, Γ={f.Gamma}, C={f.C}, residual={residual}"

    def test_multiple_seeds(self):
        for seed in range(10):
            for L in [8, 12, 16, 20]:
                Q, K, scale, A, M, d_k_inv_sqrt = make_M(L, 4, seed=seed)
                _, d_k_mf = compute_dk_blocked(Q, K, scale)
                f = compute_routing_features_matrix_free(Q, K, d_k_mf, scale, rank=2)
                residual = abs(f.G**2 - f.Gamma**2 - f.C**2)
                assert residual < 0.02, f"seed={seed}, L={L}: residual={residual}"

    def test_gamma_nonneg(self):
        for seed in range(5):
            Q, K, scale, A, M, d_k_inv_sqrt = make_M(16, 4, seed=seed)
            _, d_k_mf = compute_dk_blocked(Q, K, scale)
            f = compute_routing_features_matrix_free(Q, K, d_k_mf, scale, rank=2)
            assert f.Gamma >= 0.0

    def test_curl_bounded_by_G(self):
        for seed in range(5):
            Q, K, scale, A, M, d_k_inv_sqrt = make_M(16, 4, seed=seed)
            _, d_k_mf = compute_dk_blocked(Q, K, scale)
            f = compute_routing_features_matrix_free(Q, K, d_k_mf, scale, rank=2)
            assert f.C <= f.G + 0.01, f"C={f.C} > G={f.G}"

    def test_exact_at_exhaustive_n(self):
        # n=5: C(5,3)=10 < floor=200, all triangles enumerated
        Q, K, scale, A, M, d_k_inv_sqrt = make_M(5, 4, seed=42)
        _, d_k_mf = compute_dk_blocked(Q, K, scale)
        f = compute_routing_features_matrix_free(Q, K, d_k_mf, scale, rank=2)
        residual = abs(f.G**2 - f.Gamma**2 - f.C**2)
        assert residual < 1e-4, f"Exact case residual={residual}"


# ===========================================================================
# Group 6: Spectral Features — σ₂(M) and φ̂
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
        assert f.G is not None
        assert f.C is not None
        assert f.Gamma is not None
        assert f.sigma2_asym is not None
        assert f.commutator_norm is not None
        # Spectral fields populated
        assert f.sv1 is not None
        assert f.sv_ratio is not None

    def test_value_ranges(self):
        Q, K, scale, A, M, d_k_inv_sqrt = make_M(16, 4)
        _, d_k_mf = compute_dk_blocked(Q, K, scale)
        f = compute_routing_features_matrix_free(Q, K, d_k_mf, scale, rank=4)
        assert 0.0 <= f.sigma2 <= 1.0
        assert 0.0 <= f.phi_hat <= 1.0
        assert f.G >= 0.0
        assert f.C >= 0.0
        assert f.Gamma >= 0.0
        assert f.sigma2_asym >= 0.0
        assert f.commutator_norm >= 0.0
        assert len(f.singular_values) > 0

    def test_singular_values_descending(self):
        Q, K, scale, A, M, d_k_inv_sqrt = make_M(16, 4)
        _, d_k_mf = compute_dk_blocked(Q, K, scale)
        f = compute_routing_features_matrix_free(Q, K, d_k_mf, scale, rank=4)
        for i in range(len(f.singular_values) - 1):
            assert f.singular_values[i] >= f.singular_values[i + 1] - 1e-6


# ===========================================================================
# Group 11: Cross-Validation Against Exact Hodge (Inline)
# ===========================================================================


class TestExactHodgeCrossValidation:
    def _exact_hodge_coefficients(self, M):
        """Compute exact G, C, Gamma via inline Hodge decomposition."""
        n = M.shape[0]
        M_f64 = M.to(torch.float64)
        B1, edges = _build_B1(n)
        B2 = _build_B2(n, edges)
        f = _matrix_to_edge_flow(M_f64, edges)
        f_grad, f_curl = _hodge_decompose(f, B1, B2)

        M_fro = torch.linalg.norm(M_f64, "fro")
        sqrt2 = math.sqrt(2.0)
        G = (torch.linalg.norm(f) / (sqrt2 * M_fro)).item()
        C = (torch.linalg.norm(f_curl) / (sqrt2 * M_fro)).item()
        Gamma = (torch.linalg.norm(f_grad) / (sqrt2 * M_fro)).item()
        return G, C, Gamma, f_grad, f_curl

    def test_cross_validation_n5(self):
        """For n=5, verify G agrees exactly and both C estimates are nonzero.

        The RMS-based curl estimator C_rms = RMS(circulations) / (sqrt(2) * ||M||_F)
        and the exact Hodge C_exact = ||f_curl|| / (sqrt(2) * ||M||_F) are related
        but not identical: C_rms computes the RMS of triangle circulations, while
        C_exact computes the norm of the curl projection.  They agree in sign and
        ordering but differ in magnitude.  The Pythagorean identity holds for
        C_rms by construction.
        """
        Q, K, scale, A, M, d_k_inv_sqrt = make_M(5, 4, seed=42)
        _, d_k_mf = compute_dk_blocked(Q, K, scale)
        f = compute_routing_features_matrix_free(Q, K, d_k_mf, scale, rank=2)
        G_exact, C_exact, Gamma_exact, _, _ = self._exact_hodge_coefficients(M)
        # G should agree (both exact)
        assert abs(f.G - G_exact) < 0.02, f"G: mf={f.G}, exact={G_exact}"
        # Both C should be nonzero (correlated)
        assert f.C > 0 and C_exact > 0
        # Pythagorean holds for the RMS-based estimate
        residual = abs(f.G**2 - f.Gamma**2 - f.C**2)
        assert residual < 1e-4

    def test_cross_validation_n8(self):
        """For n=8, the RMS-based C and exact Hodge C are related but not
        identical (RMS of circulations vs norm of curl flow). We verify
        they are correlated: both should be nonzero for asymmetric M, and
        the Pythagorean identity should hold for the RMS-based estimate."""
        Q, K, scale, A, M, d_k_inv_sqrt = make_M(8, 4, seed=77)
        _, d_k_mf = compute_dk_blocked(Q, K, scale)
        f = compute_routing_features_matrix_free(Q, K, d_k_mf, scale, rank=2)
        G_exact, C_exact, Gamma_exact, _, _ = self._exact_hodge_coefficients(M)
        # Both G should agree (exact computation)
        assert abs(f.G - G_exact) < 0.02, f"G: mf={f.G}, exact={G_exact}"
        # Both C should be nonzero (correlated)
        assert f.C > 0 and C_exact > 0
        # Pythagorean should hold for the RMS-based estimate
        residual = abs(f.G**2 - f.Gamma**2 - f.C**2)
        assert residual < 1e-4

    def test_hodge_orthogonality(self):
        """f_grad ⊥ f_curl (Hodge orthogonality)."""
        Q, K, scale, A, M, d_k_inv_sqrt = make_M(5, 4, seed=42)
        _, _, _, f_grad, f_curl = self._exact_hodge_coefficients(M)
        inner = f_grad.dot(f_curl).item()
        assert abs(inner) < 1e-8, f"<f_grad, f_curl> = {inner}"

    def test_hodge_completeness(self):
        """||f_grad||² + ||f_curl||² = ||f||² (no harmonic on K_n)."""
        Q, K, scale, A, M, d_k_inv_sqrt = make_M(5, 4, seed=42)
        n = M.shape[0]
        M_f64 = M.to(torch.float64)
        B1, edges = _build_B1(n)
        B2 = _build_B2(n, edges)
        f = _matrix_to_edge_flow(M_f64, edges)
        f_grad, f_curl = _hodge_decompose(f, B1, B2)

        f_sq = f.dot(f).item()
        grad_sq = f_grad.dot(f_grad).item()
        curl_sq = f_curl.dot(f_curl).item()
        assert abs(f_sq - grad_sq - curl_sq) < 1e-8, (
            f"||f||²={f_sq}, ||f_grad||²={grad_sq}, ||f_curl||²={curl_sq}"
        )


# ===========================================================================
# Group 12: Edge Cases and Robustness
# ===========================================================================


class TestEdgeCases:
    def test_very_small_sequence_L3(self):
        """L=3 should not crash. Only 1 triangle."""
        Q, K, scale, A, M, d_k_inv_sqrt = make_M(3, 4)
        _, d_k_mf = compute_dk_blocked(Q, K, scale)
        f = compute_routing_features_matrix_free(Q, K, d_k_mf, scale, rank=2)
        assert f.G >= 0.0
        assert math.isfinite(f.C)

    def test_numerical_stability_float32(self):
        Q, K, scale, A, M, d_k_inv_sqrt = make_M(16, 4)
        Q, K = Q.float(), K.float()
        _, d_k_mf = compute_dk_blocked(Q, K, scale)
        f = compute_routing_features_matrix_free(Q, K, d_k_mf, scale, rank=2)
        for key, val in f.model_dump().items():
            if isinstance(val, float):
                assert math.isfinite(val), f"{key} is not finite: {val}"
        residual = abs(f.G**2 - f.Gamma**2 - f.C**2)
        assert residual < 0.02


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
        assert f.G is not None
        assert f.C is not None

    def test_value_ranges(self):
        Q, K, scale, A, M, d_k_inv_sqrt = make_M(16, 4)
        f = compute_routing_features_materialized(M, rank=4)
        assert 0.0 <= f.sigma2 <= 1.0
        assert 0.0 <= f.phi_hat <= 1.0
        assert f.G >= 0.0
        assert f.C >= 0.0
        assert f.Gamma >= 0.0
        assert f.sigma2_asym >= 0.0
        assert f.commutator_norm >= 0.0

    def test_pythagorean(self):
        Q, K, scale, A, M, d_k_inv_sqrt = make_M(20, 4, seed=99)
        f = compute_routing_features_materialized(M, rank=4)
        residual = abs(f.G**2 - f.Gamma**2 - f.C**2)
        assert residual < 0.01

    def test_symmetric_near_zero(self):
        torch.manual_seed(77)
        X = torch.randn(12, 12)
        M = X @ X.T
        M = M / M.sum(dim=1, keepdim=True)
        M = (M + M.T) / 2.0
        f = compute_routing_features_materialized(M, rank=4)
        assert f.G < 0.01
        assert f.C < 0.01

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
        assert abs(f_mat.G - f_mf.G) < 0.02

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

    def test_curl_agreement(self):
        Q, K, scale, A, M, d_k_inv_sqrt = make_M(16, 4)
        _, d_k_mf = compute_dk_blocked(Q, K, scale)
        f_mat = compute_routing_features_materialized(M, rank=4)
        f_mf = compute_routing_features_matrix_free(
            Q,
            K,
            d_k_mf,
            scale,
            rank=4,
            seed=42,
        )
        assert abs(f_mat.C - f_mf.C) < 0.05

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
                seed=42,
            )
            for key in ["G", "C", "Gamma", "curl_ratio"]:
                assert abs(getattr(f_mat, key) - getattr(f_mf, key)) < 0.05, (
                    f"seed={seed}, {key}: mat={getattr(f_mat, key)}, mf={getattr(f_mf, key)}"
                )
            assert abs(f_mat.sigma2 - f_mf.sigma2) < 0.1
            assert abs(f_mat.sigma2_asym - f_mf.sigma2_asym) < 0.1
            assert abs(f_mat.commutator_norm - f_mf.commutator_norm) < 0.15


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
            seed=42,
        )
        assert f.sigma2 is not None
        assert f.G is not None
        assert f.C is not None
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
    for v in (feats.sigma2, feats.G, feats.phi_hat, feats.sigma2_asym):
        assert v is not None and math.isfinite(v)
