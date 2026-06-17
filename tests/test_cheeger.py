"""Comprehensive test suite for bipartite sweep conductance (Cheeger).

Organized into 4 groups:
  1. Bipartite sweep conductance — bounds, invariants, mathematical properties
  2. Edge cases — small L, degenerate matrices, numerical stability
  3. Matrix-free sweep — agreement with materialized path
  4. Witness — cut membership properties
"""

import math

import pytest
import torch

from glassbox.cheeger import (
    SweepResult,
    bipartite_sweep_conductance,
    bipartite_sweep_conductance_matrix_free,
    compute_cheeger_features_materialized,
    compute_cheeger_features_matrix_free,
    compute_cheeger_witness_materialized,
    compute_cheeger_witness_matrix_free,
)
from glassbox.svd import compute_degree_normalized_M, compute_dk_blocked


def _make_M(L, D, seed=42):
    """Generate Q, K, scale, A, M and related quantities."""
    torch.manual_seed(seed)
    Q = torch.randn(L, D)
    K = torch.randn(L, D)
    scale = 1.0 / math.sqrt(D)
    A = torch.softmax(Q @ K.T * scale, dim=-1)
    M, _, d_k_inv_sqrt = compute_degree_normalized_M(A)
    return Q, K, scale, A, M, d_k_inv_sqrt


def _make_sweep_inputs(L, D, seed=42):
    """Build M, run full SVD, return (u2, v2, M, sigma2)."""
    _, _, _, _, M, _ = _make_M(L, D, seed=seed)
    U, sigma, Vt = torch.linalg.svd(M, full_matrices=False)
    sigma2 = sigma[1].item() if len(sigma) > 1 else 0.0
    u2 = U[:, 1] if sigma.shape[0] > 1 else torch.zeros(L)
    v2 = Vt[1, :] if sigma.shape[0] > 1 else torch.zeros(L)
    return u2, v2, M, sigma2


# ===========================================================================
# Group 1: Bipartite Sweep Conductance — Properties & Bounds
# ===========================================================================


class TestBipartiteSweepConductance:
    def test_non_negative(self):
        for seed in range(5):
            u2, v2, M, _ = _make_sweep_inputs(16, 4, seed=seed)
            result = bipartite_sweep_conductance(u2, v2, M)
            assert result.phi_star >= 0.0, f"seed={seed}: phi={result.phi_star} < 0"

    def test_upper_bounded_by_one(self):
        for seed in range(5):
            u2, v2, M, _ = _make_sweep_inputs(16, 4, seed=seed)
            result = bipartite_sweep_conductance(u2, v2, M)
            assert result.phi_star <= 1.0, f"seed={seed}: phi={result.phi_star} > 1"

    def test_cheeger_upper_bound(self):
        """φ* ≤ √(2(1 - σ₂)) — the sweep cut guarantee."""
        for seed in range(10):
            u2, v2, M, sigma2 = _make_sweep_inputs(16, 4, seed=seed)
            result = bipartite_sweep_conductance(u2, v2, M)
            gap = 1.0 - sigma2
            upper = math.sqrt(max(2.0 * gap, 0.0))
            assert result.phi_star <= upper + 1e-6, (
                f"seed={seed}: phi={result.phi_star:.6f} > upper={upper:.6f}"
            )

    def test_cheeger_lower_bound(self):
        """φ* ≥ (1 - σ₂) / 2 — Cheeger lower bound."""
        for seed in range(10):
            u2, v2, M, sigma2 = _make_sweep_inputs(16, 4, seed=seed)
            result = bipartite_sweep_conductance(u2, v2, M)
            gap = 1.0 - sigma2
            lower = gap / 2.0
            assert result.phi_star >= lower - 1e-6, (
                f"seed={seed}: phi={result.phi_star:.6f} < lower={lower:.6f}"
            )

    def test_sign_invariance(self):
        """φ(u2, v2, M) == φ(-u2, -v2, M) — sign ambiguity must not matter."""
        for seed in range(5):
            u2, v2, M, _ = _make_sweep_inputs(16, 4, seed=seed)
            r_pos = bipartite_sweep_conductance(u2, v2, M)
            r_neg = bipartite_sweep_conductance(-u2, -v2, M)
            assert abs(r_pos.phi_star - r_neg.phi_star) < 1e-10, (
                f"seed={seed}: phi_pos={r_pos.phi_star:.10f} != phi_neg={r_neg.phi_star:.10f}"
            )

    def test_not_equal_to_spectral_gap(self):
        """φ* ≠ 1 - σ₂ in general — this is the whole point."""
        differences_found = 0
        for seed in range(10):
            u2, v2, M, sigma2 = _make_sweep_inputs(16, 4, seed=seed)
            result = bipartite_sweep_conductance(u2, v2, M)
            spectral_gap = 1.0 - sigma2
            if abs(result.phi_star - spectral_gap) > 0.01:
                differences_found += 1
        assert differences_found > 0

    def test_permutation_invariance(self):
        """Permuting rows/cols of M with matching vector reorder gives same φ*."""
        u2, v2, M, _ = _make_sweep_inputs(16, 4, seed=42)
        r_orig = bipartite_sweep_conductance(u2, v2, M)

        torch.manual_seed(99)
        perm = torch.randperm(16)
        M_perm = M[perm][:, perm]
        u2_perm = u2[perm]
        v2_perm = v2[perm]
        r_perm = bipartite_sweep_conductance(u2_perm, v2_perm, M_perm)

        assert abs(r_orig.phi_star - r_perm.phi_star) < 1e-6

    def test_returns_sweep_result(self):
        """bipartite_sweep_conductance returns a SweepResult."""
        u2, v2, M, _ = _make_sweep_inputs(16, 4, seed=42)
        result = bipartite_sweep_conductance(u2, v2, M)
        assert isinstance(result, SweepResult)
        assert isinstance(result.phi_star, float)
        assert result.cut_membership.shape == (32,)  # 2L


# ===========================================================================
# Group 2: Edge Cases
# ===========================================================================


class TestSweepEdgeCases:
    def test_small_L2(self):
        u2, v2, M, _ = _make_sweep_inputs(2, 4, seed=42)
        result = bipartite_sweep_conductance(u2, v2, M)
        assert 0.0 <= result.phi_star <= 1.0

    def test_small_L3(self):
        u2, v2, M, _ = _make_sweep_inputs(3, 4, seed=42)
        result = bipartite_sweep_conductance(u2, v2, M)
        assert 0.0 <= result.phi_star <= 1.0

    def test_symmetric_M(self):
        """M = M^T (doubly stochastic) should still work."""
        torch.manual_seed(42)
        M = torch.rand(8, 8)
        M = (M + M.T) / 2.0
        M = M / M.sum(dim=1, keepdim=True)
        U, sigma, Vt = torch.linalg.svd(M, full_matrices=False)
        u2 = U[:, 1]
        v2 = Vt[1, :]
        result = bipartite_sweep_conductance(u2, v2, M)
        assert 0.0 <= result.phi_star <= 1.0

    def test_near_rank1_M(self):
        """Nearly rank-1 M (σ₂ ≈ 0) should give high conductance."""
        L = 8
        torch.manual_seed(42)
        a = torch.softmax(torch.randn(L), dim=0)
        b = torch.softmax(torch.randn(L), dim=0)
        M = torch.outer(a, b) + 1e-6 * torch.eye(L)
        U, sigma, Vt = torch.linalg.svd(M, full_matrices=False)
        assert sigma[1].item() < 0.01
        u2 = U[:, 1]
        v2 = Vt[1, :]
        result = bipartite_sweep_conductance(u2, v2, M)
        assert result.phi_star > 0.3

    def test_float32_stability(self):
        u2, v2, M, _ = _make_sweep_inputs(16, 4, seed=42)
        M32 = M.float()
        u2_32 = u2.float()
        v2_32 = v2.float()
        result = bipartite_sweep_conductance(u2_32, v2_32, M32)
        assert math.isfinite(result.phi_star)
        assert 0.0 <= result.phi_star <= 1.0

    def test_empty_graph(self):
        M = torch.zeros(8, 8)
        u2 = torch.randn(8)
        v2 = torch.randn(8)
        result = bipartite_sweep_conductance(u2, v2, M)
        assert result.phi_star == 0.0


# ===========================================================================
# Group 3: Matrix-Free Sweep — Agreement with Materialized
# ===========================================================================


class TestMatrixFreeSweep:
    def test_matches_materialized(self):
        """Blocked matrix-free sweep should match dense materialized sweep."""
        Q, K, scale, A, M, d_k_inv_sqrt = _make_M(16, 4, seed=42)
        U, sigma, Vt = torch.linalg.svd(M, full_matrices=False)
        u2 = U[:, 1]
        v2 = Vt[1, :]

        r_mat = bipartite_sweep_conductance(u2, v2, M)
        r_mf = bipartite_sweep_conductance_matrix_free(
            u2, v2, Q, K, d_k_inv_sqrt, scale, block_size=8
        )

        assert abs(r_mat.phi_star - r_mf.phi_star) < 0.05, (
            f"Materialized={r_mat.phi_star:.6f} vs matrix-free={r_mf.phi_star:.6f}"
        )

    def test_matches_materialized_causal(self):
        """Same with causal masking."""
        L, D = 16, 4
        torch.manual_seed(42)
        Q = torch.randn(L, D)
        K = torch.randn(L, D)
        scale = 1.0 / math.sqrt(D)

        scores = Q @ K.T * scale
        causal_mask = torch.tril(torch.ones(L, L, dtype=torch.bool))
        scores = scores.masked_fill(~causal_mask, float("-inf"))
        A = torch.softmax(scores, dim=-1)
        M, _, d_k_inv_sqrt = compute_degree_normalized_M(A)

        U, sigma, Vt = torch.linalg.svd(M, full_matrices=False)
        u2 = U[:, 1]
        v2 = Vt[1, :]

        r_mat = bipartite_sweep_conductance(u2, v2, M)
        r_mf = bipartite_sweep_conductance_matrix_free(
            u2, v2, Q, K, d_k_inv_sqrt, scale, block_size=8, causal=True
        )

        assert abs(r_mat.phi_star - r_mf.phi_star) < 0.05, (
            f"Causal: materialized={r_mat.phi_star:.6f} vs matrix-free={r_mf.phi_star:.6f}"
        )

    def test_multiple_sizes(self):
        """Verify agreement across L=8, 16, 32."""
        for L in [8, 16, 32]:
            Q, K, scale, A, M, d_k_inv_sqrt = _make_M(L, 4, seed=42)
            U, sigma, Vt = torch.linalg.svd(M, full_matrices=False)
            u2 = U[:, 1]
            v2 = Vt[1, :]

            r_mat = bipartite_sweep_conductance(u2, v2, M)
            r_mf = bipartite_sweep_conductance_matrix_free(
                u2, v2, Q, K, d_k_inv_sqrt, scale, block_size=8
            )

            assert abs(r_mat.phi_star - r_mf.phi_star) < 0.05, (
                f"L={L}: mat={r_mat.phi_star:.6f} vs mf={r_mf.phi_star:.6f}"
            )


# ===========================================================================
# Group 4: Witness — Cut Membership Properties
# ===========================================================================


class TestWitness:
    def test_cut_membership_binary(self):
        """cut_membership should be in {0, 1}."""
        u2, v2, M, _ = _make_sweep_inputs(16, 4, seed=42)
        result = bipartite_sweep_conductance(u2, v2, M)
        assert set(result.cut_membership.tolist()).issubset({0, 1})

    def test_cut_membership_both_sides(self):
        """Both sides of the cut should be populated."""
        u2, v2, M, _ = _make_sweep_inputs(16, 4, seed=42)
        result = bipartite_sweep_conductance(u2, v2, M)
        assert result.cut_membership.sum() > 0
        assert result.cut_membership.sum() < len(result.cut_membership)

    def test_witness_materialized_shape(self):
        """compute_cheeger_witness_materialized returns [nL] tensor in {-1, +1}."""
        _, _, _, _, M, _ = _make_M(16, 4, seed=42)
        w = compute_cheeger_witness_materialized(M, rank=2)
        assert w.shape == (16,)
        assert set(w.tolist()).issubset({-1, 1})

    def test_witness_matrix_free_shape(self):
        """compute_cheeger_witness_matrix_free returns [L] tensor in {-1, +1}."""
        Q, K, scale, _, _, d_k_inv_sqrt = _make_M(16, 4, seed=42)
        _, d_k_mf = compute_dk_blocked(Q, K, scale)
        w = compute_cheeger_witness_matrix_free(Q, K, d_k_mf, scale, rank=2, block_size=8)
        assert w.shape == (16,)
        assert set(w.tolist()).issubset({-1, 1})

    def test_witness_has_both_sides(self):
        """Witness should have tokens on both sides of the cut."""
        _, _, _, _, M, _ = _make_M(16, 4, seed=42)
        w = compute_cheeger_witness_materialized(M, rank=2)
        assert (w == 1).any()
        assert (w == -1).any()


# ===========================================================================
# Group 5: High-level Feature Entry Points
# ===========================================================================


class TestCheegerFeatures:
    def test_materialized_features(self):
        _, _, _, _, M, _ = _make_M(16, 4, seed=42)
        f = compute_cheeger_features_materialized(M, rank=2)
        assert 0.0 <= f.phi_star <= 1.0
        assert f.sigma2 is not None
        assert f.cheeger_lower is not None
        assert f.cheeger_upper is not None
        assert f.cheeger_lower <= f.phi_star + 1e-6
        assert f.phi_star <= f.cheeger_upper + 1e-6

    def test_matrix_free_features(self):
        Q, K, scale, _, _, d_k_inv_sqrt = _make_M(16, 4, seed=42)
        _, d_k_mf = compute_dk_blocked(Q, K, scale)
        f = compute_cheeger_features_matrix_free(Q, K, d_k_mf, scale, rank=2, block_size=8)
        assert 0.0 <= f.phi_star <= 1.0
        assert f.sigma2 is not None
        assert f.cheeger_lower is not None
        assert f.cheeger_upper is not None

    def test_features_agreement(self):
        """Materialized and matrix-free features should roughly agree."""
        Q, K, scale, _, M, d_k_inv_sqrt = _make_M(16, 4, seed=42)
        _, d_k_mf = compute_dk_blocked(Q, K, scale)

        f_mat = compute_cheeger_features_materialized(M, rank=2)
        f_mf = compute_cheeger_features_matrix_free(Q, K, d_k_mf, scale, rank=2, block_size=8)

        assert abs(f_mat.phi_star - f_mf.phi_star) < 0.1
        assert abs(f_mat.sigma2 - f_mf.sigma2) < 0.1
