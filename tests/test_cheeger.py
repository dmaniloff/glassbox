"""Comprehensive test suite for bipartite sweep conductance (Cheeger).

Organized into 4 groups:
  1. Bipartite sweep conductance — bounds, invariants, mathematical properties
  2. Edge cases — small L, degenerate matrices, numerical stability
  3. Matrix-free sweep — agreement with materialized path
  4. Integration — phi_hat in RoutingFeatures pipeline
"""

import math

import torch

from glassbox.cheeger import (
    bipartite_sweep_conductance,
    bipartite_sweep_conductance_matrix_free,
)
from glassbox.hodge import (
    compute_routing_features_materialized,
    compute_routing_features_matrix_free,
)
from glassbox.svd import (
    compute_degree_normalized_M,
    compute_dk_blocked,
    compute_logsumexp_blocked,
)

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


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
            phi = bipartite_sweep_conductance(u2, v2, M)
            assert phi >= 0.0, f"seed={seed}: phi={phi} < 0"

    def test_upper_bounded_by_one(self):
        for seed in range(5):
            u2, v2, M, _ = _make_sweep_inputs(16, 4, seed=seed)
            phi = bipartite_sweep_conductance(u2, v2, M)
            assert phi <= 1.0, f"seed={seed}: phi={phi} > 1"

    def test_cheeger_upper_bound(self):
        """φ* ≤ √(2(1 - σ₂)) — the sweep cut guarantee."""
        for seed in range(10):
            u2, v2, M, sigma2 = _make_sweep_inputs(16, 4, seed=seed)
            phi = bipartite_sweep_conductance(u2, v2, M)
            gap = 1.0 - sigma2
            upper = math.sqrt(max(2.0 * gap, 0.0))
            assert phi <= upper + 1e-6, (
                f"seed={seed}: phi={phi:.6f} > upper={upper:.6f} (sigma2={sigma2:.6f})"
            )

    def test_cheeger_lower_bound(self):
        """φ* ≥ (1 - σ₂) / 2 — Cheeger lower bound."""
        for seed in range(10):
            u2, v2, M, sigma2 = _make_sweep_inputs(16, 4, seed=seed)
            phi = bipartite_sweep_conductance(u2, v2, M)
            gap = 1.0 - sigma2
            lower = gap / 2.0
            assert phi >= lower - 1e-6, (
                f"seed={seed}: phi={phi:.6f} < lower={lower:.6f} (sigma2={sigma2:.6f})"
            )

    def test_sign_invariance(self):
        """φ(u2, v2, M) == φ(-u2, -v2, M) — sign ambiguity must not matter."""
        for seed in range(5):
            u2, v2, M, _ = _make_sweep_inputs(16, 4, seed=seed)
            phi_pos = bipartite_sweep_conductance(u2, v2, M)
            phi_neg = bipartite_sweep_conductance(-u2, -v2, M)
            assert abs(phi_pos - phi_neg) < 1e-10, (
                f"seed={seed}: phi_pos={phi_pos:.10f} != phi_neg={phi_neg:.10f}"
            )

    def test_not_equal_to_spectral_gap(self):
        """φ* ≠ 1 - σ₂ in general — this is the whole point of the fix."""
        differences_found = 0
        for seed in range(10):
            u2, v2, M, sigma2 = _make_sweep_inputs(16, 4, seed=seed)
            phi = bipartite_sweep_conductance(u2, v2, M)
            spectral_gap = 1.0 - sigma2
            if abs(phi - spectral_gap) > 0.01:
                differences_found += 1
        assert differences_found > 0, (
            "phi_hat should differ from spectral gap (1 - sigma2) for at least some inputs"
        )

    def test_identity_like_matrix(self):
        """Near-identity M should have very low conductance (strongly connected)."""
        L = 16
        M = torch.eye(L) * 0.9 + torch.ones(L, L) * 0.1 / L
        U, sigma, Vt = torch.linalg.svd(M, full_matrices=False)
        u2 = U[:, 1]
        v2 = Vt[1, :]
        phi = bipartite_sweep_conductance(u2, v2, M)
        assert phi < 0.5, f"Near-identity M should have low conductance, got {phi}"

    def test_permutation_invariance(self):
        """Permuting rows/cols of M with matching vector reorder gives same φ*."""
        u2, v2, M, _ = _make_sweep_inputs(16, 4, seed=42)
        phi_orig = bipartite_sweep_conductance(u2, v2, M)

        # Apply random permutation
        torch.manual_seed(99)
        perm = torch.randperm(16)
        M_perm = M[perm][:, perm]
        u2_perm = u2[perm]
        v2_perm = v2[perm]
        phi_perm = bipartite_sweep_conductance(u2_perm, v2_perm, M_perm)

        assert abs(phi_orig - phi_perm) < 1e-6, (
            f"Permutation should not change φ*: {phi_orig:.8f} vs {phi_perm:.8f}"
        )


# ===========================================================================
# Group 2: Edge Cases
# ===========================================================================


class TestSweepEdgeCases:
    def test_small_L2(self):
        """L=2: minimum non-degenerate case."""
        u2, v2, M, _ = _make_sweep_inputs(2, 4, seed=42)
        phi = bipartite_sweep_conductance(u2, v2, M)
        assert 0.0 <= phi <= 1.0

    def test_small_L3(self):
        u2, v2, M, _ = _make_sweep_inputs(3, 4, seed=42)
        phi = bipartite_sweep_conductance(u2, v2, M)
        assert 0.0 <= phi <= 1.0

    def test_symmetric_M(self):
        """M = M^T (doubly stochastic) should still work."""
        torch.manual_seed(42)
        M = torch.rand(8, 8)
        M = (M + M.T) / 2.0
        M = M / M.sum(dim=1, keepdim=True)
        U, sigma, Vt = torch.linalg.svd(M, full_matrices=False)
        u2 = U[:, 1]
        v2 = Vt[1, :]
        phi = bipartite_sweep_conductance(u2, v2, M)
        assert 0.0 <= phi <= 1.0

    def test_near_rank1_M(self):
        """Nearly rank-1 M (σ₂ ≈ 0) should give high conductance."""
        L = 8
        # Rank-1: outer product
        torch.manual_seed(42)
        a = torch.softmax(torch.randn(L), dim=0)
        b = torch.softmax(torch.randn(L), dim=0)
        M = torch.outer(a, b) + 1e-6 * torch.eye(L)
        U, sigma, Vt = torch.linalg.svd(M, full_matrices=False)
        assert sigma[1].item() < 0.01, "Test setup: sigma2 should be near 0"
        u2 = U[:, 1]
        v2 = Vt[1, :]
        phi = bipartite_sweep_conductance(u2, v2, M)
        assert phi > 0.3, f"Near-rank-1 M should have high conductance, got {phi}"

    def test_float32_stability(self):
        """Verify sweep works in float32 without NaN/Inf."""
        u2, v2, M, _ = _make_sweep_inputs(16, 4, seed=42)
        M32 = M.float()
        u2_32 = u2.float()
        v2_32 = v2.float()
        phi = bipartite_sweep_conductance(u2_32, v2_32, M32)
        assert math.isfinite(phi), f"Got non-finite phi: {phi}"
        assert 0.0 <= phi <= 1.0

    def test_empty_graph(self):
        """All-zero M should return 0.0."""
        M = torch.zeros(8, 8)
        u2 = torch.randn(8)
        v2 = torch.randn(8)
        phi = bipartite_sweep_conductance(u2, v2, M)
        assert phi == 0.0


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

        phi_mat = bipartite_sweep_conductance(u2, v2, M)
        phi_mf = bipartite_sweep_conductance_matrix_free(
            u2, v2, Q, K, d_k_inv_sqrt, scale, block_size=8,
        )

        assert abs(phi_mat - phi_mf) < 0.05, (
            f"Materialized={phi_mat:.6f} vs matrix-free={phi_mf:.6f}"
        )

    def test_matches_materialized_causal(self):
        """Same with causal masking."""
        L, D = 16, 4
        torch.manual_seed(42)
        Q = torch.randn(L, D)
        K = torch.randn(L, D)
        scale = 1.0 / math.sqrt(D)

        # Build causal attention
        scores = Q @ K.T * scale
        causal_mask = torch.tril(torch.ones(L, L, dtype=torch.bool))
        scores = scores.masked_fill(~causal_mask, float("-inf"))
        A = torch.softmax(scores, dim=-1)
        M, _, d_k_inv_sqrt = compute_degree_normalized_M(A)

        U, sigma, Vt = torch.linalg.svd(M, full_matrices=False)
        u2 = U[:, 1]
        v2 = Vt[1, :]

        phi_mat = bipartite_sweep_conductance(u2, v2, M)
        phi_mf = bipartite_sweep_conductance_matrix_free(
            u2, v2, Q, K, d_k_inv_sqrt, scale, block_size=8, causal=True,
        )

        assert abs(phi_mat - phi_mf) < 0.05, (
            f"Causal: materialized={phi_mat:.6f} vs matrix-free={phi_mf:.6f}"
        )

    def test_multiple_sizes(self):
        """Verify agreement across L=8, 16, 32."""
        for L in [8, 16, 32]:
            Q, K, scale, A, M, d_k_inv_sqrt = _make_M(L, 4, seed=42)
            U, sigma, Vt = torch.linalg.svd(M, full_matrices=False)
            u2 = U[:, 1]
            v2 = Vt[1, :]

            phi_mat = bipartite_sweep_conductance(u2, v2, M)
            phi_mf = bipartite_sweep_conductance_matrix_free(
                u2, v2, Q, K, d_k_inv_sqrt, scale, block_size=8,
            )

            assert abs(phi_mat - phi_mf) < 0.05, (
                f"L={L}: materialized={phi_mat:.6f} vs matrix-free={phi_mf:.6f}"
            )


# ===========================================================================
# Group 4: Integration — phi_hat in RoutingFeatures Pipeline
# ===========================================================================


class TestPhiHatIntegration:
    def test_materialized_uses_sweep_cut(self):
        """phi_hat from routing features should differ from 1 - sigma2."""
        differences = 0
        for seed in range(10):
            _, _, _, _, M, _ = _make_M(16, 4, seed=seed)
            features = compute_routing_features_materialized(M, rank=4)
            spectral_gap = 1.0 - features.sigma2
            if abs(features.phi_hat - spectral_gap) > 0.01:
                differences += 1
        assert differences > 0, (
            "phi_hat should differ from 1 - sigma2 for at least some inputs"
        )

    def test_matrix_free_uses_sweep_cut(self):
        """phi_hat from matrix-free path should differ from 1 - sigma2."""
        differences = 0
        for seed in range(10):
            Q, K, scale, _, _, d_k_inv_sqrt = _make_M(16, 4, seed=seed)
            _, d_k_mf = compute_dk_blocked(Q, K, scale)
            lse = compute_logsumexp_blocked(Q, K, scale)
            features = compute_routing_features_matrix_free(
                Q, K, d_k_mf, scale, lse, rank=4, min_samples=50,
            )
            spectral_gap = 1.0 - features.sigma2
            if abs(features.phi_hat - spectral_gap) > 0.01:
                differences += 1
        assert differences > 0, (
            "Matrix-free phi_hat should differ from 1 - sigma2 for at least some inputs"
        )

    def test_materialized_vs_matrix_free(self):
        """phi_hat should agree between materialized and matrix-free paths."""
        Q, K, scale, _, M, d_k_inv_sqrt = _make_M(16, 4, seed=42)
        _, d_k_mf = compute_dk_blocked(Q, K, scale)
        lse = compute_logsumexp_blocked(Q, K, scale)

        feat_mat = compute_routing_features_materialized(M, rank=4)
        feat_mf = compute_routing_features_matrix_free(
            Q, K, d_k_mf, scale, lse, rank=4, min_samples=50,
        )

        assert abs(feat_mat.phi_hat - feat_mf.phi_hat) < 0.1, (
            f"phi_hat: materialized={feat_mat.phi_hat:.6f} vs matrix-free={feat_mf.phi_hat:.6f}"
        )

    def test_cheeger_bounds_hold(self):
        """Cheeger bounds should hold in the full pipeline."""
        for seed in range(5):
            _, _, _, _, M, _ = _make_M(16, 4, seed=seed)
            features = compute_routing_features_materialized(M, rank=4)
            gap = 1.0 - features.sigma2
            lower = gap / 2.0
            upper = math.sqrt(max(2.0 * gap, 0.0))
            assert features.phi_hat >= lower - 1e-6, (
                f"seed={seed}: phi_hat={features.phi_hat:.6f} < lower={lower:.6f}"
            )
            assert features.phi_hat <= upper + 1e-6, (
                f"seed={seed}: phi_hat={features.phi_hat:.6f} > upper={upper:.6f}"
            )

    def test_routing_features_phi_hat_range(self):
        """0 ≤ phi_hat ≤ 1 across multiple seeds."""
        for seed in range(10):
            _, _, _, _, M, _ = _make_M(16, 4, seed=seed)
            features = compute_routing_features_materialized(M, rank=4)
            assert 0.0 <= features.phi_hat <= 1.0, (
                f"seed={seed}: phi_hat={features.phi_hat} out of [0, 1]"
            )
