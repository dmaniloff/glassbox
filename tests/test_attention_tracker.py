"""Tests for AttentionTracker features on raw attention matrix A.

Validates materialized and matrix-free paths, cross-checks their agreement,
and verifies the d_k_inv_sqrt=ones trick that reuses M-family matvecs for A.
"""

import math

import torch

from glassbox.attention_tracker import (
    compute_attention_tracker_features_materialized,
    compute_attention_tracker_features_matrix_free,
)
from glassbox.results import AttentionTrackerFeatures
from glassbox.svd import (
    apply_A_blocked,
    compute_M_fro_norm_blocked,
    matvec_M_blocked,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_A(L, D, seed=42):
    """Generate Q, K, scale, and materialized A."""
    torch.manual_seed(seed)
    Q = torch.randn(L, D)
    K = torch.randn(L, D)
    scale = 1.0 / math.sqrt(D)
    A = torch.softmax(Q @ K.T * scale, dim=-1)
    return Q, K, scale, A


# ---------------------------------------------------------------------------
# 1. Materialized path
# ---------------------------------------------------------------------------


class TestMaterialized:
    def test_returns_correct_type(self):
        _, _, _, A = _make_A(32, 16)
        feats = compute_attention_tracker_features_materialized(A, rank=4)
        assert isinstance(feats, AttentionTrackerFeatures)

    def test_fields_populated(self):
        _, _, _, A = _make_A(32, 16)
        feats = compute_attention_tracker_features_materialized(A, rank=4)
        assert feats.sigma2 is not None
        assert feats.sigma2_asym is not None
        assert feats.commutator_norm is not None
        assert feats.sv1 is not None
        assert feats.sv_ratio is not None
        assert feats.sv_entropy is not None

    def test_singular_values_length(self):
        _, _, _, A = _make_A(32, 16)
        feats = compute_attention_tracker_features_materialized(A, rank=4)
        assert len(feats.singular_values) == 4

    def test_value_ranges(self):
        _, _, _, A = _make_A(32, 16)
        feats = compute_attention_tracker_features_materialized(A, rank=4)
        assert feats.sigma2 >= 0
        assert feats.sigma2_asym >= 0
        assert feats.commutator_norm >= 0

    def test_sigma2_matches_svdvals(self):
        _, _, _, A = _make_A(32, 16)
        feats = compute_attention_tracker_features_materialized(A, rank=4)
        sigma = torch.linalg.svdvals(A)
        assert abs(feats.sigma2 - sigma[1].item()) < 1e-6


# ---------------------------------------------------------------------------
# 2. Matrix-free path
# ---------------------------------------------------------------------------


class TestMatrixFree:
    def test_returns_correct_type(self):
        Q, K, scale, _ = _make_A(32, 16)
        feats = compute_attention_tracker_features_matrix_free(
            Q, K, scale, rank=4, block_size=16
        )
        assert isinstance(feats, AttentionTrackerFeatures)

    def test_fields_populated(self):
        Q, K, scale, _ = _make_A(32, 16)
        feats = compute_attention_tracker_features_matrix_free(
            Q, K, scale, rank=4, block_size=16
        )
        assert feats.sigma2 is not None
        assert feats.sigma2_asym is not None
        assert feats.commutator_norm is not None

    def test_value_ranges(self):
        Q, K, scale, _ = _make_A(32, 16)
        feats = compute_attention_tracker_features_matrix_free(
            Q, K, scale, rank=4, block_size=16
        )
        assert feats.sigma2 >= 0
        assert feats.sigma2_asym >= 0
        assert feats.commutator_norm >= 0


# ---------------------------------------------------------------------------
# 3. Materialized vs matrix-free agreement
# ---------------------------------------------------------------------------


class TestAgreement:
    def _compare(self, L, D, seed):
        Q, K, scale, A = _make_A(L, D, seed=seed)
        mat = compute_attention_tracker_features_materialized(A, rank=4)
        mf = compute_attention_tracker_features_matrix_free(
            Q, K, scale, rank=4, block_size=16
        )
        return mat, mf

    def test_sigma2_agreement(self):
        for seed in [42, 123, 456]:
            mat, mf = self._compare(32, 16, seed)
            assert abs(mat.sigma2 - mf.sigma2) < 0.05, (
                f"sigma2 mismatch seed={seed}: mat={mat.sigma2}, mf={mf.sigma2}"
            )

    def test_sigma2_asym_agreement(self):
        for seed in [42, 123, 456]:
            mat, mf = self._compare(32, 16, seed)
            assert abs(mat.sigma2_asym - mf.sigma2_asym) < 0.1, (
                f"sigma2_asym mismatch seed={seed}: mat={mat.sigma2_asym}, mf={mf.sigma2_asym}"
            )

    def test_commutator_norm_agreement(self):
        for seed in [42, 123, 456]:
            mat, mf = self._compare(32, 16, seed)
            assert abs(mat.commutator_norm - mf.commutator_norm) < 0.15, (
                f"commutator_norm mismatch seed={seed}: mat={mat.commutator_norm}, mf={mf.commutator_norm}"
            )


# ---------------------------------------------------------------------------
# 4. Mathematical properties
# ---------------------------------------------------------------------------


class TestMathProperties:
    def test_symmetric_A_has_zero_asym_features(self):
        """A truly symmetric A has sigma2_asym and commutator_norm ~ 0."""
        torch.manual_seed(42)
        L = 32
        # Construct a symmetric doubly-stochastic-ish A directly
        # (softmax(QK^T) is NOT symmetric even when Q==K because rows are
        # independently normalized)
        raw = torch.randn(L, L)
        raw = (raw + raw.T) / 2  # symmetric
        A = torch.softmax(raw, dim=-1)
        A = (A + A.T) / 2  # force exact symmetry after softmax
        feats = compute_attention_tracker_features_materialized(A, rank=4)
        assert feats.sigma2_asym < 1e-6, f"sigma2_asym should be ~0, got {feats.sigma2_asym}"
        assert feats.commutator_norm < 1e-6, f"commutator_norm should be ~0, got {feats.commutator_norm}"


# ---------------------------------------------------------------------------
# 5. d_k_inv_sqrt = ones trick
# ---------------------------------------------------------------------------


class TestOnesTrick:
    def test_matvec_M_with_ones_equals_A(self):
        """matvec_M_blocked(Q, K, v, ones, scale, bs) == apply_A_blocked(Q, K, v, scale, bs)."""
        Q, K, scale, _ = _make_A(32, 16)
        ones = torch.ones(Q.shape[0])
        v = torch.randn(Q.shape[0])
        result_M = matvec_M_blocked(Q, K, v, ones, scale, block_size=16)
        result_A = apply_A_blocked(Q, K, v, scale, block_size=16)
        assert torch.allclose(result_M, result_A, atol=1e-6), (
            f"max diff: {(result_M - result_A).abs().max()}"
        )

    def test_fro_norm_with_ones_equals_A_fro(self):
        """compute_M_fro_norm_blocked with ones equals ||A||_F."""
        Q, K, scale, A = _make_A(32, 16)
        ones = torch.ones(Q.shape[0])
        fro_mf = compute_M_fro_norm_blocked(Q, K, ones, scale, block_size=16).item()
        fro_mat = torch.linalg.norm(A, "fro").item()
        assert abs(fro_mf - fro_mat) < 1e-4, (
            f"||A||_F mismatch: mf={fro_mf}, mat={fro_mat}"
        )
