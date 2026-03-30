"""Tests for causal masking in attention reconstruction (issue #13).

Verifies that:
  1. causal=True produces lower-triangular attention (future tokens masked)
  2. causal=True changes feature values vs causal=False
  3. Materialized and matrix-free paths agree under causal=True
  4. Config defaults causal=True for all post-softmax signals
"""

import math

import pytest
import torch

from glassbox.attention_diagonal import (
    compute_attention_diagonal_features_materialized,
    compute_attention_diagonal_features_matrix_free,
)
from glassbox.attention_tracker import (
    compute_attention_tracker_features_materialized,
    compute_attention_tracker_features_matrix_free,
)
from glassbox.config import LaplacianConfig, RoutingConfig, SelfAttnConfig, TrackerConfig
from glassbox.laplacian_eigvals import (
    compute_laplacian_eigvals_materialized,
    compute_laplacian_eigvals_matrix_free,
)
from glassbox.svd import (
    _mask_causal,
    apply_A_blocked,
    apply_AT_blocked,
    compute_degree_normalized_M,
    compute_dk_blocked,
    compute_logsumexp_blocked,
    compute_M_fro_norm_blocked,
    get_M_entries_batch,
    matvec_M_blocked,
    matvec_MT_blocked,
)

L = 16
D = 4


@pytest.fixture
def qk():
    torch.manual_seed(42)
    Q = torch.randn(L, D)
    K = torch.randn(L, D)
    scale = 1.0 / math.sqrt(D)
    return Q, K, scale


def _ref_A_causal(Q, K, scale):
    """Reference causal attention: mask then softmax."""
    scores = Q @ K.T * scale
    mask = torch.tril(torch.ones(Q.shape[0], K.shape[0], dtype=torch.bool))
    scores = scores.masked_fill(~mask, float("-inf"))
    return torch.softmax(scores, dim=-1)


# ---------------------------------------------------------------------------
# _mask_causal helper
# ---------------------------------------------------------------------------


class TestMaskCausal:
    def test_zeros_above_diagonal(self):
        scores = torch.ones(4, 8)
        masked = _mask_causal(scores, row_offset=0)
        # Row 0: only col 0 allowed, cols 1-7 should be -inf
        assert masked[0, 0] == 1.0
        assert masked[0, 1] == float("-inf")
        # Row 3: cols 0-3 allowed, cols 4-7 should be -inf
        assert masked[3, 3] == 1.0
        assert masked[3, 4] == float("-inf")

    def test_row_offset(self):
        scores = torch.ones(2, 8)
        masked = _mask_causal(scores, row_offset=4)
        # Row 0 is global row 4: cols 0-4 allowed, col 5+ masked
        assert masked[0, 4] == 1.0
        assert masked[0, 5] == float("-inf")

    def test_softmax_after_mask_is_lower_triangular(self):
        torch.manual_seed(0)
        scores = torch.randn(6, 6)
        masked = _mask_causal(scores, row_offset=0)
        A = torch.softmax(masked, dim=-1)
        # Upper triangle should be 0
        upper = torch.triu(A, diagonal=1)
        assert torch.allclose(upper, torch.zeros_like(upper), atol=1e-7)
        # Rows should sum to 1
        torch.testing.assert_close(A.sum(dim=-1), torch.ones(6), atol=1e-6, rtol=0)


# ---------------------------------------------------------------------------
# Core blocked functions with causal=True
# ---------------------------------------------------------------------------


class TestBlockedCausal:
    def test_apply_A_blocked_causal(self, qk):
        Q, K, scale = qk
        A = _ref_A_causal(Q, K, scale)
        v = torch.randn(L)
        expected = A @ v
        result = apply_A_blocked(Q, K, v, scale, block_size=4, causal=True)
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_apply_AT_blocked_causal(self, qk):
        Q, K, scale = qk
        A = _ref_A_causal(Q, K, scale)
        u = torch.randn(L)
        expected = A.T @ u
        result = apply_AT_blocked(Q, K, u, scale, block_size=4, causal=True)
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_logsumexp_blocked_causal(self, qk):
        Q, K, scale = qk
        scores = Q @ K.T * scale
        mask = torch.tril(torch.ones(L, L, dtype=torch.bool))
        scores_masked = scores.masked_fill(~mask, float("-inf"))
        expected = torch.logsumexp(scores_masked, dim=-1)
        result = compute_logsumexp_blocked(Q, K, scale, block_size=4, causal=True)
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_compute_dk_blocked_causal(self, qk):
        Q, K, scale = qk
        A = _ref_A_causal(Q, K, scale)
        expected_dk = A.sum(dim=0)
        dk, _ = compute_dk_blocked(Q, K, scale, block_size=4, causal=True)
        torch.testing.assert_close(dk, expected_dk, atol=1e-5, rtol=1e-5)

    def test_M_fro_norm_blocked_causal(self, qk):
        Q, K, scale = qk
        A = _ref_A_causal(Q, K, scale)
        M, _, d_k_inv_sqrt = compute_degree_normalized_M(A)
        expected = torch.linalg.norm(M, "fro")
        _, d_k_inv_sqrt_mf = compute_dk_blocked(Q, K, scale, block_size=4, causal=True)
        result = compute_M_fro_norm_blocked(Q, K, d_k_inv_sqrt_mf, scale, block_size=4, causal=True)
        torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-3)

    def test_get_M_entries_batch_causal(self, qk):
        Q, K, scale = qk
        A = _ref_A_causal(Q, K, scale)
        M, _, d_k_inv_sqrt = compute_degree_normalized_M(A)
        lse = compute_logsumexp_blocked(Q, K, scale, causal=True)

        # Test entries both below and above diagonal
        ii = torch.tensor([0, 1, 2, 5, 3, 0])
        jj = torch.tensor([0, 0, 1, 3, 8, 5])  # last two: j > i (should be 0)
        result = get_M_entries_batch(Q, K, lse, d_k_inv_sqrt, scale, ii, jj, causal=True)
        expected = M[ii, jj]
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-4)

    def test_matvec_M_blocked_causal(self, qk):
        Q, K, scale = qk
        A = _ref_A_causal(Q, K, scale)
        M, _, d_k_inv_sqrt = compute_degree_normalized_M(A)
        v = torch.randn(L)
        expected = M @ v
        _, d_k_inv_sqrt_mf = compute_dk_blocked(Q, K, scale, block_size=4, causal=True)
        result = matvec_M_blocked(Q, K, v, d_k_inv_sqrt_mf, scale, block_size=4, causal=True)
        torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-3)

    def test_matvec_MT_blocked_causal(self, qk):
        Q, K, scale = qk
        A = _ref_A_causal(Q, K, scale)
        M, _, d_k_inv_sqrt = compute_degree_normalized_M(A)
        u = torch.randn(L)
        expected = M.T @ u
        _, d_k_inv_sqrt_mf = compute_dk_blocked(Q, K, scale, block_size=4, causal=True)
        result = matvec_MT_blocked(Q, K, u, d_k_inv_sqrt_mf, scale, block_size=4, causal=True)
        torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-3)


# ---------------------------------------------------------------------------
# Causal changes feature values
# ---------------------------------------------------------------------------


class TestCausalChangesValues:
    def test_causal_attention_is_lower_triangular(self, qk):
        """Causal softmax should produce a lower-triangular attention matrix."""
        Q, K, scale = qk
        A = _ref_A_causal(Q, K, scale)
        upper = torch.triu(A, diagonal=1)
        assert torch.allclose(upper, torch.zeros_like(upper), atol=1e-7)

    def test_causal_changes_logsumexp(self, qk):
        """Causal logsumexp should differ from non-causal (fewer terms in sum)."""
        Q, K, scale = qk
        lse_full = compute_logsumexp_blocked(Q, K, scale, block_size=4, causal=False)
        lse_causal = compute_logsumexp_blocked(Q, K, scale, block_size=4, causal=True)
        # Row 0 should be the same (only 1 token either way)
        # But later rows should differ
        assert not torch.allclose(lse_full[1:], lse_causal[1:])

    def test_causal_changes_diagonal_features(self, qk):
        Q, K, scale = qk
        f_full = compute_attention_diagonal_features_matrix_free(
            Q, K, scale, top_k=5, block_size=4, causal=False
        )
        f_causal = compute_attention_diagonal_features_matrix_free(
            Q, K, scale, top_k=5, block_size=4, causal=True
        )
        assert f_full.attn_diag_logmean != f_causal.attn_diag_logmean

    def test_causal_changes_laplacian_features(self, qk):
        Q, K, scale = qk
        f_full = compute_laplacian_eigvals_matrix_free(
            Q, K, scale, top_k=5, block_size=4, causal=False
        )
        f_causal = compute_laplacian_eigvals_matrix_free(
            Q, K, scale, top_k=5, block_size=4, causal=True
        )
        assert f_full.eigvals != f_causal.eigvals


# ---------------------------------------------------------------------------
# Materialized vs matrix-free agreement under causal=True
# ---------------------------------------------------------------------------


class TestCausalTwoTierAgreement:
    def test_selfattn_agreement(self, qk):
        Q, K, scale = qk
        A = _ref_A_causal(Q, K, scale)
        f_mat = compute_attention_diagonal_features_materialized(A, top_k=5)
        f_mf = compute_attention_diagonal_features_matrix_free(
            Q, K, scale, top_k=5, block_size=4, causal=True
        )
        assert abs(f_mat.attn_diag_logmean - f_mf.attn_diag_logmean) < 1e-4
        for v_mat, v_mf in zip(f_mat.eigvals, f_mf.eigvals):
            assert abs(v_mat - v_mf) < 1e-4

    def test_laplacian_agreement(self, qk):
        Q, K, scale = qk
        A = _ref_A_causal(Q, K, scale)
        f_mat = compute_laplacian_eigvals_materialized(A, top_k=5)
        f_mf = compute_laplacian_eigvals_matrix_free(
            Q, K, scale, top_k=5, block_size=4, causal=True
        )
        for v_mat, v_mf in zip(f_mat.eigvals, f_mf.eigvals):
            assert abs(v_mat - v_mf) < 1e-4

    def test_tracker_agreement(self):
        torch.manual_seed(77)
        L_test = 32
        D_test = 8
        Q = torch.randn(L_test, D_test)
        K = torch.randn(L_test, D_test)
        scale = 1.0 / math.sqrt(D_test)

        A = _ref_A_causal(Q, K, scale)
        f_mat = compute_attention_tracker_features_materialized(A, rank=4)
        f_mf = compute_attention_tracker_features_matrix_free(
            Q, K, scale, rank=4, block_size=8, causal=True
        )
        assert abs(f_mat.sigma2 - f_mf.sigma2) < 0.05


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------


class TestConfigCausalDefaults:
    def test_routing_causal_default(self):
        assert RoutingConfig().causal is True

    def test_tracker_causal_default(self):
        assert TrackerConfig().causal is True

    def test_selfattn_causal_default(self):
        assert SelfAttnConfig().causal is True

    def test_laplacian_causal_default(self):
        assert LaplacianConfig().causal is True

    def test_causal_can_be_disabled(self):
        cfg = RoutingConfig(causal=False)
        assert cfg.causal is False
