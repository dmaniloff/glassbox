import math

import pytest
import torch

from glassbox.attention_diagonal import (
    compute_attention_diagonal_features_materialized,
    compute_attention_diagonal_features_matrix_free,
)


class TestMaterialized:
    def test_returns_correct_type(self):
        L, d = 32, 16
        Q = torch.randn(L, d)
        K = torch.randn(L, d)
        scale = 1.0 / math.sqrt(d)
        A = torch.softmax(Q @ K.T * scale, dim=-1)
        feat = compute_attention_diagonal_features_materialized(A)
        assert hasattr(feat, "attn_diag_logmean")
        assert isinstance(feat.attn_diag_logmean, float)

    def test_identity_attention(self):
        """Identity attention matrix has diag = 1.0, so log(1) = 0."""
        L = 16
        A = torch.eye(L)
        feat = compute_attention_diagonal_features_materialized(A)
        assert feat.attn_diag_logmean == pytest.approx(0.0, abs=1e-6)

    def test_uniform_attention(self):
        """Uniform attention has diag = 1/L, so logmean = log(1/L)."""
        L = 32
        A = torch.ones(L, L) / L
        feat = compute_attention_diagonal_features_materialized(A)
        expected = math.log(1.0 / L)
        assert feat.attn_diag_logmean == pytest.approx(expected, abs=1e-5)

    def test_negative_logmean(self):
        """For typical attention, diagonal < 1, so logmean < 0."""
        L, d = 64, 16
        Q = torch.randn(L, d)
        K = torch.randn(L, d)
        scale = 1.0 / math.sqrt(d)
        A = torch.softmax(Q @ K.T * scale, dim=-1)
        feat = compute_attention_diagonal_features_materialized(A)
        # Each diagonal element is a softmax output in [0, 1], log is <= 0
        assert feat.attn_diag_logmean <= 0.0


class TestMatrixFree:
    def test_returns_correct_type(self):
        L, d = 32, 16
        Q = torch.randn(L, d)
        K = torch.randn(L, d)
        scale = 1.0 / math.sqrt(d)
        feat = compute_attention_diagonal_features_matrix_free(Q, K, scale)
        assert hasattr(feat, "attn_diag_logmean")
        assert isinstance(feat.attn_diag_logmean, float)

    def test_negative_logmean(self):
        L, d = 64, 16
        Q = torch.randn(L, d)
        K = torch.randn(L, d)
        scale = 1.0 / math.sqrt(d)
        feat = compute_attention_diagonal_features_matrix_free(Q, K, scale)
        assert feat.attn_diag_logmean <= 0.0


class TestEigvals:
    def test_eigvals_returned_when_top_k(self):
        L, d = 32, 16
        Q = torch.randn(L, d)
        K = torch.randn(L, d)
        scale = 1.0 / math.sqrt(d)
        A = torch.softmax(Q @ K.T * scale, dim=-1)
        feat = compute_attention_diagonal_features_materialized(A, top_k=5)
        assert len(feat.eigvals) == 5
        for i in range(len(feat.eigvals) - 1):
            assert feat.eigvals[i] >= feat.eigvals[i + 1]

    def test_eigvals_empty_when_zero(self):
        L = 16
        A = torch.eye(L)
        feat = compute_attention_diagonal_features_materialized(A, top_k=0)
        assert feat.eigvals == []

    def test_eigvals_in_unit_interval(self):
        L, d = 64, 16
        Q = torch.randn(L, d)
        K = torch.randn(L, d)
        scale = 1.0 / math.sqrt(d)
        A = torch.softmax(Q @ K.T * scale, dim=-1)
        feat = compute_attention_diagonal_features_materialized(A, top_k=10)
        for v in feat.eigvals:
            assert 0.0 <= v <= 1.0

    def test_matrix_free_eigvals(self):
        L, d = 32, 16
        Q = torch.randn(L, d)
        K = torch.randn(L, d)
        scale = 1.0 / math.sqrt(d)
        feat = compute_attention_diagonal_features_matrix_free(Q, K, scale, top_k=5)
        assert len(feat.eigvals) == 5
        for i in range(len(feat.eigvals) - 1):
            assert feat.eigvals[i] >= feat.eigvals[i + 1]


class TestMaterializedVsMatrixFree:
    """Verify the two paths produce identical results."""

    @pytest.mark.parametrize("L", [16, 64, 128])
    def test_agreement(self, L):
        d = 16
        Q = torch.randn(L, d)
        K = torch.randn(L, d)
        scale = 1.0 / math.sqrt(d)
        A = torch.softmax(Q @ K.T * scale, dim=-1)

        mat = compute_attention_diagonal_features_materialized(A)
        mf = compute_attention_diagonal_features_matrix_free(Q, K, scale, block_size=32)
        # Should be numerically identical (same computation, different order)
        assert mat.attn_diag_logmean == pytest.approx(mf.attn_diag_logmean, abs=1e-4)

    def test_agreement_small_blocks(self):
        """Tiny block_size to exercise the loop edge cases."""
        L, d = 33, 8  # not a multiple of block_size
        Q = torch.randn(L, d)
        K = torch.randn(L, d)
        scale = 1.0 / math.sqrt(d)
        A = torch.softmax(Q @ K.T * scale, dim=-1)

        mat = compute_attention_diagonal_features_materialized(A)
        mf = compute_attention_diagonal_features_matrix_free(Q, K, scale, block_size=7)
        assert mat.attn_diag_logmean == pytest.approx(mf.attn_diag_logmean, abs=1e-4)

    @pytest.mark.parametrize("L", [16, 64])
    def test_eigvals_agreement(self, L):
        d = 16
        Q = torch.randn(L, d)
        K = torch.randn(L, d)
        scale = 1.0 / math.sqrt(d)
        A = torch.softmax(Q @ K.T * scale, dim=-1)

        mat = compute_attention_diagonal_features_materialized(A, top_k=5)
        mf = compute_attention_diagonal_features_matrix_free(Q, K, scale, top_k=5, block_size=32)
        for v_mat, v_mf in zip(mat.eigvals, mf.eigvals):
            assert v_mat == pytest.approx(v_mf, abs=1e-4)
