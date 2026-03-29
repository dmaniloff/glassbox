import math

import pytest
import torch

from glassbox.laplacian_eigvals import (
    compute_laplacian_eigvals_materialized,
    compute_laplacian_eigvals_matrix_free,
)


class TestMaterialized:
    def test_returns_correct_type(self):
        L, d = 32, 16
        Q = torch.randn(L, d)
        K = torch.randn(L, d)
        scale = 1.0 / math.sqrt(d)
        A = torch.softmax(Q @ K.T * scale, dim=-1)
        feat = compute_laplacian_eigvals_materialized(A, top_k=5)
        assert hasattr(feat, "eigvals")
        assert isinstance(feat.eigvals, list)
        assert len(feat.eigvals) == 5

    def test_eigvals_sorted_descending(self):
        L, d = 64, 16
        Q = torch.randn(L, d)
        K = torch.randn(L, d)
        scale = 1.0 / math.sqrt(d)
        A = torch.softmax(Q @ K.T * scale, dim=-1)
        feat = compute_laplacian_eigvals_materialized(A, top_k=10)
        for i in range(len(feat.eigvals) - 1):
            assert feat.eigvals[i] >= feat.eigvals[i + 1]

    def test_top_k_clamped_to_L(self):
        """top_k larger than L should return L values."""
        L, d = 8, 4
        Q = torch.randn(L, d)
        K = torch.randn(L, d)
        scale = 1.0 / math.sqrt(d)
        A = torch.softmax(Q @ K.T * scale, dim=-1)
        feat = compute_laplacian_eigvals_materialized(A, top_k=100)
        assert len(feat.eigvals) == L

    def test_identity_attention(self):
        """Identity attention: col sums = 1, diag = 1, so diag(L) = 0."""
        L = 16
        A = torch.eye(L)
        feat = compute_laplacian_eigvals_materialized(A, top_k=5)
        for v in feat.eigvals:
            assert v == pytest.approx(0.0, abs=1e-6)

    def test_uniform_attention(self):
        """Uniform attention: col sums = 1, diag = 1/L, so diag(L) = 1 - 1/L."""
        L = 32
        A = torch.ones(L, L) / L
        feat = compute_laplacian_eigvals_materialized(A, top_k=5)
        expected = 1.0 - 1.0 / L
        for v in feat.eigvals:
            assert v == pytest.approx(expected, abs=1e-6)


class TestMatrixFree:
    def test_returns_correct_type(self):
        L, d = 32, 16
        Q = torch.randn(L, d)
        K = torch.randn(L, d)
        scale = 1.0 / math.sqrt(d)
        feat = compute_laplacian_eigvals_matrix_free(Q, K, scale, top_k=5)
        assert hasattr(feat, "eigvals")
        assert isinstance(feat.eigvals, list)
        assert len(feat.eigvals) == 5

    def test_eigvals_sorted_descending(self):
        L, d = 64, 16
        Q = torch.randn(L, d)
        K = torch.randn(L, d)
        scale = 1.0 / math.sqrt(d)
        feat = compute_laplacian_eigvals_matrix_free(Q, K, scale, top_k=10)
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

        mat = compute_laplacian_eigvals_materialized(A, top_k=5)
        mf = compute_laplacian_eigvals_matrix_free(Q, K, scale, top_k=5, block_size=32)
        for v_mat, v_mf in zip(mat.eigvals, mf.eigvals):
            assert v_mat == pytest.approx(v_mf, abs=1e-4)

    def test_agreement_small_blocks(self):
        """Tiny block_size to exercise loop edge cases."""
        L, d = 33, 8  # not a multiple of block_size
        Q = torch.randn(L, d)
        K = torch.randn(L, d)
        scale = 1.0 / math.sqrt(d)
        A = torch.softmax(Q @ K.T * scale, dim=-1)

        mat = compute_laplacian_eigvals_materialized(A, top_k=5)
        mf = compute_laplacian_eigvals_matrix_free(Q, K, scale, top_k=5, block_size=7)
        for v_mat, v_mf in zip(mat.eigvals, mf.eigvals):
            assert v_mat == pytest.approx(v_mf, abs=1e-4)

    def test_agreement_top_k_equals_L(self):
        """All eigenvalues should match when top_k >= L."""
        L, d = 16, 8
        Q = torch.randn(L, d)
        K = torch.randn(L, d)
        scale = 1.0 / math.sqrt(d)
        A = torch.softmax(Q @ K.T * scale, dim=-1)

        mat = compute_laplacian_eigvals_materialized(A, top_k=L)
        mf = compute_laplacian_eigvals_matrix_free(Q, K, scale, top_k=L, block_size=8)
        assert len(mat.eigvals) == len(mf.eigvals) == L
        for v_mat, v_mf in zip(mat.eigvals, mf.eigvals):
            assert v_mat == pytest.approx(v_mf, abs=1e-4)


class TestFrozen:
    def test_frozen(self):
        L, d = 16, 8
        Q = torch.randn(L, d)
        K = torch.randn(L, d)
        scale = 1.0 / math.sqrt(d)
        A = torch.softmax(Q @ K.T * scale, dim=-1)
        feat = compute_laplacian_eigvals_materialized(A, top_k=5)
        with pytest.raises(Exception):
            feat.eigvals = [0.0]
