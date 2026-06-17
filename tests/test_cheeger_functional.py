"""Functional tests: batch (materialized) vs streaming (matrix-free) Cheeger.

Tests at realistic scale (L=64..512, D=64) to catch systematic numerical
drift from softmax recomputation, degree accumulation across blocks,
scatter-add ordering, and randomized SVD vector quality.
"""

import math

import pytest
import torch

from glassbox.diagnostics.cheeger import CheegerDiagnostic


def _random_qk(L, D, seed=42):
    torch.manual_seed(seed)
    Q = torch.randn(L, D)
    K = torch.randn(L, D)
    return Q, K


# ===========================================================================
# Batch vs Streaming Agreement
# ===========================================================================


class TestCheegerBatchVsStreaming:
    @pytest.mark.parametrize("L", [64, 128, 256])
    @pytest.mark.parametrize("causal", [False, True])
    def test_phi_star_agreement(self, L, causal):
        """Batch (materialized) and streaming (matrix-free) phi_star must agree."""
        D = 64
        Q, K = _random_qk(L, D, seed=42)

        diag_mat = CheegerDiagnostic(rank=2, threshold=L + 1, causal=causal)
        result_mat = diag_mat.reduce(Q, K, L)

        diag_mf = CheegerDiagnostic(rank=2, threshold=0, block_size=64, causal=causal)
        result_mf = diag_mf.reduce(Q, K, L)

        phi_mat = result_mat["features"].phi_star
        phi_mf = result_mf["features"].phi_star

        atol = 0.05
        rtol = 0.15
        assert abs(phi_mat - phi_mf) < max(atol, rtol * max(phi_mat, 1e-6)), (
            f"L={L} causal={causal}: mat={phi_mat:.6f} vs mf={phi_mf:.6f}, "
            f"gap={abs(phi_mat - phi_mf):.6f}"
        )

    @pytest.mark.parametrize("L", [64, 128])
    def test_witness_agreement(self, L):
        """Cut membership from both paths should largely agree."""
        D = 64
        Q, K = _random_qk(L, D, seed=42)

        diag_mat = CheegerDiagnostic(rank=2, threshold=L + 1)
        diag_mf = CheegerDiagnostic(rank=2, threshold=0, block_size=64)

        w_mat = diag_mat.witness(Q, K, L)
        w_mf = diag_mf.witness(Q, K, L)

        agree_same = (w_mat == w_mf).float().mean()
        agree_flip = (w_mat == -w_mf).float().mean()
        agreement = max(agree_same.item(), agree_flip.item())

        assert agreement > 0.7, f"L={L}: witness agreement={agreement:.2f} (too low)"

    @pytest.mark.parametrize("L", [64, 128, 256])
    def test_cheeger_bounds_at_scale(self, L):
        """Cheeger inequality bracket holds at realistic L."""
        D = 64
        Q, K = _random_qk(L, D, seed=42)

        diag = CheegerDiagnostic(rank=2, threshold=L + 1)
        result = diag.reduce(Q, K, L)
        f = result["features"]

        assert f.cheeger_lower - 1e-6 <= f.phi_star <= f.cheeger_upper + 1e-6, (
            f"L={L}: bounds violated: {f.cheeger_lower:.6f} <= "
            f"{f.phi_star:.6f} <= {f.cheeger_upper:.6f}"
        )

    @pytest.mark.parametrize("block_size", [32, 64, 128])
    def test_block_size_invariance(self, block_size):
        """phi_star should be approximately invariant to block_size."""
        L, D = 128, 64
        Q, K = _random_qk(L, D, seed=42)

        diag = CheegerDiagnostic(rank=2, threshold=0, block_size=block_size)
        result = diag.reduce(Q, K, L)
        phi = result["features"].phi_star

        # Reference with block_size=L (single block = no streaming boundary effects)
        diag_ref = CheegerDiagnostic(rank=2, threshold=0, block_size=L)
        result_ref = diag_ref.reduce(Q, K, L)
        phi_ref = result_ref["features"].phi_star

        assert abs(phi - phi_ref) < 0.02, (
            f"block_size={block_size}: phi={phi:.6f} vs ref={phi_ref:.6f}"
        )


# ===========================================================================
# Scaling Behavior
# ===========================================================================


class TestCheegerScaling:
    @pytest.mark.parametrize("L", [32, 64, 128, 256])
    def test_sigma2_agreement(self, L):
        """sigma2 from materialized and matrix-free should closely agree."""
        D = 64
        Q, K = _random_qk(L, D, seed=42)

        diag_mat = CheegerDiagnostic(rank=2, threshold=L + 1)
        diag_mf = CheegerDiagnostic(rank=2, threshold=0, block_size=64)

        f_mat = diag_mat.reduce(Q, K, L)["features"]
        f_mf = diag_mf.reduce(Q, K, L)["features"]

        assert abs(f_mat.sigma2 - f_mf.sigma2) < 0.05, (
            f"L={L}: sigma2 mat={f_mat.sigma2:.6f} vs mf={f_mf.sigma2:.6f}"
        )

    def test_bounds_bracket_tightness(self):
        """The Cheeger bracket should be reasonably tight (not vacuous)."""
        L, D = 128, 64
        Q, K = _random_qk(L, D, seed=42)

        diag = CheegerDiagnostic(rank=2, threshold=L + 1)
        f = diag.reduce(Q, K, L)["features"]

        bracket_width = f.cheeger_upper - f.cheeger_lower
        assert bracket_width < 1.0, (
            f"Bracket too wide: [{f.cheeger_lower:.4f}, {f.cheeger_upper:.4f}] "
            f"(width={bracket_width:.4f})"
        )

    @pytest.mark.parametrize("seed", range(5))
    def test_reproducibility(self, seed):
        """Same seed should give same results."""
        L, D = 64, 32
        Q, K = _random_qk(L, D, seed=seed)

        diag = CheegerDiagnostic(rank=2, threshold=L + 1)
        f1 = diag.reduce(Q, K, L)["features"]
        f2 = diag.reduce(Q, K, L)["features"]

        assert f1.phi_star == f2.phi_star
        assert f1.sigma2 == f2.sigma2
