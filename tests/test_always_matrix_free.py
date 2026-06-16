"""Tests for always-matrix-free mode (threshold=0).

Issue #36: verify that threshold=0 forces the matrix-free tier on every
threshold-based diagnostic, and that outputs match the materialized
reference within tolerance.
"""

import pytest
import torch

from glassbox.config import THRESHOLD_SIGNALS, GlassboxConfig
from glassbox.diagnostics import (
    LaplacianDiagnostic,
    RoutingDiagnostic,
    SelfAttnDiagnostic,
    TrackerDiagnostic,
)

L = 32
D = 16
RANK = 2
TOP_K = 5


@pytest.fixture
def qk():
    torch.manual_seed(99)
    Q = torch.randn(L, D)
    K = torch.randn(L, D)
    return Q, K


class TestAlwaysMatrixFreeConfig:
    def test_threshold_zero_valid(self):
        cfg = GlassboxConfig(routing={"enabled": True, "threshold": 0})
        assert cfg.routing.threshold == 0

    def test_from_cli_args_threshold_zero(self):
        cfg = GlassboxConfig.from_cli_args(
            signals=("routing", "tracker", "selfattn", "laplacian"),
            threshold=0,
        )
        for sig in THRESHOLD_SIGNALS:
            assert getattr(cfg, sig).threshold == 0

    def test_threshold_signals_complete(self):
        assert THRESHOLD_SIGNALS == {"routing", "tracker", "selfattn", "laplacian"}


class TestRoutingAlwaysMatrixFree:
    def test_threshold_zero_forces_matrix_free(self, qk):
        Q, K = qk
        diag = RoutingDiagnostic(rank=RANK, threshold=0, block_size=8)
        result = diag.reduce(Q, K, L)
        assert result["tier"] == "matrix_free"

    def test_matrix_free_matches_materialized(self, qk):
        Q, K = qk
        mat = RoutingDiagnostic(rank=RANK, threshold=1024)
        mf = RoutingDiagnostic(rank=RANK, threshold=0, block_size=8)
        r_mat = mat.reduce(Q, K, L)
        r_mf = mf.reduce(Q, K, L)
        f_mat, f_mf = r_mat["features"], r_mf["features"]
        torch.testing.assert_close(
            torch.tensor(f_mat.singular_values),
            torch.tensor(f_mf.singular_values),
            atol=5e-3,
            rtol=5e-2,
        )
        assert abs(f_mat.G - f_mf.G) < 0.05
        assert abs(f_mat.sigma2 - f_mf.sigma2) < 5e-3


class TestTrackerAlwaysMatrixFree:
    def test_threshold_zero_forces_matrix_free(self, qk):
        Q, K = qk
        diag = TrackerDiagnostic(rank=RANK, threshold=0, block_size=8)
        result = diag.reduce(Q, K, L)
        assert result["tier"] == "matrix_free"

    def test_matrix_free_matches_materialized(self, qk):
        Q, K = qk
        mat = TrackerDiagnostic(rank=RANK, threshold=1024)
        mf = TrackerDiagnostic(rank=RANK, threshold=0, block_size=8)
        r_mat = mat.reduce(Q, K, L)
        r_mf = mf.reduce(Q, K, L)
        f_mat, f_mf = r_mat["features"], r_mf["features"]
        torch.testing.assert_close(
            torch.tensor(f_mat.singular_values),
            torch.tensor(f_mf.singular_values),
            atol=5e-3,
            rtol=5e-2,
        )
        assert abs(f_mat.sigma2 - f_mf.sigma2) < 5e-3


class TestSelfAttnAlwaysMatrixFree:
    def test_threshold_zero_forces_matrix_free(self, qk):
        Q, K = qk
        diag = SelfAttnDiagnostic(top_k=TOP_K, threshold=0, block_size=8)
        result = diag.reduce(Q, K, L)
        assert result["tier"] == "matrix_free"

    def test_matrix_free_matches_materialized(self, qk):
        Q, K = qk
        mat = SelfAttnDiagnostic(top_k=TOP_K, threshold=1024)
        mf = SelfAttnDiagnostic(top_k=TOP_K, threshold=0, block_size=8)
        r_mat = mat.reduce(Q, K, L)
        r_mf = mf.reduce(Q, K, L)
        f_mat, f_mf = r_mat["features"], r_mf["features"]
        assert abs(f_mat.attn_diag_logmean - f_mf.attn_diag_logmean) < 1e-4
        torch.testing.assert_close(
            torch.tensor(f_mat.eigvals),
            torch.tensor(f_mf.eigvals),
            atol=1e-4,
            rtol=1e-3,
        )


class TestLaplacianAlwaysMatrixFree:
    def test_threshold_zero_forces_matrix_free(self, qk):
        Q, K = qk
        diag = LaplacianDiagnostic(top_k=TOP_K, threshold=0, block_size=8)
        result = diag.reduce(Q, K, L)
        assert result["tier"] == "matrix_free"

    def test_matrix_free_matches_materialized(self, qk):
        Q, K = qk
        mat = LaplacianDiagnostic(top_k=TOP_K, threshold=1024)
        mf = LaplacianDiagnostic(top_k=TOP_K, threshold=0, block_size=8)
        r_mat = mat.reduce(Q, K, L)
        r_mf = mf.reduce(Q, K, L)
        f_mat, f_mf = r_mat["features"], r_mf["features"]
        torch.testing.assert_close(
            torch.tensor(f_mat.eigvals),
            torch.tensor(f_mf.eigvals),
            atol=1e-4,
            rtol=1e-3,
        )


class TestNoMaterialization:
    """Verify threshold=0 never allocates an L×L attention matrix."""

    @staticmethod
    def _check_no_lxl_alloc(diag, Q, K, seq_len):
        """Run reduce and verify no L×L tensor was allocated."""

        class AllocTracker:
            def __init__(self):
                self.shapes = []

            def __enter__(self):
                self._orig = torch.empty
                tracker = self

                def patched_empty(*args, **kwargs):
                    result = tracker._orig(*args, **kwargs)
                    tracker.shapes.append(result.shape)
                    return result

                torch.empty = patched_empty
                return self

            def __exit__(self, *args):
                torch.empty = self._orig

        with AllocTracker() as tracker:
            diag.reduce(Q, K, seq_len)

        lxl = [s for s in tracker.shapes if len(s) == 2 and s[0] == seq_len and s[1] == seq_len]
        return lxl

    def test_routing_no_lxl(self, qk):
        Q, K = qk
        diag = RoutingDiagnostic(rank=RANK, threshold=0, block_size=8)
        lxl = self._check_no_lxl_alloc(diag, Q, K, L)
        assert lxl == [], f"Unexpected L×L allocations via torch.empty: {lxl}"

    def test_selfattn_no_lxl(self, qk):
        Q, K = qk
        diag = SelfAttnDiagnostic(top_k=TOP_K, threshold=0, block_size=8)
        lxl = self._check_no_lxl_alloc(diag, Q, K, L)
        assert lxl == [], f"Unexpected L×L allocations via torch.empty: {lxl}"

    def test_laplacian_no_lxl(self, qk):
        Q, K = qk
        diag = LaplacianDiagnostic(top_k=TOP_K, threshold=0, block_size=8)
        lxl = self._check_no_lxl_alloc(diag, Q, K, L)
        assert lxl == [], f"Unexpected L×L allocations via torch.empty: {lxl}"

    def test_tracker_no_lxl(self, qk):
        Q, K = qk
        diag = TrackerDiagnostic(rank=RANK, threshold=0, block_size=8)
        lxl = self._check_no_lxl_alloc(diag, Q, K, L)
        assert lxl == [], f"Unexpected L×L allocations via torch.empty: {lxl}"
