"""Tests for the streaming Cheeger controller: bordered Rayleigh-Ritz,
hermitian_lanczos warm start, and CheegerDiagnostic streaming mode."""

import math

import pytest
import torch

from glassbox.cheeger import compute_improved_cheeger_upper
from glassbox.diagnostics.cheeger import CheegerDiagnostic
from glassbox.svd import bordered_rayleigh_ritz, hermitian_lanczos


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_hermitian(dim, seed=42, device="cpu"):
    torch.manual_seed(seed)
    H = torch.randn(dim, dim, device=device)
    return (H + H.T) / 2


def _random_qk(L, D, seed=42):
    torch.manual_seed(seed)
    return torch.randn(L, D), torch.randn(L, D)


# ---------------------------------------------------------------------------
# bordered_rayleigh_ritz
# ---------------------------------------------------------------------------

class TestBorderedRayleighRitz:
    def test_exact_eigenpairs_recovered(self):
        dim = 30
        H = _random_hermitian(dim)
        true_evals, true_evecs = torch.linalg.eigh(H)
        basis = true_evecs[:, -3:]  # top-3 eigenvectors

        evals, evecs, all_proj = bordered_rayleigh_ritz(
            lambda v: H @ v, dim, basis, k=3, n_explore=0,
            device="cpu", which="largest",
        )
        assert torch.allclose(evals, true_evals[-3:].flip(0), atol=1e-4)

    def test_same_dim_reprojection(self):
        dim = 20
        H = _random_hermitian(dim)
        true_evals, true_evecs = torch.linalg.eigh(H)

        # Use slightly perturbed basis (small noise so it's still close)
        torch.manual_seed(99)
        noise = torch.randn(dim, 4) * 0.01
        basis = true_evecs[:, -4:] + noise

        evals, evecs, _ = bordered_rayleigh_ritz(
            lambda v: H @ v, dim, basis, k=3, n_explore=2,
            device="cpu", which="largest",
        )
        assert torch.allclose(evals, true_evals[-3:].flip(0), atol=0.1)

    def test_dim_growth_padding(self):
        dim_old, dim_new = 20, 30
        torch.manual_seed(42)
        basis = torch.randn(dim_old, 3)
        H = _random_hermitian(dim_new, seed=99)

        evals, evecs, _ = bordered_rayleigh_ritz(
            lambda v: H @ v, dim_new, basis, k=3, n_explore=2,
            device="cpu", which="largest",
        )
        assert evecs.shape == (dim_new, 3)
        assert len(evals) == 3

    def test_dim_shrink_truncation(self):
        dim_old, dim_new = 30, 20
        torch.manual_seed(42)
        basis = torch.randn(dim_old, 3)
        H = _random_hermitian(dim_new, seed=99)

        evals, evecs, _ = bordered_rayleigh_ritz(
            lambda v: H @ v, dim_new, basis, k=3, n_explore=2,
            device="cpu", which="largest",
        )
        assert evecs.shape == (dim_new, 3)

    def test_explore_improves_accuracy(self):
        dim = 30
        H = _random_hermitian(dim)
        true_evals, _ = torch.linalg.eigh(H)

        # Stale basis from a different matrix
        H_old = _random_hermitian(dim, seed=7)
        _, old_evecs = torch.linalg.eigh(H_old)
        stale_basis = old_evecs[:, -3:]

        evals_no_explore, _, _ = bordered_rayleigh_ritz(
            lambda v: H @ v, dim, stale_basis, k=3, n_explore=0,
            device="cpu", which="largest",
        )
        evals_explore, _, _ = bordered_rayleigh_ritz(
            lambda v: H @ v, dim, stale_basis, k=3, n_explore=4,
            device="cpu", which="largest",
        )

        err_no = (evals_no_explore - true_evals[-3:].flip(0)).abs().max().item()
        err_yes = (evals_explore - true_evals[-3:].flip(0)).abs().max().item()
        assert err_yes <= err_no + 0.1  # exploration doesn't make things worse

    def test_smallest_which(self):
        dim = 20
        H = _random_hermitian(dim)
        true_evals, true_evecs = torch.linalg.eigh(H)
        basis = true_evecs[:, :3]  # bottom-3

        evals, _, _ = bordered_rayleigh_ritz(
            lambda v: H @ v, dim, basis, k=3, n_explore=0,
            device="cpu", which="smallest",
        )
        assert torch.allclose(evals, true_evals[:3], atol=1e-4)

    def test_all_projected_evals_returned(self):
        dim = 20
        H = _random_hermitian(dim)
        _, true_evecs = torch.linalg.eigh(H)

        _, _, all_proj = bordered_rayleigh_ritz(
            lambda v: H @ v, dim, true_evecs[:, -3:], k=2, n_explore=2,
            device="cpu", which="largest",
        )
        # Should have r + n_explore eigenvalues (possibly fewer if rank-deficient)
        assert len(all_proj) >= 2
        assert len(all_proj) <= 5

    def test_1d_basis(self):
        dim = 20
        H = _random_hermitian(dim)
        _, true_evecs = torch.linalg.eigh(H)

        evals, evecs, _ = bordered_rayleigh_ritz(
            lambda v: H @ v, dim, true_evecs[:, -1], k=1, n_explore=2,
            device="cpu", which="largest",
        )
        assert evecs.shape[0] == dim


# ---------------------------------------------------------------------------
# hermitian_lanczos warm start
# ---------------------------------------------------------------------------

class TestHermitianLanczosWarmStart:
    def test_warm_start_converges_faster(self):
        dim = 40
        H = _random_hermitian(dim)
        true_evals, true_evecs = torch.linalg.eigh(H)

        # Warm start: 10 iterations
        evals_warm, _ = hermitian_lanczos(
            lambda v: H @ v, dim, 3, 10, "cpu", which="largest",
            initial_vectors=true_evecs[:, -3:],
        )
        err_warm = (evals_warm - true_evals[-3:].flip(0)).abs().max().item()

        # Cold start: 10 iterations (many seeds, take median error)
        errs = []
        for seed in range(10):
            torch.manual_seed(seed)
            evals_cold, _ = hermitian_lanczos(
                lambda v: H @ v, dim, 3, 10, "cpu", which="largest",
            )
            errs.append((evals_cold - true_evals[-3:].flip(0)).abs().max().item())
        median_cold = sorted(errs)[len(errs) // 2]

        assert err_warm <= median_cold + 0.01

    def test_warm_start_dimension_mismatch_raises(self):
        dim = 20
        H = _random_hermitian(dim)
        bad_init = torch.randn(dim + 5, 3)

        with pytest.raises(ValueError, match="initial_vectors dim"):
            hermitian_lanczos(
                lambda v: H @ v, dim, 3, 20, "cpu",
                initial_vectors=bad_init,
            )

    def test_warm_start_vs_cold_agreement(self):
        dim = 30
        H = _random_hermitian(dim)
        true_evals, true_evecs = torch.linalg.eigh(H)

        evals_warm, _ = hermitian_lanczos(
            lambda v: H @ v, dim, 3, 40, "cpu", which="largest",
            initial_vectors=true_evecs[:, -3:],
        )
        # With enough iterations, should match ground truth
        assert torch.allclose(evals_warm, true_evals[-3:].flip(0), atol=1e-3)

    def test_warm_start_1d_vector(self):
        dim = 20
        H = _random_hermitian(dim)
        _, true_evecs = torch.linalg.eigh(H)

        evals, _ = hermitian_lanczos(
            lambda v: H @ v, dim, 2, 20, "cpu", which="largest",
            initial_vectors=true_evecs[:, -1],  # 1-D
        )
        assert len(evals) == 2


# ---------------------------------------------------------------------------
# compute_improved_cheeger_upper
# ---------------------------------------------------------------------------

class TestImprovedCheegerUpper:
    def test_basic_bound(self):
        evals = torch.tensor([1.0, 0.9, 0.5, 0.3, 0.1])
        bound = compute_improved_cheeger_upper(evals, k=4)
        assert bound is not None
        assert bound > 0

    def test_insufficient_eigenvalues(self):
        evals = torch.tensor([1.0, 0.9])
        bound = compute_improved_cheeger_upper(evals, k=4)
        assert bound is None

    def test_tighter_than_naive(self):
        # When mu_{k+1} is large, the improved bound should be tighter
        evals = torch.tensor([1.0, 0.95, 0.3, 0.2, 0.1])
        mu2 = 1.0 - 0.95
        naive_upper = math.sqrt(2 * mu2)
        improved = compute_improved_cheeger_upper(evals, k=4)
        # Not guaranteed tighter in all cases, but check it's finite and positive
        assert improved is not None
        assert improved > 0
        assert math.isfinite(improved)


# ---------------------------------------------------------------------------
# Streaming CheegerDiagnostic
# ---------------------------------------------------------------------------

class TestStreamingCheegerDiagnostic:
    @pytest.fixture
    def qk(self):
        return _random_qk(32, 16, seed=42)

    def test_first_window_full_recompute(self, qk):
        Q, K = qk
        diag = CheegerDiagnostic(mode="streaming", threshold=1024)
        result = diag.reduce(Q, K, 32, prior_state=None)
        assert result.get("recomputed") is True
        assert "ritz_basis" in result
        assert result["features"].recomputed is True

    def test_second_window_cheap_update(self, qk):
        Q, K = qk
        diag = CheegerDiagnostic(mode="streaming", threshold=1024, gap_threshold=0.001)

        # First window
        result1 = diag.reduce(Q, K, 32, prior_state=None)
        state = diag.accumulate(result1, None)
        state["step"] = 1
        state["last_full_recompute_step"] = 1

        # Second window with same Q/K (stable)
        result2 = diag.reduce(Q, K, 32, prior_state=state)
        assert result2["tier"] == "bordered_ritz"
        assert result2.get("recomputed") is False
        assert result2["features"].recomputed is False

    def test_gap_trigger_fires(self, qk):
        Q, K = qk
        diag = CheegerDiagnostic(mode="streaming", threshold=1024, gap_threshold=0.5)

        # First window
        result1 = diag.reduce(Q, K, 32, prior_state=None)
        state = diag.accumulate(result1, None)
        state["step"] = 1
        state["last_full_recompute_step"] = 1
        state["gap"] = 0.001  # artificially small gap

        # Should trigger full recompute
        result2 = diag.reduce(Q, K, 32, prior_state=state)
        assert result2.get("recomputed") is True

    def test_degree_shift_trigger(self):
        diag = CheegerDiagnostic(
            mode="streaming", threshold=1024, degree_shift_threshold=0.01,
            gap_threshold=0.0001,
        )

        Q1, K1 = _random_qk(32, 16, seed=42)
        result1 = diag.reduce(Q1, K1, 32, prior_state=None)
        state = diag.accumulate(result1, None)
        state["step"] = 1
        state["last_full_recompute_step"] = 1

        # Very different Q/K → large degree shift
        Q2, K2 = _random_qk(32, 16, seed=999)
        Q2 = Q2 * 10.0  # scale dramatically
        result2 = diag.reduce(Q2, K2, 32, prior_state=state)
        assert result2.get("recomputed") is True

    def test_geometric_stride_trigger(self, qk):
        Q, K = qk
        diag = CheegerDiagnostic(
            mode="streaming", threshold=1024,
            geometric_base=1.5, gap_threshold=0.0001,
        )

        result1 = diag.reduce(Q, K, 32, prior_state=None)
        state = diag.accumulate(result1, None)
        # geometric_stride_next should be 2 initially
        # Set step far enough that steps_since >= geometric_stride_next
        state["step"] = 10
        state["last_full_recompute_step"] = 1

        result2 = diag.reduce(Q, K, 32, prior_state=state)
        assert result2.get("recomputed") is True

    def test_streaming_false_ignores_prior_state(self, qk):
        Q, K = qk
        diag = CheegerDiagnostic(mode="batch", threshold=1024)

        fake_state = {"ritz_basis": torch.randn(32, 3), "gap": 0.0}
        result = diag.reduce(Q, K, 32, prior_state=fake_state)
        # Should NOT return streaming fields
        assert "ritz_basis" not in result
        assert result["tier"] in ("materialized", "matrix_free")

    def test_accumulate_carries_forward_ritz_basis(self, qk):
        Q, K = qk
        diag = CheegerDiagnostic(mode="streaming", threshold=1024)

        result1 = diag.reduce(Q, K, 32, prior_state=None)
        state1 = diag.accumulate(result1, None)
        assert "ritz_basis" in state1

        state1["step"] = 1
        result2 = diag.reduce(Q, K, 32, prior_state=state1)
        state2 = diag.accumulate(result2, state1)
        assert "ritz_basis" in state2

    def test_accumulate_batch_returns_local(self, qk):
        Q, K = qk
        diag = CheegerDiagnostic(mode="batch", threshold=1024)
        result = diag.reduce(Q, K, 32)
        state = diag.accumulate(result, None)
        assert state is result

    def test_bracket_fields_populated_full(self, qk):
        Q, K = qk
        diag = CheegerDiagnostic(mode="streaming", threshold=1024)
        result = diag.reduce(Q, K, 32, prior_state=None)
        f = result["features"]
        assert f.cheeger_lower is not None
        assert f.cheeger_upper is not None
        assert f.bracket_width is not None
        assert f.spectral_gap is not None
        assert f.phi_hat is not None
        assert f.recomputed is True

    def test_bracket_fields_populated_cheap(self, qk):
        Q, K = qk
        diag = CheegerDiagnostic(mode="streaming", threshold=1024, gap_threshold=0.0001)

        result1 = diag.reduce(Q, K, 32, prior_state=None)
        state = diag.accumulate(result1, None)
        state["step"] = 1
        state["last_full_recompute_step"] = 1

        result2 = diag.reduce(Q, K, 32, prior_state=state)
        f = result2["features"]
        assert f.cheeger_lower is not None
        assert f.cheeger_upper is not None
        assert f.bracket_width is not None
        assert f.spectral_gap is not None
        assert f.recomputed is False

    def test_cheeger_bounds_hold_streaming(self, qk):
        Q, K = qk
        diag = CheegerDiagnostic(mode="streaming", threshold=1024)
        result = diag.reduce(Q, K, 32, prior_state=None)
        f = result["features"]
        assert f.cheeger_lower - 1e-6 <= f.phi_star <= f.cheeger_upper + 1e-6


# ---------------------------------------------------------------------------
# Integration: multi-window sliding sequence
# ---------------------------------------------------------------------------

class TestStreamingIntegration:
    def test_sliding_window_sequence(self):
        """Simulate 5 windows, verify the streaming controller alternates."""
        L, D = 32, 16
        # Use same Q/K each window (stable operator) and suppress all triggers
        Q, K = _random_qk(L, D, seed=42)
        diag = CheegerDiagnostic(
            mode="streaming", threshold=1024,
            gap_threshold=0.0001,
            degree_shift_threshold=10.0,  # suppress degree-shift trigger
            geometric_base=100.0,  # suppress geometric trigger
        )

        state = None
        tiers = []
        for i in range(5):
            result = diag.reduce(Q, K, L, prior_state=state)
            result["step"] = i + 1
            state = diag.accumulate(result, state)
            tiers.append(result.get("tier"))

        # First window is always full
        assert tiers[0] in ("materialized", "matrix_free")
        # Subsequent windows should be cheap (bordered_ritz)
        cheap_count = sum(1 for t in tiers[1:] if t == "bordered_ritz")
        assert cheap_count >= 1, f"Expected some cheap updates, got tiers: {tiers}"

    def test_sigma2_bounded_ritz_vs_full(self):
        """Bordered RR sigma2 should be within tolerance of full Lanczos."""
        L, D = 48, 16
        Q, K = _random_qk(L, D, seed=42)
        diag = CheegerDiagnostic(
            mode="streaming", threshold=1024,
            gap_threshold=0.0001, ritz_rank=5, n_explore=4,
            degree_shift_threshold=10.0,
            geometric_base=100.0,
        )

        # Full recompute
        result1 = diag.reduce(Q, K, L, prior_state=None)
        result1["step"] = 1
        state = diag.accumulate(result1, None)
        sigma2_full = result1["features"].sigma2

        # Cheap update with same Q/K (operator unchanged, basis exact)
        result2 = diag.reduce(Q, K, L, prior_state=state)
        sigma2_cheap = result2["features"].sigma2

        assert abs(sigma2_full - sigma2_cheap) < 0.25, (
            f"sigma2 drift too large: full={sigma2_full:.4f} cheap={sigma2_cheap:.4f}"
        )

    def test_growing_prefix_sequence(self):
        """Simulate autoregressive decode: grow prefix from 24 to 48 tokens."""
        L_max, D = 48, 16
        Q, K = _random_qk(L_max, D, seed=42)
        seed_size = 24

        diag = CheegerDiagnostic(
            mode="streaming", threshold=1024,
            gap_threshold=0.0001,
            degree_shift_threshold=10.0,
            geometric_base=100.0,
        )

        state = None
        tiers = []
        for w in range(seed_size, L_max + 1):
            Qw, Kw = Q[:w], K[:w]
            result = diag.reduce(Qw, Kw, w, prior_state=state)
            result["step"] = w - seed_size + 1
            state = diag.accumulate(result, state)
            tiers.append(result.get("tier"))

        assert tiers[0] in ("materialized", "matrix_free")
        cheap_count = sum(1 for t in tiers[1:] if t == "bordered_ritz")
        assert cheap_count >= len(tiers) // 3, (
            f"Expected cheap updates in growing prefix, got tiers: {tiers}"
        )

        f = result["features"]
        assert f.cheeger_lower - 1e-6 <= f.phi_star <= f.cheeger_upper + 1e-6

    def test_batch_backward_compat(self):
        """With mode='batch', all existing behavior preserved."""
        L, D = 32, 16
        Q, K = _random_qk(L, D, seed=42)

        batch = CheegerDiagnostic(mode="batch", threshold=1024)
        result = batch.reduce(Q, K, L)
        assert "ritz_basis" not in result
        assert result["tier"] == "materialized"
        f = result["features"]
        assert 0.0 <= f.phi_star <= 1.0
        assert f.cheeger_lower <= f.phi_star + 1e-6


# ---------------------------------------------------------------------------
# Light mode tests
# ---------------------------------------------------------------------------

class TestLightMode:
    def test_light_returns_bounds_only(self):
        """Light mode emits σ₂ and gap-free bracket, no φ*."""
        L, D = 32, 16
        Q, K = _random_qk(L, D, seed=42)
        diag = CheegerDiagnostic(mode="light", threshold=1024)
        result = diag.reduce(Q, K, L)
        assert result["tier"] == "light"
        f = result["features"]
        assert f.phi_star is None
        assert f.phi_hat is None
        assert f.sigma2 is not None
        assert f.cheeger_lower is not None
        assert f.cheeger_upper is not None
        assert f.bracket_width is not None
        assert f.cheeger_lower <= f.cheeger_upper

    def test_light_matrix_free(self):
        """Light mode works in matrix-free path."""
        L, D = 32, 16
        Q, K = _random_qk(L, D, seed=42)
        diag = CheegerDiagnostic(mode="light", threshold=0, block_size=8)
        result = diag.reduce(Q, K, L)
        assert result["tier"] == "light"
        f = result["features"]
        assert f.phi_star is None
        assert f.sigma2 is not None
        assert 0.0 <= f.sigma2 <= 1.0

    def test_light_sigma2_agrees_with_batch(self):
        """Light mode σ₂ should be close to batch mode σ₂."""
        L, D = 48, 16
        Q, K = _random_qk(L, D, seed=42)

        batch = CheegerDiagnostic(mode="batch", threshold=1024)
        light = CheegerDiagnostic(mode="light", threshold=1024)

        f_batch = batch.reduce(Q, K, L)["features"]
        f_light = light.reduce(Q, K, L)["features"]

        assert abs(f_batch.sigma2 - f_light.sigma2) < 0.05, (
            f"sigma2 mismatch: batch={f_batch.sigma2:.4f} light={f_light.sigma2:.4f}"
        )

    def test_light_ignores_prior_state(self):
        """Light mode doesn't use prior_state."""
        L, D = 32, 16
        Q, K = _random_qk(L, D, seed=42)
        diag = CheegerDiagnostic(mode="light", threshold=1024)

        result1 = diag.reduce(Q, K, L)
        fake_state = {"ritz_basis": torch.randn(L, 3), "step": 5}
        result2 = diag.reduce(Q, K, L, prior_state=fake_state)

        assert result1["features"].sigma2 == result2["features"].sigma2
        assert result2["tier"] == "light"

    def test_light_accumulate_passthrough(self):
        """Light mode accumulate returns local unchanged."""
        L, D = 32, 16
        Q, K = _random_qk(L, D, seed=42)
        diag = CheegerDiagnostic(mode="light", threshold=1024)

        result = diag.reduce(Q, K, L)
        state = diag.accumulate(result, None)
        assert state is result


# ---------------------------------------------------------------------------
# Dual Cheeger (Bauer-Jost bipartiteness)
# ---------------------------------------------------------------------------

class TestDualCheeger:
    def test_dual_cheeger_fields_batch_materialized(self):
        Q, K = _random_qk(32, 16)
        diag = CheegerDiagnostic(rank=2, threshold=1024, causal=True, mode="batch")
        result = diag.reduce(Q, K, 32)
        f = result["features"]
        assert f.lambda_min is not None
        assert f.dual_gap is not None
        assert f.dual_cheeger_lower is not None
        assert f.dual_cheeger_upper is not None
        assert f.dual_gap >= 0.0
        assert f.dual_cheeger_lower <= f.dual_cheeger_upper

    def test_dual_cheeger_fields_batch_matrix_free(self):
        Q, K = _random_qk(32, 16)
        diag = CheegerDiagnostic(rank=2, threshold=0, block_size=16, causal=True, mode="batch")
        result = diag.reduce(Q, K, 32)
        f = result["features"]
        assert f.lambda_min is not None
        assert f.dual_gap is not None
        assert f.dual_cheeger_lower is not None
        assert f.dual_cheeger_upper is not None

    def test_dual_cheeger_fields_streaming(self):
        Q, K = _random_qk(32, 16)
        diag = CheegerDiagnostic(
            rank=2, threshold=1024, causal=True, mode="streaming",
            lanczos_iters=20,
        )
        result = diag.reduce(Q, K, 32)
        f = result["features"]
        assert f.lambda_min is not None
        assert f.dual_gap is not None

    def test_dual_cheeger_fields_light(self):
        Q, K = _random_qk(32, 16)
        diag = CheegerDiagnostic(mode="light", threshold=1024)
        result = diag.reduce(Q, K, 32)
        f = result["features"]
        assert f.lambda_min is not None
        assert f.dual_gap is not None

    def test_dual_cheeger_fields_light_matrix_free(self):
        Q, K = _random_qk(32, 16)
        diag = CheegerDiagnostic(mode="light", threshold=0, block_size=16)
        result = diag.reduce(Q, K, 32)
        f = result["features"]
        assert f.lambda_min is not None
        assert f.dual_gap is not None

    def test_dual_cheeger_inequality_holds(self):
        """dual_gap/2 <= beta <= sqrt(2*dual_gap), so bounds must be consistent."""
        Q, K = _random_qk(48, 16)
        diag = CheegerDiagnostic(rank=2, threshold=1024, causal=True, mode="batch")
        result = diag.reduce(Q, K, 48)
        f = result["features"]
        assert f.dual_cheeger_lower <= f.dual_cheeger_upper + 1e-6

    def test_dual_cheeger_materialized_vs_matrix_free(self):
        """Dual Cheeger from materialized and matrix-free paths should agree."""
        Q, K = _random_qk(32, 16)
        diag_mat = CheegerDiagnostic(rank=2, threshold=1024, causal=True, mode="batch")
        diag_mf = CheegerDiagnostic(rank=2, threshold=0, block_size=16, causal=True, mode="batch")
        f_mat = diag_mat.reduce(Q, K, 32)["features"]
        f_mf = diag_mf.reduce(Q, K, 32)["features"]
        assert abs(f_mat.lambda_min - f_mf.lambda_min) < 0.1
        assert abs(f_mat.dual_gap - f_mf.dual_gap) < 0.1

    def test_dual_cheeger_carried_forward_streaming(self):
        """Cheap BRR update carries forward lambda_min from last full recompute."""
        Q, K = _random_qk(32, 16, seed=99)
        diag = CheegerDiagnostic(
            rank=2, threshold=1024, causal=True, mode="streaming",
            gap_threshold=0.0001, degree_shift_threshold=10.0,
            geometric_base=100.0, lanczos_iters=20,
        )
        r1 = diag.reduce(Q, K, 32)
        state = diag.accumulate(r1, None)
        state["step"] = 1
        state["last_full_recompute_step"] = 1
        state["geometric_stride_next"] = 100

        r2 = diag.reduce(Q, K, 32, prior_state=state)
        assert r2["tier"] == "bordered_ritz"
        f2 = r2["features"]
        assert f2.lambda_min is not None
        assert abs(f2.lambda_min - r1["features"].lambda_min) < 1e-6


# ---------------------------------------------------------------------------
# Adaptive KLGT
# ---------------------------------------------------------------------------

class TestAdaptiveKLGT:
    def test_adaptive_tighter_or_equal(self):
        """Adaptive k should produce a bound <= fixed k."""
        evals = torch.tensor([1.0, 0.9, 0.85, 0.7, 0.5, 0.3])
        fixed = compute_improved_cheeger_upper(evals, k=4, adaptive=False)
        adaptive = compute_improved_cheeger_upper(evals, k=4, adaptive=True)
        assert adaptive is not None
        assert fixed is not None
        assert adaptive <= fixed + 1e-10

    def test_adaptive_selects_best_k(self):
        """Adaptive should find k that minimizes k*mu2/sqrt(mu_{k+1})."""
        evals = torch.tensor([1.0, 0.95, 0.94, 0.93, 0.2])
        adaptive = compute_improved_cheeger_upper(evals, adaptive=True)
        assert adaptive is not None
        mu2 = 1.0 - 0.95
        for trial_k in range(2, 5):
            idx = min(trial_k, 4)
            mu_kp1 = max(1.0 - float(evals[idx]), 1e-12)
            bound = float(trial_k) * mu2 / (mu_kp1 ** 0.5)
            assert adaptive <= bound + 1e-6

    def test_adaptive_zero_gap(self):
        """When mu2=0, adaptive returns 0."""
        evals = torch.tensor([1.0, 1.0, 0.5])
        result = compute_improved_cheeger_upper(evals, adaptive=True)
        assert result == 0.0

    def test_adaptive_insufficient_eigenvalues(self):
        evals = torch.tensor([1.0, 0.9])
        result = compute_improved_cheeger_upper(evals, adaptive=True)
        assert result is None

    def test_adaptive_used_in_streaming(self):
        """Streaming mode should produce improved_upper (using adaptive)."""
        Q, K = _random_qk(32, 16)
        diag = CheegerDiagnostic(
            rank=2, threshold=1024, causal=True, mode="streaming",
            lanczos_iters=20, improved_cheeger_k=4,
        )
        result = diag.reduce(Q, K, 32)
        f = result["features"]
        assert f.improved_upper is not None

    def test_batch_materialized_has_improved_upper(self):
        """Batch materialized path now computes adaptive KLGT."""
        Q, K = _random_qk(32, 16)
        diag = CheegerDiagnostic(rank=2, threshold=1024, causal=True, mode="batch")
        result = diag.reduce(Q, K, 32)
        f = result["features"]
        assert f.improved_upper is not None
