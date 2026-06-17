import math

import pytest
import torch

from glassbox.svd import (
    apply_A_blocked,
    apply_AT_blocked,
    compare_svd_results,
    compute_degree_normalized_M,
    compute_dk_blocked,
    compute_logsumexp_blocked,
    compute_M_fro_norm_blocked,
    compute_scores_matrix_features,
    get_M_entries_batch,
    hermitian_lanczos,
    matvec_M_blocked,
    matvec_MT_blocked,
    matvec_S,
    matvec_ST,
    randomized_svd,
    svd_via_lanczos,
)

L = 8
D = 2


@pytest.fixture
def qk():
    torch.manual_seed(42)
    Q = torch.randn(L, D)
    K = torch.randn(L, D)

    def matvec(x):
        return matvec_S(Q, K, x)

    def matvec_t(x):
        return matvec_ST(Q, K, x)

    return Q, K, matvec, matvec_t


def test_randomized_svd_vs_lanczos(qk):
    Q, K, matvec, matvec_t = qk

    U_l, S_l, V_l = svd_via_lanczos(
        matvec=matvec, matvec_t=matvec_t, dim=L, k=2, iters=20, device="cpu"
    )
    U_r, S_r, V_r = randomized_svd(
        matvec=matvec, matvec_t=matvec_t, dim=L, k=2, p=6, q=2, device="cpu"
    )

    metrics = compare_svd_results(
        matvec=matvec,
        matvec_t=matvec_t,
        U1=U_l,
        S1=S_l,
        V1=V_l,
        U2=U_r,
        S2=S_r,
        V2=V_r,
    )

    assert metrics["sv_rel_max"] < 0.1
    assert metrics["ang_U_max_deg"] < 5.0
    assert metrics["ang_V_max_deg"] < 5.0
    assert metrics["mv_res_max"] < 0.1
    assert metrics["mtu_res_max"] < 0.1
    assert metrics["recon_M_minus_USVt_max"] < 0.1


def test_svd_vs_torch(qk):
    Q, K, matvec, matvec_t = qk
    S_full = Q @ K.T

    _, S_torch, _ = torch.linalg.svd(S_full, full_matrices=False)
    S_torch_top2 = S_torch[:2]

    U_l, S_l, V_l = svd_via_lanczos(
        matvec=matvec, matvec_t=matvec_t, dim=L, k=2, iters=20, device="cpu"
    )
    U_r, S_r, V_r = randomized_svd(
        matvec=matvec, matvec_t=matvec_t, dim=L, k=2, p=6, q=2, device="cpu"
    )

    S_l_sorted, _ = torch.sort(S_l, descending=True)
    S_r_sorted, _ = torch.sort(S_r, descending=True)

    torch.testing.assert_close(S_l_sorted, S_torch_top2, atol=1e-4, rtol=1e-3)
    torch.testing.assert_close(S_r_sorted, S_torch_top2, atol=1e-4, rtol=1e-3)


# --- Blocked matvec tests ---

L_BLOCK = 16
D_BLOCK = 4


@pytest.fixture
def qk_block():
    torch.manual_seed(123)
    Q = torch.randn(L_BLOCK, D_BLOCK)
    K = torch.randn(L_BLOCK, D_BLOCK)
    scale = 1.0 / math.sqrt(D_BLOCK)
    return Q, K, scale


def _ref_A(Q, K, scale):
    return torch.softmax(Q @ K.T * scale, dim=-1)


def test_apply_A_blocked(qk_block):
    Q, K, scale = qk_block
    v = torch.randn(L_BLOCK)
    A = _ref_A(Q, K, scale)
    expected = A @ v
    result = apply_A_blocked(Q, K, v, scale, block_size=4)
    torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)


def test_apply_AT_blocked(qk_block):
    Q, K, scale = qk_block
    u = torch.randn(L_BLOCK)
    A = _ref_A(Q, K, scale)
    expected = A.T @ u
    result = apply_AT_blocked(Q, K, u, scale, block_size=4)
    torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)


def test_compute_dk_blocked(qk_block):
    Q, K, scale = qk_block
    A = _ref_A(Q, K, scale)
    expected_dk = A.sum(dim=0)
    dk, dk_inv_sqrt = compute_dk_blocked(Q, K, scale, block_size=4)
    torch.testing.assert_close(dk, expected_dk, atol=1e-5, rtol=1e-5)


def test_logsumexp_blocked(qk_block):
    Q, K, scale = qk_block
    expected = torch.logsumexp(Q @ K.T * scale, dim=-1)
    result = compute_logsumexp_blocked(Q, K, scale, block_size=4)
    torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)


def test_get_M_entries_batch(qk_block):
    Q, K, scale = qk_block
    A = _ref_A(Q, K, scale)
    M, _, d_k_inv_sqrt = compute_degree_normalized_M(A)
    lse = compute_logsumexp_blocked(Q, K, scale)

    # Test random entries
    torch.manual_seed(99)
    ii = torch.randint(0, L_BLOCK, (20,))
    jj = torch.randint(0, L_BLOCK, (20,))
    result = get_M_entries_batch(Q, K, lse, d_k_inv_sqrt, scale, ii, jj)
    expected = M[ii, jj]
    torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-4)


def test_matvec_M_blocked(qk_block):
    Q, K, scale = qk_block
    A = _ref_A(Q, K, scale)
    M, _, d_k_inv_sqrt = compute_degree_normalized_M(A)
    v = torch.randn(L_BLOCK)
    expected = M @ v
    result = matvec_M_blocked(Q, K, v, d_k_inv_sqrt, scale, block_size=4)
    torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-4)


def test_M_fro_norm_blocked(qk_block):
    Q, K, scale = qk_block
    A = _ref_A(Q, K, scale)
    M, _, d_k_inv_sqrt = compute_degree_normalized_M(A)
    expected = torch.linalg.norm(M, "fro")
    result = compute_M_fro_norm_blocked(Q, K, d_k_inv_sqrt, scale, block_size=4)
    torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-4)


def test_svd_M_randomized_vs_direct():
    """Compare randomized SVD of M via matvecs against torch.linalg.svdvals(M)."""
    torch.manual_seed(77)
    L_test = 32
    D_test = 8
    Q = torch.randn(L_test, D_test)
    K = torch.randn(L_test, D_test)
    scale = 1.0 / math.sqrt(D_test)

    A = _ref_A(Q, K, scale)
    M, _, d_k_inv_sqrt = compute_degree_normalized_M(A)
    sigma_ref = torch.linalg.svdvals(M)[:4]

    def matvec(v):
        return matvec_M_blocked(Q, K, v, d_k_inv_sqrt, scale)

    def matvec_t(u):
        return matvec_MT_blocked(Q, K, u, d_k_inv_sqrt, scale)

    _, S_rand, _ = randomized_svd(matvec, matvec_t, L_test, 4, device="cpu")

    S_sorted, _ = torch.sort(S_rand, descending=True)
    rel_err = (S_sorted - sigma_ref).abs() / (sigma_ref + 1e-12)
    assert rel_err.max().item() < 0.01, f"Relative error too high: {rel_err}"


def test_two_tier_agreement():
    """Both materialized and matrix-free paths should give similar singular values."""
    torch.manual_seed(55)
    L_test = 64
    D_test = 8
    Q = torch.randn(L_test, D_test)
    K = torch.randn(L_test, D_test)
    scale = 1.0 / math.sqrt(D_test)

    # Materialized
    A = _ref_A(Q, K, scale)
    M, _, d_k_inv_sqrt = compute_degree_normalized_M(A)
    sigma_mat = torch.linalg.svdvals(M)[:4]

    # Matrix-free
    _, d_k_inv_sqrt_mf = compute_dk_blocked(Q, K, scale)

    def matvec(v):
        return matvec_M_blocked(Q, K, v, d_k_inv_sqrt_mf, scale)

    def matvec_t(u):
        return matvec_MT_blocked(Q, K, u, d_k_inv_sqrt_mf, scale)

    _, S_mf, _ = randomized_svd(matvec, matvec_t, L_test, 4, device="cpu")
    S_mf_sorted, _ = torch.sort(S_mf, descending=True)

    rel_err = (S_mf_sorted - sigma_mat).abs() / (sigma_mat + 1e-12)
    assert rel_err.max().item() < 0.01, f"Two-tier disagreement: {rel_err}"


# --- compute_scores_matrix_features end-to-end tests ---


def test_compute_scores_matrix_features_vs_torch():
    """compute_scores_matrix_features should match torch.linalg.svd on S=QK^T."""
    torch.manual_seed(42)
    L_test = 32
    D_test = 8
    Q = torch.randn(L_test, D_test)
    K = torch.randn(L_test, D_test)

    S_full = Q @ K.T
    sigma_ref = torch.linalg.svdvals(S_full)[:4].tolist()

    features = compute_scores_matrix_features(Q, K, rank=4)

    assert len(features.singular_values) == 4
    for sv_feat, sv_ref in zip(features.singular_values, sigma_ref):
        assert abs(sv_feat - sv_ref) < 0.05, f"sv mismatch: {sv_feat} vs {sv_ref}"

    assert features.sv1 == features.singular_values[0]
    assert features.sv_ratio is not None
    assert abs(features.sv_ratio - sigma_ref[0] / sigma_ref[1]) < 0.1
    assert features.sv_entropy is not None
    assert features.sv_entropy > 0


def test_compute_scores_matrix_features_lanczos_vs_randomized():
    """Both SVD methods should approximate the true singular values well."""
    torch.manual_seed(99)
    L_test = 32
    D_test = 8
    Q = torch.randn(L_test, D_test)
    K = torch.randn(L_test, D_test)

    # Ground truth via exact SVD
    S = Q @ K.T
    sigma_exact = torch.linalg.svdvals(S)[:4].tolist()

    f_rand = compute_scores_matrix_features(Q, K, rank=4, method="randomized")
    f_lanc = compute_scores_matrix_features(Q, K, rank=4, method="lanczos")

    # Each method should be within 1% relative error of the exact values
    for method_name, f in [("randomized", f_rand), ("lanczos", f_lanc)]:
        for sv_approx, sv_true in zip(f.singular_values, sigma_exact):
            rel_err = abs(sv_approx - sv_true) / max(abs(sv_true), 1e-6)
            assert rel_err < 0.01, (
                f"{method_name} sv mismatch: got {sv_approx}, expected {sv_true} "
                f"(rel_err={rel_err:.3f})"
            )


# ---------------------------------------------------------------------------
# hermitian_lanczos tests
# ---------------------------------------------------------------------------

N_HERM = 16
K_HERM = 4
ITERS_HERM = 30

COMPLEX_DTYPES = [torch.complex64, torch.complex128]
ALL_HERMITIAN_DTYPES = [
    torch.float32,
    torch.float16,
    torch.bfloat16,
    torch.complex64,
    torch.complex128,
]
HERMITIAN_EIGVAL_ATOL = {
    torch.float32: 1e-3,
    torch.float16: 0.15,
    torch.bfloat16: 0.15,
    torch.complex64: 1e-3,
    torch.complex128: 1e-6,
}
HERMITIAN_DTYPE_IDS = {
    torch.float32: "fp32",
    torch.float16: "fp16",
    torch.bfloat16: "bf16",
    torch.complex64: "c64",
    torch.complex128: "c128",
}


def _random_hermitian(n, dtype, device="cpu", seed=42):
    torch.manual_seed(seed)
    R = torch.randn(n, n, dtype=dtype, device=device)
    return (R + R.conj().T) / 2


def _ref_eigh(H, k, which, real_dtype=torch.float32):
    evals, _ = torch.linalg.eigh(H.to(torch.complex128 if H.is_complex() else torch.float64))
    if which == "smallest":
        return evals[:k].to(real_dtype)
    return evals[-k:].flip(0).to(real_dtype)


class TestHermitianLanczos:
    def test_real_largest_vs_eigh(self):
        H = _random_hermitian(N_HERM, torch.float32)
        ref = _ref_eigh(H, K_HERM, "largest")
        evals, evecs = hermitian_lanczos(
            lambda v: H @ v, N_HERM, K_HERM, ITERS_HERM, "cpu", which="largest"
        )
        torch.testing.assert_close(evals, ref, atol=1e-3, rtol=0.01)
        assert evecs.shape == (N_HERM, K_HERM)

    def test_real_smallest_vs_eigh(self):
        H = _random_hermitian(N_HERM, torch.float32)
        ref = _ref_eigh(H, K_HERM, "smallest")
        evals, evecs = hermitian_lanczos(
            lambda v: H @ v, N_HERM, K_HERM, ITERS_HERM, "cpu", which="smallest"
        )
        torch.testing.assert_close(evals, ref, atol=1e-3, rtol=0.01)
        assert evecs.shape == (N_HERM, K_HERM)

    def test_complex64_largest(self):
        H = _random_hermitian(N_HERM, torch.complex64)
        ref = _ref_eigh(H, K_HERM, "largest")
        evals, evecs = hermitian_lanczos(
            lambda v: H @ v,
            N_HERM,
            K_HERM,
            ITERS_HERM,
            "cpu",
            which="largest",
            dtype=torch.complex64,
        )
        torch.testing.assert_close(evals, ref, atol=1e-3, rtol=0.01)
        assert evecs.dtype == torch.complex64

    def test_complex64_smallest(self):
        H = _random_hermitian(N_HERM, torch.complex64)
        ref = _ref_eigh(H, K_HERM, "smallest")
        evals, evecs = hermitian_lanczos(
            lambda v: H @ v,
            N_HERM,
            K_HERM,
            ITERS_HERM,
            "cpu",
            which="smallest",
            dtype=torch.complex64,
        )
        torch.testing.assert_close(evals, ref, atol=1e-3, rtol=0.01)

    def test_complex128(self):
        H = _random_hermitian(N_HERM, torch.complex128)
        ref_lg = _ref_eigh(H, K_HERM, "largest", real_dtype=torch.float64)
        ref_sm = _ref_eigh(H, K_HERM, "smallest", real_dtype=torch.float64)
        evals_lg, _ = hermitian_lanczos(
            lambda v: H @ v,
            N_HERM,
            K_HERM,
            ITERS_HERM,
            "cpu",
            which="largest",
            dtype=torch.complex128,
        )
        evals_sm, _ = hermitian_lanczos(
            lambda v: H @ v,
            N_HERM,
            K_HERM,
            ITERS_HERM,
            "cpu",
            which="smallest",
            dtype=torch.complex128,
        )
        torch.testing.assert_close(evals_lg, ref_lg, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(evals_sm, ref_sm, atol=1e-6, rtol=1e-5)

    @pytest.mark.parametrize(
        "dtype",
        [torch.float16, torch.bfloat16],
        ids=lambda d: HERMITIAN_DTYPE_IDS[d],
    )
    def test_fp16_bf16_stability(self, dtype):
        H_fp32 = _random_hermitian(N_HERM, torch.float32)
        ref = _ref_eigh(H_fp32, K_HERM, "largest")
        H_half = H_fp32.to(dtype)
        evals, evecs = hermitian_lanczos(
            lambda v: H_half @ v,
            N_HERM,
            K_HERM,
            ITERS_HERM,
            "cpu",
            which="largest",
            dtype=dtype,
        )
        assert evecs.dtype == dtype
        torch.testing.assert_close(evals, ref, atol=0.15, rtol=0.15)

    def test_k_zero(self):
        evals, evecs = hermitian_lanczos(
            lambda v: v,
            8,
            0,
            10,
            "cpu",
        )
        assert evals.shape == (0,)
        assert evecs.shape == (8, 0)

    def test_which_invalid(self):
        with pytest.raises(ValueError, match="smallest.*largest"):
            hermitian_lanczos(lambda v: v, 8, 4, 10, "cpu", which="middle")

    def test_eigenvectors_orthonormal(self):
        H = _random_hermitian(N_HERM, torch.complex64)
        _, evecs = hermitian_lanczos(
            lambda v: H @ v,
            N_HERM,
            K_HERM,
            ITERS_HERM,
            "cpu",
            dtype=torch.complex64,
        )
        gram = evecs.conj().T @ evecs
        eye = torch.eye(K_HERM, dtype=gram.dtype)
        torch.testing.assert_close(gram, eye, atol=1e-4, rtol=0)

    def test_residual(self):
        H = _random_hermitian(N_HERM, torch.float32)
        evals, evecs = hermitian_lanczos(
            lambda v: H @ v,
            N_HERM,
            K_HERM,
            ITERS_HERM,
            "cpu",
        )
        for i in range(K_HERM):
            v = evecs[:, i]
            residual = torch.linalg.norm(H @ v - evals[i] * v)
            assert residual < 1e-3, f"Residual {residual} too large for eigenpair {i}"

    def test_early_termination(self):
        v = torch.randn(N_HERM)
        v = v / torch.linalg.norm(v)
        H = 3.0 * v.outer(v)
        evals, evecs = hermitian_lanczos(
            lambda x: H @ x,
            N_HERM,
            1,
            ITERS_HERM,
            "cpu",
            which="largest",
        )
        torch.testing.assert_close(evals[0], torch.tensor(3.0), atol=1e-4, rtol=0)

    @pytest.mark.parametrize(
        "dtype",
        ALL_HERMITIAN_DTYPES,
        ids=lambda d: HERMITIAN_DTYPE_IDS[d],
    )
    def test_dtype_propagation(self, dtype):
        H = _random_hermitian(N_HERM, dtype)
        evals, evecs = hermitian_lanczos(
            lambda v: H @ v,
            N_HERM,
            K_HERM,
            ITERS_HERM,
            "cpu",
            dtype=dtype,
        )
        assert evecs.dtype == dtype
        if dtype in (torch.complex128, torch.float64):
            assert evals.dtype == torch.float64
        else:
            assert evals.dtype == torch.float32
