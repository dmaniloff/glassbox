import math

import pytest
import torch

from glassbox.svd import (
    apply_A_blocked,
    apply_AT_blocked,
    compute_degree_normalized_M,
    compute_dk_blocked,
    compute_logsumexp_blocked,
    compute_M_fro_norm_blocked,
    compute_scores_matrix_features,
    get_M_entries_batch,
    matvec_M_blocked,
    matvec_MT_blocked,
    matvec_S,
    matvec_ST,
    randomized_svd,
    svd_via_lanczos,
)

ALL_DTYPES = [torch.float32, torch.float16, torch.bfloat16]
HALF_DTYPES = [torch.float16, torch.bfloat16]
DTYPE_SV_ATOL = {torch.float32: 1e-3, torch.float16: 0.15, torch.bfloat16: 0.15}
DTYPE_IDS = {torch.float32: "fp32", torch.float16: "fp16", torch.bfloat16: "bf16"}

L = 8
D = 2


# ---------------------------------------------------------------------------
# SVD comparison helpers (test-only diagnostics; not used in glassbox runtime)
# ---------------------------------------------------------------------------


def _principal_angles(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Principal angles between column spaces of A and B.

    Both inputs are (dim, k), assumed approximately orthonormal.
    Returns angles in radians, length k, sorted ascending.
    """
    # QR/svdvals require float32 (LAPACK does not support fp16/bf16).
    QA, _ = torch.linalg.qr(A.float(), mode="reduced")
    QB, _ = torch.linalg.qr(B.float(), mode="reduced")
    # Singular values of QA^T QB are cosines of principal angles.
    s = torch.linalg.svdvals(QA.T @ QB).clamp(-1.0, 1.0)
    return torch.acos(s)


def compare_svd_results(matvec, matvec_t, U1, S1, V1, U2, S2, V2, trials: int = 8):
    """Compare two (U,S,V) factorizations via batched operator calls.

    Returns a dict of agreement metrics, or ``None`` when there are no
    singular triplets to compare (``k == 0``) — empty factorizations have no
    meaningful discrepancy, and a zero-filled dict would misleadingly read as
    perfect agreement.
    """
    device = S1.device
    k = min(S1.numel(), S2.numel())

    if k == 0:
        return None

    S1 = S1[:k]
    S2 = S2[:k]
    U1 = U1[:, :k]
    U2 = U2[:, :k]
    V1 = V1[:, :k]
    V2 = V2[:, :k]

    # singular values (order them descending before comparing)
    S1s, _ = torch.sort(S1, descending=True)
    S2s, _ = torch.sort(S2, descending=True)
    sv_abs = (S1s - S2s).abs()
    sv_rel = sv_abs / torch.clamp(torch.max(S1s.abs(), S2s.abs()), min=1e-12)

    # subspace alignment
    ang_U = _principal_angles(U1, U2)
    ang_V = _principal_angles(V1, V2)

    # Cast S to native dtype so mixed arithmetic with U/V doesn't error.
    native = V1.dtype
    S1n = S1s.to(native)
    S2n = S2s.to(native)

    # Batched residuals: 2 matvec calls instead of 2k
    MV2 = matvec(V2)
    MTU2 = matvec_t(U2)
    s_denom = torch.max(S1s, S2s).clamp(min=1e-12)
    mv_res = torch.linalg.norm(MV2.float() - S2s.unsqueeze(0) * U2.float(), dim=0) / s_denom
    mtu_res = torch.linalg.norm(MTU2.float() - S2s.unsqueeze(0) * V2.float(), dim=0) / s_denom

    # Batched reconstruction: 1 matvec call instead of trials
    X = torch.randn(V1.shape[0], trials, device=device, dtype=V1.dtype)
    Y = matvec(X)
    Y1 = U1 @ (S1n.unsqueeze(1) * (V1.T @ X))
    Y2 = U2 @ (S2n.unsqueeze(1) * (V2.T @ X))
    denom = torch.linalg.norm(Y.float(), dim=0).clamp(min=1e-12)
    recon = torch.stack(
        [
            torch.linalg.norm((Y - Y1).float(), dim=0) / denom,
            torch.linalg.norm((Y - Y2).float(), dim=0) / denom,
            torch.linalg.norm((Y1 - Y2).float(), dim=0) / denom,
        ],
        dim=1,
    )  # (trials, 3)

    return {
        "k": k,
        "sv_abs_max": sv_abs.max().item(),
        "sv_rel_max": sv_rel.max().item(),
        "ang_U_max_deg": (ang_U.max() * 180.0 / torch.pi).item(),
        "ang_V_max_deg": (ang_V.max() * 180.0 / torch.pi).item(),
        "mv_res_max": mv_res.max().item(),
        "mtu_res_max": mtu_res.max().item(),
        "recon_M_minus_USVt_mean": recon[:, 0].mean().item(),
        "recon_M_minus_USVt_max": recon[:, 0].max().item(),
        "recon_method_diff_mean": recon[:, 2].mean().item(),
        "recon_method_diff_max": recon[:, 2].max().item(),
    }


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


# --- Dtype propagation tests (fp32 / fp16 / bf16) ---

L_DTYPE = 32
D_DTYPE = 8


@pytest.fixture(params=ALL_DTYPES, ids=lambda d: DTYPE_IDS[d])
def qk_dtype(request):
    dtype = request.param
    torch.manual_seed(42)
    Q = torch.randn(L_DTYPE, D_DTYPE).to(dtype)
    K = torch.randn(L_DTYPE, D_DTYPE).to(dtype)
    return Q, K, dtype


def _ref_svdvals(Q, K, k):
    """Ground-truth singular values via dense float32 SVD."""
    S = Q.float() @ K.float().T
    return torch.linalg.svdvals(S)[:k]


class TestDtypePropagation:
    def test_randomized_svd_dtype(self, qk_dtype):
        Q, K, dtype = qk_dtype

        def mv(v):
            return matvec_S(Q, K, v)

        def mv_t(u):
            return matvec_ST(Q, K, u)

        U, S, V = randomized_svd(mv, mv_t, L_DTYPE, k=4, device="cpu", dtype=dtype)
        assert U.dtype == dtype, f"U dtype {U.dtype} != {dtype}"
        assert V.dtype == dtype, f"V dtype {V.dtype} != {dtype}"
        assert S.dtype == torch.float32, f"S should be float32, got {S.dtype}"
        assert S.shape[0] == 4

        S_ref = _ref_svdvals(Q, K, 4)
        S_sorted, _ = torch.sort(S, descending=True)
        atol = DTYPE_SV_ATOL[dtype]
        torch.testing.assert_close(S_sorted, S_ref, atol=atol, rtol=0.15)

    def test_lanczos_svd_dtype(self, qk_dtype):
        Q, K, dtype = qk_dtype

        def mv(v):
            return matvec_S(Q, K, v)

        def mv_t(u):
            return matvec_ST(Q, K, u)

        U, S, V = svd_via_lanczos(mv, mv_t, L_DTYPE, k=4, iters=40, device="cpu", dtype=dtype)
        assert U.dtype == dtype, f"U dtype {U.dtype} != {dtype}"
        assert V.dtype == dtype, f"V dtype {V.dtype} != {dtype}"
        assert S.dtype == torch.float32, f"S should be float32, got {S.dtype}"

        # Lanczos via M^T M squares the condition number (issue #33),
        # so only check the leading singular value tightly.
        S_ref = _ref_svdvals(Q, K, 1)
        S_sorted, _ = torch.sort(S, descending=True)
        atol = DTYPE_SV_ATOL[dtype]
        torch.testing.assert_close(S_sorted[:1], S_ref, atol=atol, rtol=0.15)

    def test_svd_methods_agree_dtype(self, qk_dtype):
        Q, K, dtype = qk_dtype

        def mv(v):
            return matvec_S(Q, K, v)

        def mv_t(u):
            return matvec_ST(Q, K, u)

        _, S_rand, _ = randomized_svd(mv, mv_t, L_DTYPE, k=4, device="cpu", dtype=dtype)
        _, S_lanc, _ = svd_via_lanczos(mv, mv_t, L_DTYPE, k=4, iters=40, device="cpu", dtype=dtype)

        # Only compare leading singular value — Lanczos via M^T M degrades
        # trailing values due to condition-number squaring (issue #33).
        S_rand_sorted, _ = torch.sort(S_rand, descending=True)
        S_lanc_sorted, _ = torch.sort(S_lanc, descending=True)
        atol = DTYPE_SV_ATOL[dtype]
        torch.testing.assert_close(S_rand_sorted[:1], S_lanc_sorted[:1], atol=atol, rtol=0.15)

    @pytest.mark.parametrize("dtype", HALF_DTYPES, ids=lambda d: DTYPE_IDS[d])
    def test_compute_scores_matrix_features_half(self, dtype):
        torch.manual_seed(42)
        Q = torch.randn(L_DTYPE, D_DTYPE).to(dtype)
        K = torch.randn(L_DTYPE, D_DTYPE).to(dtype)

        features = compute_scores_matrix_features(Q, K, rank=4)
        assert len(features.singular_values) == 4
        assert all(sv > 0 for sv in features.singular_values)

    @pytest.mark.parametrize("dtype", HALF_DTYPES, ids=lambda d: DTYPE_IDS[d])
    def test_blocked_matvec_preserves_dtype(self, dtype):
        torch.manual_seed(42)
        Q = torch.randn(L_DTYPE, D_DTYPE).to(dtype)
        K = torch.randn(L_DTYPE, D_DTYPE).to(dtype)
        v = torch.randn(L_DTYPE, dtype=dtype)
        scale = 1.0 / math.sqrt(D_DTYPE)

        result_a = apply_A_blocked(Q, K, v, scale, block_size=8)
        result_at = apply_AT_blocked(Q, K, v, scale, block_size=8)
        assert result_a.dtype == dtype, f"apply_A_blocked returned {result_a.dtype}"
        assert result_at.dtype == dtype, f"apply_AT_blocked returned {result_at.dtype}"

    @pytest.mark.parametrize("dtype", HALF_DTYPES, ids=lambda d: DTYPE_IDS[d])
    def test_degree_normalized_matvecs_half(self, dtype):
        """matvec_M_blocked, matvec_MT_blocked, compute_M_fro_norm_blocked with half."""
        torch.manual_seed(42)
        Q = torch.randn(L_DTYPE, D_DTYPE).to(dtype)
        K = torch.randn(L_DTYPE, D_DTYPE).to(dtype)
        scale = 1.0 / math.sqrt(D_DTYPE)
        _, d_k_inv_sqrt = compute_dk_blocked(Q, K, scale, block_size=8)
        v = torch.randn(L_DTYPE, dtype=dtype)

        mv = matvec_M_blocked(Q, K, v, d_k_inv_sqrt, scale, block_size=8)
        mvt = matvec_MT_blocked(Q, K, v, d_k_inv_sqrt, scale, block_size=8)
        fro = compute_M_fro_norm_blocked(Q, K, d_k_inv_sqrt, scale, block_size=8)
        assert mv.dtype == dtype
        assert mvt.dtype == dtype
        assert fro.dtype == dtype
        assert fro.item() > 0

    @pytest.mark.parametrize("dtype", HALF_DTYPES, ids=lambda d: DTYPE_IDS[d])
    def test_compare_svd_results_half(self, dtype):
        """compare_svd_results should not crash with half-precision inputs."""
        torch.manual_seed(42)
        Q = torch.randn(L_DTYPE, D_DTYPE).to(dtype)
        K = torch.randn(L_DTYPE, D_DTYPE).to(dtype)

        def mv(v):
            return matvec_S(Q, K, v)

        def mv_t(u):
            return matvec_ST(Q, K, u)

        U1, S1, V1 = randomized_svd(mv, mv_t, L_DTYPE, k=2, device="cpu", dtype=dtype)
        U2, S2, V2 = randomized_svd(mv, mv_t, L_DTYPE, k=2, device="cpu", dtype=dtype)
        result = compare_svd_results(mv, mv_t, U1, S1, V1, U2, S2, V2, trials=4)
        assert "sv_rel_max" in result
        assert result["sv_rel_max"] < 0.5

    def test_randomized_svd_no_power_iterations(self):
        """q=0 skips the QR re-orthogonalization loop entirely."""
        torch.manual_seed(42)
        Q = torch.randn(L_DTYPE, D_DTYPE, dtype=torch.float16)
        K = torch.randn(L_DTYPE, D_DTYPE, dtype=torch.float16)

        def mv(v):
            return matvec_S(Q, K, v)

        def mv_t(u):
            return matvec_ST(Q, K, u)

        U, S, V = randomized_svd(mv, mv_t, L_DTYPE, k=2, q=0, device="cpu", dtype=torch.float16)
        assert U.dtype == torch.float16
        assert S.shape[0] == 2

    @pytest.mark.parametrize("L_small", [2, 3])
    @pytest.mark.parametrize("dtype", HALF_DTYPES, ids=lambda d: DTYPE_IDS[d])
    def test_small_sequence_half(self, L_small, dtype):
        """Small L with half precision: k gets clamped, should not crash."""
        torch.manual_seed(42)
        Q = torch.randn(L_small, D_DTYPE).to(dtype)
        K = torch.randn(L_small, D_DTYPE).to(dtype)

        features = compute_scores_matrix_features(Q, K, rank=4)
        assert len(features.singular_values) > 0
        assert all(sv > 0 for sv in features.singular_values)

    def test_dtype_none_backward_compat(self):
        """dtype=None produces float32 outputs, matching pre-fix behavior."""
        torch.manual_seed(42)
        Q = torch.randn(L_DTYPE, D_DTYPE)
        K = torch.randn(L_DTYPE, D_DTYPE)

        def mv(v):
            return matvec_S(Q, K, v)

        def mv_t(u):
            return matvec_ST(Q, K, u)

        U, S, V = randomized_svd(mv, mv_t, L_DTYPE, k=2, device="cpu", dtype=None)
        assert U.dtype == torch.float32
        assert V.dtype == torch.float32
        assert S.dtype == torch.float32


# ---------------------------------------------------------------------------
# Small sequence length (L=0, L=1, L=2) — issue #32
# ---------------------------------------------------------------------------


class TestSmallSequenceLength:
    """SVD entry points must not crash for degenerate L values."""

    @pytest.mark.parametrize("seq_len", [0, 1], ids=["L=0", "L=1"])
    def test_randomized_svd_k_zero(self, seq_len):
        def mv(v):
            return v

        def mv_t(u):
            return u

        U, S, V = randomized_svd(mv, mv_t, seq_len, k=0, device="cpu")
        assert U.shape == (seq_len, 0)
        assert S.shape == (0,)
        assert V.shape == (seq_len, 0)

    @pytest.mark.parametrize("seq_len", [0, 1], ids=["L=0", "L=1"])
    def test_svd_via_lanczos_k_zero(self, seq_len):
        def mv(v):
            return v

        def mv_t(u):
            return u

        U, S, V = svd_via_lanczos(mv, mv_t, seq_len, k=0, iters=10, device="cpu")
        assert U.shape == (seq_len, 0)
        assert S.shape == (0,)
        assert V.shape == (seq_len, 0)

    def test_lanczos_dim_zero(self):
        from glassbox.svd import lanczos

        evals, vecs = lanczos(lambda v: v, dim=0, k=1, iters=10, device="cpu")
        assert evals.shape == (0,)
        assert vecs.shape == (0, 0)

    def test_compute_scores_matrix_features_L1(self):
        Q = torch.randn(1, 4)
        K = torch.randn(1, 4)
        feats = compute_scores_matrix_features(Q, K, rank=2)
        assert feats.singular_values == []
        assert feats.sv1 is None

    def test_compute_scores_matrix_features_L0(self):
        Q = torch.randn(0, 4)
        K = torch.randn(0, 4)
        feats = compute_scores_matrix_features(Q, K, rank=2)
        assert feats.singular_values == []

    def test_compare_svd_results_k_zero(self):
        empty_U = torch.empty(4, 0)
        empty_S = torch.empty(0)
        empty_V = torch.empty(4, 0)
        result = compare_svd_results(
            lambda v: v,
            lambda u: u,
            empty_U,
            empty_S,
            empty_V,
            empty_U,
            empty_S,
            empty_V,
        )
        assert result is None

    def test_randomized_svd_L2_k1(self):
        """L=2 is the smallest non-degenerate case — should produce 1 SV."""
        torch.manual_seed(42)
        Q = torch.randn(2, 4)
        K = torch.randn(2, 4)

        def mv(v):
            return matvec_S(Q, K, v)

        def mv_t(u):
            return matvec_ST(Q, K, u)

        U, S, V = randomized_svd(mv, mv_t, 2, k=1, device="cpu")
        assert S.shape == (1,)
        assert S[0] > 0
        assert U.shape == (2, 1)
        assert V.shape == (2, 1)

    def test_svd_via_lanczos_L2_k1(self):
        """L=2 should work with Lanczos too."""
        torch.manual_seed(42)
        Q = torch.randn(2, 4)
        K = torch.randn(2, 4)

        def mv(v):
            return matvec_S(Q, K, v)

        def mv_t(u):
            return matvec_ST(Q, K, u)

        U, S, V = svd_via_lanczos(mv, mv_t, 2, k=1, iters=10, device="cpu")
        assert S.shape == (1,)
        assert S[0] > 0
