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
