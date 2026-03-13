import torch
import pytest

from glassbox.svd import (
    matvec_S,
    matvec_ST,
    randomized_svd,
    svd_via_lanczos,
    compare_svd_results,
)

L = 8
D = 2


@pytest.fixture
def qk():
    torch.manual_seed(42)
    Q = torch.randn(L, D)
    K = torch.randn(L, D)
    matvec = lambda x: matvec_S(Q, K, x)
    matvec_t = lambda x: matvec_ST(Q, K, x)
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
        matvec=matvec, matvec_t=matvec_t,
        U1=U_l, S1=S_l, V1=V_l,
        U2=U_r, S2=S_r, V2=V_r,
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
