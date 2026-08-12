"""Correctness tests for the fused Triton forward-matvec kernel.

Gated on CUDA + Triton; skipped on hosts without them (CPU/CI). The kernel must
match the blocked PyTorch forward matvec it accelerates, causal and non-causal.
"""

import pytest
import torch

from glassbox.triton_kernels import HAS_TRITON

pytestmark = pytest.mark.skipif(
    not (HAS_TRITON and torch.cuda.is_available()),
    reason="requires Triton and a CUDA device",
)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("L", [256, 512, 1024])
@pytest.mark.parametrize("d", [48, 64])
def test_fused_kernel_matches_blocked(L, d, causal):
    from glassbox.svd import apply_A_blocked
    from glassbox.triton_kernels import fused_attn_multi_matvec

    torch.manual_seed(0)
    Q = torch.randn(L, d, device="cuda")
    K = torch.randn(L, d, device="cuda")
    Omega = torch.randn(L, 9, device="cuda")
    scale = 1.0 / d**0.5
    out_triton = fused_attn_multi_matvec(Q, K, Omega, scale, causal=causal)
    out_blocked = apply_A_blocked(Q, K, Omega, scale, 256, causal=causal)
    assert torch.allclose(out_triton, out_blocked, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("causal", [False, True])
def test_fused_kernel_non_tile_aligned_L(causal):
    from glassbox.svd import apply_A_blocked
    from glassbox.triton_kernels import fused_attn_multi_matvec

    torch.manual_seed(1)
    L, d = 100, 64  # not divisible by tile size, d power-of-2
    Q = torch.randn(L, d, device="cuda")
    K = torch.randn(L, d, device="cuda")
    Omega = torch.randn(L, 4, device="cuda")
    scale = 1.0 / d**0.5
    assert torch.allclose(
        fused_attn_multi_matvec(Q, K, Omega, scale, causal=causal),
        apply_A_blocked(Q, K, Omega, scale, 256, causal=causal),
        atol=1e-3,
        rtol=1e-3,
    )


def test_signal_features_match_across_strategies():
    """End-to-end at the signal level: routing and tracker features computed with
    matvec_strategy='triton' must match 'batched' (same seed -> identical SVD sketch;
    both diagnostics default causal=True, so this exercises the causal kernel path)."""
    from glassbox.config import RoutingConfig, TrackerConfig
    from glassbox.diagnostics.routing import RoutingDiagnostic
    from glassbox.diagnostics.tracker import TrackerDiagnostic

    torch.manual_seed(7)
    L, d = 384, 64
    Q = torch.randn(L, d, device="cuda")
    K = torch.randn(L, d, device="cuda")

    for diag_cls, conf_cls in (
        (RoutingDiagnostic, RoutingConfig),
        (TrackerDiagnostic, TrackerConfig),
    ):
        feats = {}
        for strategy in ("batched", "triton"):
            # L > threshold -> matrix-free
            diag = diag_cls(conf_cls(rank=4, threshold=64, block_size=128))
            torch.manual_seed(11)  # identical randomized-SVD sketch across strategies
            feats[strategy] = diag.reduce(Q, K, L, matvec_strategy=strategy)["features"]
        fb = feats["batched"].model_dump()
        ft = feats["triton"].model_dump()
        for name, vb in fb.items():
            vt = ft[name]
            if vb is None:
                assert vt is None, (diag_cls.__name__, name)
            elif isinstance(vb, list):
                assert vt == pytest.approx(vb, rel=1e-2, abs=1e-3), (diag_cls.__name__, name)
            else:
                assert vt == pytest.approx(vb, rel=1e-2, abs=1e-3), (diag_cls.__name__, name)


def test_build_forward_matvec_triton_causal_m_path():
    """End-to-end dispatch: strategy='triton' + causal=True on the M operator must
    route through the kernel (not fall back) and match the blocked causal M matvec."""
    from glassbox.svd import build_forward_matvec, matvec_M_blocked

    torch.manual_seed(2)
    L, d = 300, 64
    Q = torch.randn(L, d, device="cuda")
    K = torch.randn(L, d, device="cuda")
    Omega = torch.randn(L, 6, device="cuda")
    dk = (torch.rand(L, device="cuda") + 0.5).to(Q.dtype)
    scale = 1.0 / d**0.5
    mv = build_forward_matvec(Q, K, scale, 256, True, "triton", d_k_inv_sqrt=dk)
    expected = matvec_M_blocked(Q, K, Omega, dk, scale, 256, causal=True)
    assert torch.allclose(mv(Omega), expected, atol=1e-3, rtol=1e-3)
