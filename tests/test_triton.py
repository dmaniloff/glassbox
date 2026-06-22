"""Correctness tests for the fused Triton forward-matvec kernel.

Gated on CUDA + Triton; skipped on hosts without them (CPU/CI). The kernel must
match the blocked PyTorch forward matvec it accelerates.
"""

import pytest
import torch

from glassbox.triton_kernels import HAS_TRITON

pytestmark = pytest.mark.skipif(
    not (HAS_TRITON and torch.cuda.is_available()),
    reason="requires Triton and a CUDA device",
)


@pytest.mark.parametrize("L", [256, 512, 1024])
@pytest.mark.parametrize("d", [48, 64])
def test_fused_kernel_matches_blocked(L, d):
    from glassbox.svd import apply_A_blocked
    from glassbox.triton_kernels import fused_attn_multi_matvec

    torch.manual_seed(0)
    Q = torch.randn(L, d, device="cuda")
    K = torch.randn(L, d, device="cuda")
    Omega = torch.randn(L, 9, device="cuda")
    scale = 1.0 / d**0.5
    out_triton = fused_attn_multi_matvec(Q, K, Omega, scale)
    out_blocked = apply_A_blocked(Q, K, Omega, scale, 256, causal=False)
    assert torch.allclose(out_triton, out_blocked, atol=1e-3, rtol=1e-3)


def test_fused_kernel_non_tile_aligned_L():
    from glassbox.svd import apply_A_blocked
    from glassbox.triton_kernels import fused_attn_multi_matvec

    torch.manual_seed(1)
    L, d = 100, 64  # not divisible by tile size, d power-of-2
    Q = torch.randn(L, d, device="cuda")
    K = torch.randn(L, d, device="cuda")
    Omega = torch.randn(L, 4, device="cuda")
    scale = 1.0 / d**0.5
    assert torch.allclose(
        fused_attn_multi_matvec(Q, K, Omega, scale),
        apply_A_blocked(Q, K, Omega, scale, 256, causal=False),
        atol=1e-3,
        rtol=1e-3,
    )
