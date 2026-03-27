"""Tests for fused Triton kernels (require CUDA)."""

import math

import pytest
import torch

from glassbox.svd import apply_A_blocked_batched

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)

try:
    from glassbox.triton_kernels import fused_attn_multi_matvec
except ImportError:
    pytestmark = pytest.mark.skip(reason="triton not installed")


@pytest.fixture(params=[256, 512, 1024, 2048])
def setup(request):
    L = request.param
    d = 64
    n_vecs = 9  # k=4, p=5
    scale = 1.0 / math.sqrt(d)
    torch.manual_seed(42)
    Q = torch.randn(L, d, device="cuda")
    K = torch.randn(L, d, device="cuda")
    Omega = torch.randn(L, n_vecs, device="cuda")
    return Q, K, Omega, scale


def test_fused_kernel_matches_batched(setup):
    """Triton fused kernel should match PyTorch batched reference."""
    Q, K, Omega, scale = setup
    ref = apply_A_blocked_batched(Q, K, Omega, scale, block_size=256)
    out = fused_attn_multi_matvec(Q, K, Omega, scale)
    torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)


def test_fused_kernel_non_power_of_2_d():
    """Kernel handles head dims that aren't powers of 2."""
    d = 48
    L = 128
    n_vecs = 5
    scale = 1.0 / math.sqrt(d)
    torch.manual_seed(99)
    Q = torch.randn(L, d, device="cuda")
    K = torch.randn(L, d, device="cuda")
    Omega = torch.randn(L, n_vecs, device="cuda")

    ref = apply_A_blocked_batched(Q, K, Omega, scale, block_size=64)
    out = fused_attn_multi_matvec(Q, K, Omega, scale)
    torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)


def test_fused_kernel_non_tile_aligned_L():
    """Kernel handles L not divisible by tile size."""
    L = 100
    d = 64
    n_vecs = 9
    scale = 1.0 / math.sqrt(d)
    torch.manual_seed(11)
    Q = torch.randn(L, d, device="cuda")
    K = torch.randn(L, d, device="cuda")
    Omega = torch.randn(L, n_vecs, device="cuda")

    ref = apply_A_blocked_batched(Q, K, Omega, scale, block_size=64)
    out = fused_attn_multi_matvec(Q, K, Omega, scale)
    torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)
