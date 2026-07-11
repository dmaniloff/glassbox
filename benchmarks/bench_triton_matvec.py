"""Benchmark the fused Triton forward matvec vs the blocked PyTorch path.

GPU + Triton only; prints a notice and exits otherwise. Measures the forward
A @ Omega used by the matrix-free SVD (randomized_svd) across sequence lengths.

    python benchmarks/bench_triton_matvec.py
"""

from __future__ import annotations

import torch

from glassbox.svd import apply_A_blocked
from glassbox.triton_kernels import HAS_TRITON


def _time(fn, iters=50):
    fn()  # warmup / compile
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms


def main() -> None:
    if not (HAS_TRITON and torch.cuda.is_available()):
        print("Triton + CUDA required; skipping benchmark.")
        return

    from glassbox.triton_kernels import fused_attn_multi_matvec

    d, n_vecs = 64, 9
    scale = 1.0 / d**0.5
    print(f"{'L':>6} {'blocked(ms)':>12} {'triton(ms)':>12} {'speedup':>9}")
    for L in (256, 512, 1024, 2048, 4096):
        Q = torch.randn(L, d, device="cuda")
        K = torch.randn(L, d, device="cuda")
        Omega = torch.randn(L, n_vecs, device="cuda")
        t_blocked = _time(lambda: apply_A_blocked(Q, K, Omega, scale, 256, causal=False))
        t_triton = _time(lambda: fused_attn_multi_matvec(Q, K, Omega, scale))
        print(f"{L:>6} {t_blocked:>12.3f} {t_triton:>12.3f} {t_blocked / t_triton:>8.1f}x")


if __name__ == "__main__":
    main()
