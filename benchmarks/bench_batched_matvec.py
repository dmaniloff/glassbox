"""Microbenchmark: loop vs batched vs triton for attention multi-matvec.

Compares three approaches for computing A @ Omega where A = softmax(QK^T * scale)
across sequence lengths L ∈ {256, 512, 1024, 2048, 4096}.

Usage:
    source /opt/pytorch/bin/activate
    python benchmarks/bench_batched_matvec.py
"""

from __future__ import annotations

import math
import time

import torch

from glassbox.svd import apply_A_blocked, apply_A_blocked_batched

HAS_TRITON = False
try:
    from glassbox.triton_kernels import fused_attn_multi_matvec

    HAS_TRITON = True
except ImportError:
    pass


def bench_loop(Q, K, Omega, scale, block_size, warmup=3, repeats=10):
    """Baseline: per-vector apply_A_blocked calls."""
    n_vecs = Omega.shape[1]
    # warmup
    for _ in range(warmup):
        for i in range(n_vecs):
            apply_A_blocked(Q, K, Omega[:, i], scale, block_size)
    if Q.is_cuda:
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(repeats):
        for i in range(n_vecs):
            apply_A_blocked(Q, K, Omega[:, i], scale, block_size)
    if Q.is_cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeats


def bench_batched(Q, K, Omega, scale, block_size, warmup=3, repeats=10):
    """PyTorch batched: single call, softmax computed once per block."""
    for _ in range(warmup):
        apply_A_blocked_batched(Q, K, Omega, scale, block_size)
    if Q.is_cuda:
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(repeats):
        apply_A_blocked_batched(Q, K, Omega, scale, block_size)
    if Q.is_cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeats


def bench_triton(Q, K, Omega, scale, warmup=3, repeats=10):
    """Fused Triton kernel with online softmax."""
    for _ in range(warmup):
        fused_attn_multi_matvec(Q, K, Omega, scale)
    if Q.is_cuda:
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(repeats):
        fused_attn_multi_matvec(Q, K, Omega, scale)
    if Q.is_cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeats


def main():
    seq_lengths = [256, 512, 1024, 2048, 4096]
    d = 64
    k, p = 4, 5
    n_vecs = k + p  # 9
    block_size = 256
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"device={device}  d={d}  n_vecs={n_vecs}  block_size={block_size}")
    print(f"{'L':>6}  {'loop (ms)':>10}  {'batched (ms)':>12}  {'triton (ms)':>12}  {'batch/loop':>10}  {'triton/loop':>11}")
    print("-" * 75)

    for L in seq_lengths:
        torch.manual_seed(42)
        Q = torch.randn(L, d, device=device)
        K = torch.randn(L, d, device=device)
        Omega = torch.randn(L, n_vecs, device=device)
        scale = 1.0 / math.sqrt(d)

        t_loop = bench_loop(Q, K, Omega, scale, block_size)
        t_batched = bench_batched(Q, K, Omega, scale, block_size)

        if HAS_TRITON and device == "cuda":
            t_triton = bench_triton(Q, K, Omega, scale)
            print(
                f"{L:>6}  {t_loop*1e3:>10.2f}  {t_batched*1e3:>12.2f}  {t_triton*1e3:>12.2f}"
                f"  {t_loop/t_batched:>10.2f}x  {t_loop/t_triton:>11.2f}x"
            )
        else:
            print(
                f"{L:>6}  {t_loop*1e3:>10.2f}  {t_batched*1e3:>12.2f}  {'n/a':>12}"
                f"  {t_loop/t_batched:>10.2f}x  {'n/a':>11}"
            )


if __name__ == "__main__":
    main()
