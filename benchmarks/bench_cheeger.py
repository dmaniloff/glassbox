"""Benchmark: Cheeger batch vs streaming — timing and numerical gap.

Usage:
    .venv/bin/python benchmarks/bench_cheeger.py

Reports a markdown table with:
  - L, D, causal: test configuration
  - phi_mat, phi_mf: conductance values from each path
  - abs_gap, rel_gap: numerical difference
  - time_mat_ms, time_mf_ms: wall-clock timing
  - bounds_ok: whether Cheeger inequality bracket holds
  - witness_agree: fraction of witness positions that agree
"""

import math
import time

import torch

from glassbox.diagnostics.cheeger import CheegerDiagnostic


def _random_qk(L, D, seed=42):
    torch.manual_seed(seed)
    return torch.randn(L, D), torch.randn(L, D)


def _time_reduce(diag, Q, K, L, warmup=1, repeats=3):
    """Time reduce() with warmup."""
    for _ in range(warmup):
        diag.reduce(Q, K, L)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = diag.reduce(Q, K, L)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return result, min(times)


def _witness_agreement(diag_mat, diag_mf, Q, K, L):
    """Compute witness agreement (accounting for global sign flip)."""
    w_mat = diag_mat.witness(Q, K, L)
    w_mf = diag_mf.witness(Q, K, L)
    agree_same = (w_mat == w_mf).float().mean().item()
    agree_flip = (w_mat == -w_mf).float().mean().item()
    return max(agree_same, agree_flip)


def main():
    D = 64
    sizes = [32, 64, 128, 256, 512]
    causal_modes = [False, True]
    seed = 42

    header = (
        "| L | D | causal | phi_mat | phi_mf | abs_gap | rel_gap | "
        "time_mat_ms | time_mf_ms | speedup | bounds_ok | witness_agree |"
    )
    sep = "|" + "|".join(["---"] * 12) + "|"

    print("# Cheeger Benchmark: Batch vs Streaming\n")
    print(header)
    print(sep)

    for causal in causal_modes:
        for L in sizes:
            Q, K = _random_qk(L, D, seed=seed)

            diag_mat = CheegerDiagnostic(rank=2, threshold=L + 1, causal=causal)
            diag_mf = CheegerDiagnostic(rank=2, threshold=0, block_size=64, causal=causal)

            result_mat, t_mat = _time_reduce(diag_mat, Q, K, L)
            result_mf, t_mf = _time_reduce(diag_mf, Q, K, L)

            f_mat = result_mat["features"]
            f_mf = result_mf["features"]

            abs_gap = abs(f_mat.phi_star - f_mf.phi_star)
            rel_gap = abs_gap / max(f_mat.phi_star, 1e-8)

            bounds_ok = (
                f_mat.cheeger_lower - 1e-6 <= f_mat.phi_star <= f_mat.cheeger_upper + 1e-6
            )

            witness_agree = _witness_agreement(diag_mat, diag_mf, Q, K, L)

            speedup = t_mat / t_mf if t_mf > 0 else float("inf")

            print(
                f"| {L} | {D} | {causal} | {f_mat.phi_star:.4f} | {f_mf.phi_star:.4f} | "
                f"{abs_gap:.4f} | {rel_gap:.2%} | "
                f"{t_mat*1000:.1f} | {t_mf*1000:.1f} | {speedup:.2f}x | "
                f"{'Y' if bounds_ok else 'N'} | {witness_agree:.2f} |"
            )

    print()


if __name__ == "__main__":
    main()
