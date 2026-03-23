#!/usr/bin/env python3
"""Hodge decomposition microbenchmark.

Measures wall-clock time of each matrix-free Hodge component at various
sequence lengths.  Prints a readable table and optionally writes JSON.

Usage:
    uv run benchmarks/bench_hodge.py
    uv run benchmarks/bench_hodge.py --lengths 32 64 128 --json results.json
    uv run benchmarks/bench_hodge.py --d-model 128 --warmup 3 --repeats 5
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

import torch

from glassbox.hodge import (
    compute_G_matrix_free,
    compute_routing_features,
    compute_sigma2_asym_matrix_free,
    estimate_commutator_norm_matrix_free,
    estimate_curl_matrix_free,
)
from glassbox.svd import (
    compute_dk_blocked,
    compute_logsumexp_blocked,
    compute_M_fro_norm_blocked,
    matvec_M_blocked,
    matvec_MT_blocked,
    randomized_svd,
)

DEFAULT_LENGTHS = [32, 64, 128, 256, 512, 1024, 2048]
COMPONENTS = ["dk+lse", "SVD", "||M||", "G", "C", "s2asym", "[comm]", "total"]


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def _time_fn(fn, warmup: int, repeats: int) -> float:
    """Return median wall-clock seconds over `repeats` runs after `warmup`."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return statistics.median(times)


def _make_qk(L: int, d: int, seed: int = 42):
    torch.manual_seed(seed)
    Q = torch.randn(L, d)
    K = torch.randn(L, d)
    return Q, K


# ---------------------------------------------------------------------------
# Per-component benchmarks
# ---------------------------------------------------------------------------


def bench_one(
    L: int,
    d: int,
    rank: int,
    block_size: int,
    warmup: int,
    repeats: int,
) -> dict:
    """Benchmark all Hodge components at sequence length L."""
    Q, K = _make_qk(L, d)
    scale = 1.0 / math.sqrt(d)

    # Pre-compute shared quantities
    _, d_k_inv_sqrt = compute_dk_blocked(Q, K, scale, block_size)
    lse = compute_logsumexp_blocked(Q, K, scale, block_size)
    M_fro = compute_M_fro_norm_blocked(Q, K, d_k_inv_sqrt, scale, block_size).item()

    k = min(max(rank, 2), L - 1)
    matvec = lambda v: matvec_M_blocked(Q, K, v, d_k_inv_sqrt, scale, block_size)
    matvec_t = lambda u: matvec_MT_blocked(Q, K, u, d_k_inv_sqrt, scale, block_size)

    timings = {}

    # 1. dk + lse
    timings["dk+lse"] = _time_fn(
        lambda: (
            compute_dk_blocked(Q, K, scale, block_size),
            compute_logsumexp_blocked(Q, K, scale, block_size),
        ),
        warmup, repeats,
    )

    # 2. SVD (randomized)
    timings["SVD"] = _time_fn(
        lambda: randomized_svd(matvec, matvec_t, L, k, device="cpu"),
        warmup, repeats,
    )

    # 3. ||M||_F
    timings["||M||"] = _time_fn(
        lambda: compute_M_fro_norm_blocked(Q, K, d_k_inv_sqrt, scale, block_size),
        warmup, repeats,
    )

    # 4. G (asymmetry)
    timings["G"] = _time_fn(
        lambda: compute_G_matrix_free(Q, K, d_k_inv_sqrt, scale, block_size),
        warmup, repeats,
    )

    # 5. C (curl)
    timings["C"] = _time_fn(
        lambda: estimate_curl_matrix_free(
            Q, K, lse, d_k_inv_sqrt, scale, M_fro, min_samples=200,
        ),
        warmup, repeats,
    )

    # 6. sigma2_asym
    timings["s2asym"] = _time_fn(
        lambda: compute_sigma2_asym_matrix_free(Q, K, d_k_inv_sqrt, scale, block_size),
        warmup, repeats,
    )

    # 7. commutator norm
    timings["[comm]"] = _time_fn(
        lambda: estimate_commutator_norm_matrix_free(
            Q, K, d_k_inv_sqrt, scale, M_fro, block_size, n_hutchinson=10,
        ),
        warmup, repeats,
    )

    # 8. total (end-to-end)
    timings["total"] = _time_fn(
        lambda: compute_routing_features(
            Q, K, d_k_inv_sqrt, scale, lse, rank=rank, block_size=block_size,
            min_samples=200,
        ),
        warmup, repeats,
    )

    return timings


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _fmt_ms(seconds: float) -> str:
    ms = seconds * 1000
    if ms < 1:
        return f"{ms:.2f}ms"
    elif ms < 100:
        return f"{ms:.1f}ms"
    else:
        return f"{ms:.0f}ms"


def _scaling_exponent(lengths: list[int], times: list[float]) -> float | None:
    """Estimate scaling exponent via log-log linear regression."""
    if len(lengths) < 2:
        return None
    xs = [math.log2(L) for L in lengths]
    ys = [math.log2(t) if t > 0 else -30 for t in lengths]
    # Actually use the times, not lengths for ys
    ys = [math.log2(t) if t > 0 else -30 for t in times]
    n = len(xs)
    x_mean = sum(xs) / n
    y_mean = sum(ys) / n
    num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    den = sum((x - x_mean) ** 2 for x in xs)
    if den == 0:
        return None
    return num / den


def print_report(
    lengths: list[int],
    all_timings: list[dict],
    d: int,
    rank: int,
    block_size: int,
) -> None:
    print(f"\nHodge Microbenchmark — d={d}, rank={rank}, block_size={block_size}")
    print("=" * 90)

    # Header
    col_w = 8
    header = f"{'L':>6} |"
    for comp in COMPONENTS:
        header += f" {comp:>{col_w}} |"
    print(header)
    print("-" * len(header))

    # Rows
    for L, timings in zip(lengths, all_timings):
        row = f"{L:>6} |"
        for comp in COMPONENTS:
            row += f" {_fmt_ms(timings[comp]):>{col_w}} |"
        print(row)

    # Scaling exponents
    print("-" * len(header))
    exp_row = f"{'slope':>6} |"
    for comp in COMPONENTS:
        times = [t[comp] for t in all_timings]
        exp = _scaling_exponent(lengths, times)
        if exp is not None:
            exp_row += f" {'~L^' + f'{exp:.1f}':>{col_w}} |"
        else:
            exp_row += f" {'n/a':>{col_w}} |"
    print(exp_row)
    print()

    # Summary
    total_last = all_timings[-1]["total"]
    print(f"At L={lengths[-1]}: total = {_fmt_ms(total_last)}")
    # Breakdown as percentages
    print("Component breakdown:")
    for comp in COMPONENTS[:-1]:  # skip total
        t = all_timings[-1][comp]
        pct = t / total_last * 100 if total_last > 0 else 0
        print(f"  {comp:>8}: {_fmt_ms(t):>8}  ({pct:5.1f}%)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Hodge decomposition microbenchmark")
    parser.add_argument(
        "--lengths", type=int, nargs="+", default=DEFAULT_LENGTHS,
        help=f"Sequence lengths to sweep (default: {DEFAULT_LENGTHS})",
    )
    parser.add_argument("--d-model", type=int, default=64, help="Head dimension (default: 64)")
    parser.add_argument("--rank", type=int, default=4, help="SVD rank (default: 4)")
    parser.add_argument("--block-size", type=int, default=256, help="Block size (default: 256)")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup iterations (default: 2)")
    parser.add_argument("--repeats", type=int, default=3, help="Timed repeats, report median (default: 3)")
    parser.add_argument("--json", type=str, default=None, help="Write JSON results to file")
    args = parser.parse_args()

    print(f"Config: d={args.d_model}, rank={args.rank}, block_size={args.block_size}")
    print(f"Warmup={args.warmup}, repeats={args.repeats}")
    print(f"Lengths: {args.lengths}")

    all_timings = []
    for L in args.lengths:
        sys.stdout.write(f"  Benchmarking L={L}...")
        sys.stdout.flush()
        timings = bench_one(
            L, args.d_model, args.rank, args.block_size,
            args.warmup, args.repeats,
        )
        all_timings.append(timings)
        print(f" {_fmt_ms(timings['total'])}")

    print_report(args.lengths, all_timings, args.d_model, args.rank, args.block_size)

    if args.json:
        output = {
            "config": {
                "d_model": args.d_model,
                "rank": args.rank,
                "block_size": args.block_size,
                "warmup": args.warmup,
                "repeats": args.repeats,
            },
            "results": [
                {"L": L, **{k: v for k, v in t.items()}}
                for L, t in zip(args.lengths, all_timings)
            ],
        }
        Path(args.json).write_text(json.dumps(output, indent=2))
        print(f"JSON written to {args.json}")


if __name__ == "__main__":
    main()
