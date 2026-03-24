#!/usr/bin/env python3
"""Hodge decomposition microbenchmark.

Compares materialized vs matrix-free routing features at various sequence
lengths, and breaks down matrix-free cost by component.

Usage:
    python benchmarks/bench_hodge.py
    python benchmarks/bench_hodge.py --lengths 32 64 128 --json results.json
    python benchmarks/bench_hodge.py --d-model 128 --warmup 3 --repeats 5
"""

from __future__ import annotations

import json
import math
import statistics
import sys
import time
from pathlib import Path

import click
import torch

from glassbox.hodge import (
    _compute_G_materialized,
    _estimate_curl_materialized,
    compute_G_matrix_free,
    compute_routing_features_materialized,
    compute_routing_features_matrix_free,
    compute_sigma2_asym_matrix_free,
    estimate_commutator_norm_matrix_free,
    estimate_curl_matrix_free,
)
from glassbox.svd import (
    compute_degree_normalized_M,
    compute_dk_blocked,
    compute_logsumexp_blocked,
    compute_M_fro_norm_blocked,
    matvec_M_blocked,
    matvec_MT_blocked,
    randomized_svd,
)

DEFAULT_LENGTHS = [32, 64, 128, 256, 512, 1024, 2048]
MAT_COMPONENTS = ["A+M", "SVD", "||M||", "G", "C", "s2asym", "[comm]", "total"]
MF_COMPONENTS = ["dk+lse", "SVD", "||M||", "G", "C", "s2asym", "[comm]", "total"]


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def _time_fn(fn, warmup: int, repeats: int, device: str = "cpu") -> float:
    """Return median wall-clock seconds over `repeats` runs after `warmup`."""
    for _ in range(warmup):
        fn()
        _sync(device)
    times = []
    for _ in range(repeats):
        _sync(device)
        t0 = time.perf_counter()
        fn()
        _sync(device)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return statistics.median(times)


def _make_qk(L: int, d: int, device: str = "cpu", seed: int = 42):
    torch.manual_seed(seed)
    Q = torch.randn(L, d, device=device)
    K = torch.randn(L, d, device=device)
    return Q, K


def _sync(device: str):
    """Synchronize GPU before timing."""
    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Per-component benchmarks (matrix-free)
# ---------------------------------------------------------------------------


def bench_matrix_free(
    L: int,
    d: int,
    rank: int,
    block_size: int,
    warmup: int,
    repeats: int,
    device: str = "cpu",
) -> dict:
    """Benchmark all matrix-free Hodge components at sequence length L."""
    Q, K = _make_qk(L, d, device)
    scale = 1.0 / math.sqrt(d)

    # Pre-compute shared quantities
    _, d_k_inv_sqrt = compute_dk_blocked(Q, K, scale, block_size)
    lse = compute_logsumexp_blocked(Q, K, scale, block_size)
    M_fro = compute_M_fro_norm_blocked(Q, K, d_k_inv_sqrt, scale, block_size).item()

    k = min(max(rank, 2), L - 1)
    matvec = lambda v: matvec_M_blocked(Q, K, v, d_k_inv_sqrt, scale, block_size)
    matvec_t = lambda u: matvec_MT_blocked(Q, K, u, d_k_inv_sqrt, scale, block_size)

    timings = {}

    timings["dk+lse"] = _time_fn(
        lambda: (
            compute_dk_blocked(Q, K, scale, block_size),
            compute_logsumexp_blocked(Q, K, scale, block_size),
        ),
        warmup, repeats, device,
    )

    timings["SVD"] = _time_fn(
        lambda: randomized_svd(matvec, matvec_t, L, k, device=device),
        warmup, repeats, device,
    )

    timings["||M||"] = _time_fn(
        lambda: compute_M_fro_norm_blocked(Q, K, d_k_inv_sqrt, scale, block_size),
        warmup, repeats, device,
    )

    timings["G"] = _time_fn(
        lambda: compute_G_matrix_free(Q, K, d_k_inv_sqrt, scale, block_size),
        warmup, repeats, device,
    )

    timings["C"] = _time_fn(
        lambda: estimate_curl_matrix_free(
            Q, K, lse, d_k_inv_sqrt, scale, M_fro, min_samples=200,
        ),
        warmup, repeats, device,
    )

    timings["s2asym"] = _time_fn(
        lambda: compute_sigma2_asym_matrix_free(Q, K, d_k_inv_sqrt, scale, block_size),
        warmup, repeats, device,
    )

    timings["[comm]"] = _time_fn(
        lambda: estimate_commutator_norm_matrix_free(
            Q, K, d_k_inv_sqrt, scale, M_fro, block_size, n_hutchinson=10,
        ),
        warmup, repeats, device,
    )

    timings["total"] = _time_fn(
        lambda: compute_routing_features_matrix_free(
            Q, K, d_k_inv_sqrt, scale, lse, rank=rank, block_size=block_size,
            min_samples=200,
        ),
        warmup, repeats, device,
    )

    return timings


# ---------------------------------------------------------------------------
# Per-component benchmarks (materialized)
# ---------------------------------------------------------------------------


def bench_materialized(
    L: int,
    d: int,
    rank: int,
    warmup: int,
    repeats: int,
    device: str = "cpu",
) -> dict:
    """Benchmark all materialized Hodge components at sequence length L."""
    Q, K = _make_qk(L, d, device)
    scale = 1.0 / math.sqrt(d)

    # Pre-compute M
    A = torch.softmax(Q @ K.T * scale, dim=-1)
    M, _, _ = compute_degree_normalized_M(A)

    timings = {}

    timings["A+M"] = _time_fn(
        lambda: compute_degree_normalized_M(torch.softmax(Q @ K.T * scale, dim=-1)),
        warmup, repeats, device,
    )

    timings["SVD"] = _time_fn(
        lambda: torch.linalg.svdvals(M),
        warmup, repeats, device,
    )

    timings["||M||"] = _time_fn(
        lambda: torch.linalg.norm(M, "fro"),
        warmup, repeats, device,
    )

    timings["G"] = _time_fn(
        lambda: _compute_G_materialized(M),
        warmup, repeats, device,
    )

    timings["C"] = _time_fn(
        lambda: _estimate_curl_materialized(M),
        warmup, repeats, device,
    )

    timings["s2asym"] = _time_fn(
        lambda: torch.linalg.svdvals((M - M.T) / 2.0),
        warmup, repeats, device,
    )

    M_sym = (M + M.T) / 2.0
    M_asym = (M - M.T) / 2.0
    timings["[comm]"] = _time_fn(
        lambda: torch.linalg.norm(M_sym @ M_asym - M_asym @ M_sym, "fro"),
        warmup, repeats, device,
    )

    timings["total"] = _time_fn(
        lambda: compute_routing_features_materialized(
            compute_degree_normalized_M(torch.softmax(Q @ K.T * scale, dim=-1))[0],
            rank=rank,
        ),
        warmup, repeats, device,
    )

    return timings


# ---------------------------------------------------------------------------
# End-to-end: materialized vs matrix-free
# ---------------------------------------------------------------------------


def bench_comparison(
    L: int,
    d: int,
    rank: int,
    block_size: int,
    warmup: int,
    repeats: int,
    device: str = "cpu",
) -> dict:
    """Benchmark materialized vs matrix-free end-to-end at sequence length L."""
    Q, K = _make_qk(L, d, device)
    scale = 1.0 / math.sqrt(d)

    # Materialized
    def run_mat():
        A = torch.softmax(Q @ K.T * scale, dim=-1)
        M, _, _ = compute_degree_normalized_M(A)
        return compute_routing_features_materialized(M, rank=rank)

    # Matrix-free
    _, d_k_inv_sqrt = compute_dk_blocked(Q, K, scale, block_size)
    lse = compute_logsumexp_blocked(Q, K, scale, block_size)

    def run_mf():
        return compute_routing_features_matrix_free(
            Q, K, d_k_inv_sqrt, scale, lse, rank=rank, block_size=block_size,
            min_samples=200,
        )

    return {
        "materialized": _time_fn(run_mat, warmup, repeats, device),
        "matrix_free": _time_fn(run_mf, warmup, repeats, device),
    }


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
    ys = [math.log2(t) if t > 0 else -30 for t in times]
    n = len(xs)
    x_mean = sum(xs) / n
    y_mean = sum(ys) / n
    num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    den = sum((x - x_mean) ** 2 for x in xs)
    if den == 0:
        return None
    return num / den


def print_comparison_report(
    lengths: list[int],
    all_comparisons: list[dict],
    d: int,
    rank: int,
) -> None:
    print(f"\nMaterialized vs Matrix-Free (all Hodge features) — d={d}, rank={rank}")
    print("=" * 68)
    header = f"{'L':>6} | {'materialized':>14} | {'matrix-free':>14} | {'ratio':>8}"
    print(header)
    print("-" * len(header))
    for L, comp in zip(lengths, all_comparisons):
        t_mat = comp["materialized"]
        t_mf = comp["matrix_free"]
        ratio = t_mf / t_mat if t_mat > 0 else float("inf")
        print(
            f"{L:>6} | {_fmt_ms(t_mat):>14} | {_fmt_ms(t_mf):>14} | {ratio:>7.1f}x"
        )

    # Scaling exponents
    print("-" * len(header))
    mat_times = [c["materialized"] for c in all_comparisons]
    mf_times = [c["matrix_free"] for c in all_comparisons]
    mat_exp = _scaling_exponent(lengths, mat_times)
    mf_exp = _scaling_exponent(lengths, mf_times)
    mat_s = f"~L^{mat_exp:.1f}" if mat_exp is not None else "n/a"
    mf_s = f"~L^{mf_exp:.1f}" if mf_exp is not None else "n/a"
    print(f"{'slope':>6} | {mat_s:>14} | {mf_s:>14} |")
    print()


def print_component_report(
    title: str,
    components: list[str],
    lengths: list[int],
    all_timings: list[dict],
) -> None:
    print(f"\n{title}")
    print("=" * 90)

    col_w = 8
    header = f"{'L':>6} |"
    for comp in components:
        header += f" {comp:>{col_w}} |"
    print(header)
    print("-" * len(header))

    for L, timings in zip(lengths, all_timings):
        row = f"{L:>6} |"
        for comp in components:
            row += f" {_fmt_ms(timings[comp]):>{col_w}} |"
        print(row)

    print("-" * len(header))
    exp_row = f"{'slope':>6} |"
    for comp in components:
        times = [t[comp] for t in all_timings]
        exp = _scaling_exponent(lengths, times)
        if exp is not None:
            exp_row += f" {'~L^' + f'{exp:.1f}':>{col_w}} |"
        else:
            exp_row += f" {'n/a':>{col_w}} |"
    print(exp_row)
    print()

    total_last = all_timings[-1]["total"]
    print(f"At L={lengths[-1]}: total = {_fmt_ms(total_last)}")
    print("Component breakdown:")
    for comp in components[:-1]:
        t = all_timings[-1][comp]
        pct = t / total_last * 100 if total_last > 0 else 0
        print(f"  {comp:>8}: {_fmt_ms(t):>8}  ({pct:5.1f}%)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_int_list(ctx, param, value):
    """Parse a comma-separated list of ints."""
    if not value:
        return param.default
    return tuple(int(x.strip()) for x in value.split(","))


@click.command()
@click.option(
    "--lengths", type=str, default=",".join(str(x) for x in DEFAULT_LENGTHS),
    callback=_parse_int_list, show_default=True,
    help="Comma-separated sequence lengths to sweep.",
)
@click.option("--d-model", type=int, default=64, show_default=True, help="Head dimension.")
@click.option("--rank", type=int, default=4, show_default=True, help="SVD rank.")
@click.option("--block-size", type=int, default=256, show_default=True, help="Block size.")
@click.option("--warmup", type=int, default=2, show_default=True, help="Warmup iterations.")
@click.option("--repeats", type=int, default=3, show_default=True, help="Timed repeats, report median.")
@click.option("--json", "json_path", type=click.Path(), default=None, help="Write JSON results to file.")
@click.option(
    "--comparison-only", is_flag=True, default=False,
    help="Only run materialized vs matrix-free comparison (skip component breakdown).",
)
@click.option(
    "--device", type=str, default="cpu", show_default=True,
    help="Torch device (cpu, cuda, cuda:0, etc.).",
)
def main(
    lengths: tuple[int, ...],
    d_model: int,
    rank: int,
    block_size: int,
    warmup: int,
    repeats: int,
    json_path: str | None,
    comparison_only: bool,
    device: str,
) -> None:
    """Hodge decomposition microbenchmark."""
    lengths_list = list(lengths)

    if device != "cpu" and not torch.cuda.is_available():
        raise click.UsageError(f"Device '{device}' requested but CUDA is not available.")

    print(f"Config: d={d_model}, rank={rank}, block_size={block_size}, device={device}")
    print(f"Warmup={warmup}, repeats={repeats}")
    print(f"Lengths: {lengths_list}")

    # 1. Materialized vs matrix-free comparison
    print("\n--- Materialized vs Matrix-Free ---")
    all_comparisons = []
    for L in lengths_list:
        sys.stdout.write(f"  L={L}...")
        sys.stdout.flush()
        comp = bench_comparison(L, d_model, rank, block_size, warmup, repeats, device)
        all_comparisons.append(comp)
        ratio = comp["matrix_free"] / comp["materialized"] if comp["materialized"] > 0 else float("inf")
        print(f" mat={_fmt_ms(comp['materialized'])}, mf={_fmt_ms(comp['matrix_free'])} ({ratio:.1f}x)")

    print_comparison_report(lengths_list, all_comparisons, d_model, rank)

    # 2. Component breakdowns
    all_mat_timings = []
    all_mf_timings = []
    if not comparison_only:
        print("\n--- Materialized Component Breakdown ---")
        for L in lengths_list:
            sys.stdout.write(f"  L={L}...")
            sys.stdout.flush()
            timings = bench_materialized(L, d_model, rank, warmup, repeats, device)
            all_mat_timings.append(timings)
            print(f" {_fmt_ms(timings['total'])}")

        print_component_report(
            f"Materialized Component Breakdown — d={d_model}, rank={rank}",
            MAT_COMPONENTS, lengths_list, all_mat_timings,
        )

        print("\n--- Matrix-Free Component Breakdown ---")
        for L in lengths_list:
            sys.stdout.write(f"  L={L}...")
            sys.stdout.flush()
            timings = bench_matrix_free(L, d_model, rank, block_size, warmup, repeats, device)
            all_mf_timings.append(timings)
            print(f" {_fmt_ms(timings['total'])}")

        print_component_report(
            f"Matrix-Free Component Breakdown — d={d_model}, rank={rank}, block_size={block_size}",
            MF_COMPONENTS, lengths_list, all_mf_timings,
        )

    if json_path:
        output = {
            "config": {
                "d_model": d_model,
                "rank": rank,
                "block_size": block_size,
                "warmup": warmup,
                "repeats": repeats,
            },
            "comparison": [
                {"L": L, **c}
                for L, c in zip(lengths_list, all_comparisons)
            ],
        }
        if all_mat_timings:
            output["materialized_components"] = [
                {"L": L, **t}
                for L, t in zip(lengths_list, all_mat_timings)
            ]
        if all_mf_timings:
            output["matrix_free_components"] = [
                {"L": L, **t}
                for L, t in zip(lengths_list, all_mf_timings)
            ]
        Path(json_path).write_text(json.dumps(output, indent=2))
        print(f"JSON written to {json_path}")


if __name__ == "__main__":
    main()
