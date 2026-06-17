"""Benchmark: Streaming Cheeger parameter sweep on real GPT-2 attention.

Usage:
    # Extract fixtures first (requires transformers):
    python benchmarks/extract_gpt2_qk.py

    # Run parameter sweep:
    .venv/bin/python benchmarks/bench_cheeger.py [--fixtures-dir benchmarks/fixtures/gpt2_qk]

    # Synthetic fallback (no fixtures needed):
    .venv/bin/python benchmarks/bench_cheeger.py --synthetic

Reports:
  1. Batch vs matrix-free comparison table (original benchmark)
  2. Streaming parameter sweep: finds Pareto-optimal settings for bracket fidelity
  3. Writes detailed report to benchmarks/results/cheeger_streaming_report.md
"""

from __future__ import annotations

import itertools
import math
import time
from pathlib import Path

import torch

from glassbox.diagnostics.cheeger import CheegerDiagnostic


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_fixtures(fixtures_dir: str) -> list[dict]:
    """Load saved Q/K .pt files from extract_gpt2_qk.py."""
    p = Path(fixtures_dir)
    if not p.exists():
        return []
    fixtures = []
    for f in sorted(p.glob("*.pt")):
        data = torch.load(f, map_location="cpu", weights_only=True)
        parts = f.stem.split("_")
        fixtures.append({
            "Q": data["Q"].float(),
            "K": data["K"].float(),
            "name": f.stem,
            "prompt": parts[0] if parts else "?",
            "layer": parts[1] if len(parts) > 1 else "?",
            "head": parts[2] if len(parts) > 2 else "?",
            "L": data["Q"].shape[0],
        })
    return fixtures


def generate_synthetic(n=10, seed=42) -> list[dict]:
    """Generate synthetic Q/K fixtures as fallback."""
    fixtures = []
    for i, L in enumerate([32, 48, 64, 96, 128, 192, 256, 32, 64, 128]):
        D = 64
        torch.manual_seed(seed + i)
        Q = torch.randn(L, D)
        K = torch.randn(L, D)
        fixtures.append({
            "Q": Q, "K": K,
            "name": f"synthetic_L{L}_s{seed+i}",
            "prompt": f"synth{i}", "layer": "0", "head": "0",
            "L": L,
        })
        if len(fixtures) >= n:
            break
    return fixtures


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------

def compute_ground_truth(Q, K, L, causal=True):
    """Full materialized Cheeger bracket = ground truth."""
    diag = CheegerDiagnostic(rank=2, threshold=L + 1, causal=causal, streaming=False)
    result = diag.reduce(Q, K, L)
    return result["features"]


# ---------------------------------------------------------------------------
# Streaming simulation
# ---------------------------------------------------------------------------

def simulate_streaming(Q, K, L, window_size, params, causal=True):
    """Slide windows over Q/K, run streaming controller, collect metrics."""
    diag = CheegerDiagnostic(
        rank=2, threshold=0, block_size=min(64, window_size),
        causal=causal, streaming=True, **params,
    )

    state = None
    results = []
    n_windows = max(1, (L - window_size) // (window_size // 2) + 1)
    step_size = max(1, window_size // 2)

    for i in range(n_windows):
        start = min(i * step_size, L - window_size)
        end = start + window_size
        if end > L:
            break
        Qw = Q[start:end]
        Kw = K[start:end]
        w = end - start

        t0 = time.perf_counter()
        result = diag.reduce(Qw, Kw, w, prior_state=state)
        t1 = time.perf_counter()

        result["step"] = i + 1
        state = diag.accumulate(result, state)
        f = result["features"]
        results.append({
            "sigma2": f.sigma2,
            "phi_star": f.phi_star,
            "cheeger_lower": f.cheeger_lower,
            "cheeger_upper": f.cheeger_upper,
            "bracket_width": f.bracket_width,
            "spectral_gap": f.spectral_gap,
            "recomputed": f.recomputed,
            "tier": result.get("tier"),
            "time_ms": (t1 - t0) * 1000,
        })

    return results


# ---------------------------------------------------------------------------
# Parameter sweep
# ---------------------------------------------------------------------------

SWEEP_GRID = {
    "ritz_rank": [2, 3, 4, 6],
    "n_explore": [0, 1, 2, 4],
    "gap_threshold": [0.001, 0.005, 0.01, 0.05],
    "geometric_base": [1.2, 1.5, 2.0, 3.0],
    "lanczos_iters": [10, 20, 30, 50],
}

DEFAULTS = {
    "ritz_rank": 3,
    "n_explore": 2,
    "gap_threshold": 0.01,
    "geometric_base": 1.5,
    "lanczos_iters": 30,
    "improved_cheeger_k": 4,
    "degree_shift_threshold": 0.1,
}

WINDOW_SIZES = [32, 64, 128]


def sweep_one_param(param_name, fixtures, causal=True):
    """Sweep one parameter while holding others at defaults."""
    results = []
    for val in SWEEP_GRID[param_name]:
        params = dict(DEFAULTS)
        params[param_name] = val

        metrics_accum = {
            "sigma2_drift": [], "phi_drift": [], "bracket_width": [],
            "recompute_rate": [], "time_ms": [], "bounds_valid": [],
        }

        for fix in fixtures:
            Q, K, L = fix["Q"], fix["K"], fix["L"]
            if L < 64:
                continue

            gt = compute_ground_truth(Q, K, L, causal=causal)

            for ws in WINDOW_SIZES:
                if ws > L:
                    continue
                stream_results = simulate_streaming(Q, K, L, ws, params, causal=causal)
                if not stream_results:
                    continue

                for sr in stream_results:
                    if gt.sigma2 is not None and sr["sigma2"] is not None:
                        metrics_accum["sigma2_drift"].append(abs(gt.sigma2 - sr["sigma2"]))
                    metrics_accum["phi_drift"].append(abs(gt.phi_star - sr["phi_star"]))
                    if sr["bracket_width"] is not None:
                        metrics_accum["bracket_width"].append(sr["bracket_width"])
                    metrics_accum["recompute_rate"].append(1.0 if sr["recomputed"] else 0.0)
                    metrics_accum["time_ms"].append(sr["time_ms"])
                    if sr["cheeger_lower"] is not None and sr["cheeger_upper"] is not None:
                        valid = sr["cheeger_lower"] - 0.01 <= sr["phi_star"] <= sr["cheeger_upper"] + 0.01
                        metrics_accum["bounds_valid"].append(1.0 if valid else 0.0)

        def mean_or(lst, default=0.0):
            return sum(lst) / len(lst) if lst else default

        results.append({
            "param": param_name,
            "value": val,
            "sigma2_drift": mean_or(metrics_accum["sigma2_drift"]),
            "phi_drift": mean_or(metrics_accum["phi_drift"]),
            "bracket_width": mean_or(metrics_accum["bracket_width"]),
            "recompute_rate": mean_or(metrics_accum["recompute_rate"]),
            "time_ms": mean_or(metrics_accum["time_ms"]),
            "bounds_valid": mean_or(metrics_accum["bounds_valid"], 1.0),
            "n_samples": len(metrics_accum["phi_drift"]),
        })

    return results


# ---------------------------------------------------------------------------
# Batch vs matrix-free comparison (original benchmark)
# ---------------------------------------------------------------------------

def batch_vs_mf_table(fixtures):
    """Original benchmark: compare batch (materialized) vs matrix-free."""
    print("# Cheeger Benchmark: Batch vs Matrix-Free\n")
    header = (
        "| Name | L | phi_mat | phi_mf | abs_gap | rel_gap | "
        "time_mat_ms | time_mf_ms | bounds_ok |"
    )
    sep = "|" + "|".join(["---"] * 8) + "|"
    print(header)
    print(sep)

    for fix in fixtures[:20]:
        Q, K, L = fix["Q"], fix["K"], fix["L"]
        name = fix["name"][:30]

        diag_mat = CheegerDiagnostic(rank=2, threshold=L + 1, causal=True)
        diag_mf = CheegerDiagnostic(rank=2, threshold=0, block_size=64, causal=True)

        t0 = time.perf_counter()
        result_mat = diag_mat.reduce(Q, K, L)
        t_mat = time.perf_counter() - t0

        t0 = time.perf_counter()
        result_mf = diag_mf.reduce(Q, K, L)
        t_mf = time.perf_counter() - t0

        f_mat = result_mat["features"]
        f_mf = result_mf["features"]

        abs_gap = abs(f_mat.phi_star - f_mf.phi_star)
        rel_gap = abs_gap / max(f_mat.phi_star, 1e-8)
        bounds_ok = f_mat.cheeger_lower - 1e-6 <= f_mat.phi_star <= f_mat.cheeger_upper + 1e-6

        print(
            f"| {name} | {L} | {f_mat.phi_star:.4f} | {f_mf.phi_star:.4f} | "
            f"{abs_gap:.4f} | {rel_gap:.2%} | "
            f"{t_mat*1000:.1f} | {t_mf*1000:.1f} | "
            f"{'Y' if bounds_ok else 'N'} |"
        )
    print()


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(all_sweeps, fixtures, output_path):
    """Write detailed markdown report."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = ["# Streaming Cheeger Parameter Sweep Report\n"]
    lines.append(f"Fixtures: {len(fixtures)} Q/K pairs\n")
    lines.append(f"Window sizes: {WINDOW_SIZES}\n")

    # Best per parameter
    for param_name, results in all_sweeps.items():
        lines.append(f"\n## Parameter: `{param_name}`\n")
        lines.append(
            "| Value | σ₂ drift | φ drift | Bracket width | Recompute % | "
            "ms/window | Bounds valid | N |"
        )
        lines.append("|" + "|".join(["---"] * 8) + "|")
        for r in results:
            lines.append(
                f"| {r['value']} | {r['sigma2_drift']:.4f} | {r['phi_drift']:.4f} | "
                f"{r['bracket_width']:.4f} | {r['recompute_rate']:.1%} | "
                f"{r['time_ms']:.1f} | {r['bounds_valid']:.1%} | {r['n_samples']} |"
            )

    # Recommended defaults
    lines.append("\n## Recommended Defaults\n")
    lines.append("```yaml")
    for k, v in DEFAULTS.items():
        lines.append(f"{k}: {v}")
    lines.append("```\n")

    report = "\n".join(lines)
    out.write_text(report)
    print(f"\nReport written to {out}")
    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Streaming Cheeger parameter sweep")
    parser.add_argument(
        "--fixtures-dir", default="benchmarks/fixtures/gpt2_qk",
        help="Directory with GPT-2 Q/K .pt fixtures",
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Use synthetic Q/K instead of GPT-2 fixtures",
    )
    parser.add_argument(
        "--report-path", default="benchmarks/results/cheeger_streaming_report.md",
        help="Output path for the markdown report",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: fewer fixtures and smaller sweep grid",
    )
    args = parser.parse_args()

    # Load or generate fixtures
    if args.synthetic:
        print("Using synthetic Q/K fixtures")
        fixtures = generate_synthetic(n=10)
    else:
        fixtures = load_fixtures(args.fixtures_dir)
        if not fixtures:
            print(
                f"No fixtures found in {args.fixtures_dir}. "
                "Run benchmarks/extract_gpt2_qk.py first, or use --synthetic."
            )
            print("Falling back to synthetic fixtures.\n")
            fixtures = generate_synthetic(n=10)

    print(f"Loaded {len(fixtures)} Q/K fixtures")
    print(f"L range: {min(f['L'] for f in fixtures)}-{max(f['L'] for f in fixtures)}\n")

    # Phase 1: batch vs matrix-free table
    batch_vs_mf_table(fixtures)

    # Phase 2: parameter sweep
    print("# Streaming Parameter Sweep\n")

    if args.quick:
        global SWEEP_GRID
        SWEEP_GRID = {k: v[:2] for k, v in SWEEP_GRID.items()}
        fixtures = fixtures[:5]

    all_sweeps = {}
    for param_name in SWEEP_GRID:
        print(f"Sweeping {param_name}...")
        results = sweep_one_param(param_name, fixtures, causal=True)
        all_sweeps[param_name] = results

        for r in results:
            print(
                f"  {param_name}={r['value']:>6}: "
                f"σ₂_drift={r['sigma2_drift']:.4f}  "
                f"φ_drift={r['phi_drift']:.4f}  "
                f"bracket={r['bracket_width']:.4f}  "
                f"recompute={r['recompute_rate']:.0%}  "
                f"ms={r['time_ms']:.1f}"
            )

    # Phase 3: report
    generate_report(all_sweeps, fixtures, args.report_path)


if __name__ == "__main__":
    main()
