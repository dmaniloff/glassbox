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
  2. Streaming parameter sweep with growing-prefix simulation
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
    for i, L in enumerate([48, 64, 80, 96, 112, 128, 160, 192, 224, 256]):
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
    diag = CheegerDiagnostic(rank=2, threshold=L + 1, causal=causal, mode="batch")
    result = diag.reduce(Q, K, L)
    return result["features"]


# ---------------------------------------------------------------------------
# Streaming simulation: growing prefix (models real decode)
# ---------------------------------------------------------------------------

def simulate_streaming_growing(Q, K, L, seed_size, params, causal=True, n_steps=30):
    """Grow prefix from seed_size to L, simulating autoregressive decode.

    This is the faithful simulation: in real inference, each decode step
    appends one token, so the window grows monotonically. The BRR primitive
    handles the dimension increase via left-aligned zero-padding.

    Args:
        Q, K: Full-sequence tensors [L, D].
        L: Full sequence length.
        seed_size: Initial prefix length (prompt).
        params: Streaming controller params dict.
        causal: Use causal masking.
        n_steps: Target number of decode steps to simulate.
    """
    if seed_size >= L:
        return []

    stride = max(1, (L - seed_size) // n_steps)

    diag = CheegerDiagnostic(
        rank=2, threshold=0, block_size=min(64, seed_size),
        causal=causal, mode="streaming", **params,
    )

    state = None
    results = []
    step = 0

    for w in range(seed_size, L + 1, stride):
        Qw = Q[:w]
        Kw = K[:w]

        t0 = time.perf_counter()
        result = diag.reduce(Qw, Kw, w, prior_state=state)
        t1 = time.perf_counter()

        step += 1
        result["step"] = step
        state = diag.accumulate(result, state)
        f = result["features"]
        results.append({
            "window_size": w,
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
    "gap_threshold": [0.01, 0.05, 0.1, 0.2],
    "degree_shift_threshold": [0.05, 0.1, 0.2, 0.5],
    "geometric_base": [1.2, 1.5, 2.0, 3.0],
    "lanczos_iters": [10, 20, 30, 50],
}

DEFAULTS = {
    "ritz_rank": 3,
    "n_explore": 2,
    "gap_threshold": 0.05,
    "geometric_base": 2.0,
    "lanczos_iters": 20,
    "improved_cheeger_k": 4,
    "degree_shift_threshold": 0.1,
}

SEED_SIZES = [24, 32, 48]


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
            if L < 48:
                continue

            gt = compute_ground_truth(Q, K, L, causal=causal)

            for seed in SEED_SIZES:
                if seed >= L - 4:
                    continue
                stream_results = simulate_streaming_growing(
                    Q, K, L, seed, params, causal=causal, n_steps=30,
                )
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
                        valid = (
                            sr["cheeger_lower"] - 0.01
                            <= sr["phi_star"]
                            <= sr["cheeger_upper"] + 0.01
                        )
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
# Mode comparison: batch vs streaming vs light
# ---------------------------------------------------------------------------

def compare_modes(fixtures, causal=True, n_streaming_steps=30):
    """Head-to-head comparison of batch, streaming, and light modes.

    For each fixture, runs all three modes and reports timing, feature
    availability, and accuracy relative to batch ground truth.
    """
    results = []

    for fix in fixtures:
        Q, K, L = fix["Q"], fix["K"], fix["L"]
        name = fix["name"]

        # --- Batch (ground truth) ---
        diag_batch = CheegerDiagnostic(
            rank=2, threshold=L + 1, causal=causal, mode="batch",
        )
        t0 = time.perf_counter()
        r_batch = diag_batch.reduce(Q, K, L)
        t_batch = time.perf_counter() - t0
        f_batch = r_batch["features"]

        # --- Light ---
        diag_light = CheegerDiagnostic(
            rank=2, threshold=L + 1, causal=causal, mode="light",
        )
        t0 = time.perf_counter()
        r_light = diag_light.reduce(Q, K, L)
        t_light = time.perf_counter() - t0
        f_light = r_light["features"]

        # --- Streaming (growing prefix, amortised cost) ---
        seed_size = max(16, L // 4)
        if seed_size >= L - 2:
            seed_size = max(8, L // 2)
        stride = max(1, (L - seed_size) // n_streaming_steps)

        diag_stream = CheegerDiagnostic(
            rank=2, threshold=L + 1, causal=causal, mode="streaming",
            **DEFAULTS,
        )
        state = None
        stream_times = []
        stream_features = None
        recompute_count = 0
        total_steps = 0
        for w in range(seed_size, L + 1, stride):
            Qw, Kw = Q[:w], K[:w]
            t0 = time.perf_counter()
            r_s = diag_stream.reduce(Qw, Kw, w, prior_state=state)
            t1 = time.perf_counter()
            stream_times.append(t1 - t0)
            r_s["step"] = total_steps
            state = diag_stream.accumulate(r_s, state)
            total_steps += 1
            if r_s.get("recomputed"):
                recompute_count += 1
            stream_features = r_s["features"]

        t_stream_total = sum(stream_times)
        t_stream_amort = t_stream_total / max(total_steps, 1)

        # --- Accuracy vs batch ground truth ---
        sigma2_drift_light = (
            abs(f_batch.sigma2 - f_light.sigma2)
            if f_batch.sigma2 is not None and f_light.sigma2 is not None
            else None
        )
        sigma2_drift_stream = (
            abs(f_batch.sigma2 - stream_features.sigma2)
            if f_batch.sigma2 is not None and stream_features is not None
            and stream_features.sigma2 is not None
            else None
        )
        phi_drift_stream = (
            abs(f_batch.phi_star - stream_features.phi_star)
            if f_batch.phi_star is not None and stream_features is not None
            and stream_features.phi_star is not None
            else None
        )

        # Dual Cheeger agreement
        dual_gap_drift_light = (
            abs(f_batch.dual_gap - f_light.dual_gap)
            if f_batch.dual_gap is not None and f_light.dual_gap is not None
            else None
        )

        results.append({
            "name": name,
            "L": L,
            # Timing
            "batch_ms": t_batch * 1000,
            "light_ms": t_light * 1000,
            "stream_total_ms": t_stream_total * 1000,
            "stream_amort_ms": t_stream_amort * 1000,
            "stream_steps": total_steps,
            "stream_recompute_pct": recompute_count / max(total_steps, 1),
            # Feature availability
            "batch_phi": f_batch.phi_star,
            "batch_improved_upper": f_batch.improved_upper,
            "batch_dual_gap": f_batch.dual_gap,
            "light_phi": f_light.phi_star,  # should be None
            "light_dual_gap": f_light.dual_gap,
            "stream_phi": stream_features.phi_star if stream_features else None,
            "stream_improved_upper": (
                stream_features.improved_upper if stream_features else None
            ),
            "stream_dual_gap": (
                stream_features.dual_gap if stream_features else None
            ),
            # Accuracy
            "sigma2_drift_light": sigma2_drift_light,
            "sigma2_drift_stream": sigma2_drift_stream,
            "phi_drift_stream": phi_drift_stream,
            "dual_gap_drift_light": dual_gap_drift_light,
            # Bracket width
            "batch_bracket": f_batch.bracket_width,
            "light_bracket": f_light.bracket_width,
            "stream_bracket": (
                stream_features.bracket_width if stream_features else None
            ),
        })

    return results


def print_mode_comparison(mode_results):
    """Print mode comparison table to stdout."""
    print("# Mode Comparison: Batch vs Streaming vs Light\n")

    header = (
        "| Name | L | batch_ms | light_ms | stream_amort_ms | "
        "recomp% | σ₂_drift_light | σ₂_drift_stream | φ_drift_stream | "
        "dual_drift_light |"
    )
    sep = "|" + "|".join(["---"] * 10) + "|"
    print(header)
    print(sep)

    for r in mode_results[:20]:
        name = r["name"][:25]

        def fmt(v, prec=4):
            return f"{v:.{prec}f}" if v is not None else "—"

        print(
            f"| {name} | {r['L']} | {r['batch_ms']:.1f} | "
            f"{r['light_ms']:.1f} | {r['stream_amort_ms']:.1f} | "
            f"{r['stream_recompute_pct']:.0%} | "
            f"{fmt(r['sigma2_drift_light'])} | "
            f"{fmt(r['sigma2_drift_stream'])} | "
            f"{fmt(r['phi_drift_stream'])} | "
            f"{fmt(r['dual_gap_drift_light'])} |"
        )

    # Summary
    n = len(mode_results)
    if n > 0:
        def mean_or(key, default=0.0):
            vals = [r[key] for r in mode_results if r[key] is not None]
            return sum(vals) / len(vals) if vals else default

        print()
        print(f"**Summary across {n} fixtures:**")
        print(f"  Batch:     mean {mean_or('batch_ms'):.1f} ms/window")
        print(f"  Light:     mean {mean_or('light_ms'):.1f} ms/window")
        print(
            f"  Streaming: mean {mean_or('stream_amort_ms'):.1f} ms/step "
            f"(amortised), {mean_or('stream_recompute_pct'):.0%} recompute"
        )
        print(f"  σ₂ drift (light vs batch):     {mean_or('sigma2_drift_light'):.4f}")
        print(f"  σ₂ drift (streaming vs batch): {mean_or('sigma2_drift_stream'):.4f}")
        print(f"  φ  drift (streaming vs batch): {mean_or('phi_drift_stream'):.4f}")
        print(f"  dual_gap drift (light vs batch): {mean_or('dual_gap_drift_light'):.4f}")
    print()


# ---------------------------------------------------------------------------
# φ head-to-head: batch vs streaming at each decode step
# ---------------------------------------------------------------------------

def phi_head_to_head(fixtures, causal=True, n_steps=20, max_fixtures=5):
    """Per-step comparison of φ* from fresh batch vs streaming at each prefix.

    For each fixture, grows the prefix from seed_size to L. At each step:
      - batch: fresh CheegerDiagnostic(mode="batch") on the current prefix
      - streaming: CheegerDiagnostic(mode="streaming") with carried state

    This isolates the φ accuracy question: how well does the streaming
    controller track the true φ* as the window grows?
    """
    results = []

    for fix in fixtures[:max_fixtures]:
        Q, K, L = fix["Q"], fix["K"], fix["L"]
        name = fix["name"]

        seed_size = max(16, L // 4)
        if seed_size >= L - 4:
            seed_size = max(8, L // 2)
        stride = max(1, (L - seed_size) // n_steps)

        diag_batch = CheegerDiagnostic(
            rank=2, threshold=L + 1, causal=causal, mode="batch",
        )
        diag_stream = CheegerDiagnostic(
            rank=2, threshold=L + 1, causal=causal, mode="streaming",
            **DEFAULTS,
        )
        state = None
        steps = []
        step_idx = 0

        for w in range(seed_size, L + 1, stride):
            Qw, Kw = Q[:w], K[:w]

            r_batch = diag_batch.reduce(Qw, Kw, w)
            r_stream = diag_stream.reduce(Qw, Kw, w, prior_state=state)
            r_stream["step"] = step_idx
            state = diag_stream.accumulate(r_stream, state)
            step_idx += 1

            f_b = r_batch["features"]
            f_s = r_stream["features"]

            steps.append({
                "prefix": w,
                "phi_batch": f_b.phi_star,
                "phi_stream": f_s.phi_star,
                "sigma2_batch": f_b.sigma2,
                "sigma2_stream": f_s.sigma2,
                "bracket_batch": f_b.bracket_width,
                "bracket_stream": f_s.bracket_width,
                "tier": r_stream.get("tier"),
                "recomputed": r_stream.get("recomputed", False),
            })

        results.append({"name": name, "L": L, "steps": steps})

    return results


def print_phi_head_to_head(phi_results):
    """Print φ head-to-head tables to stdout."""
    print("# φ Head-to-Head: Batch vs Streaming per Decode Step\n")

    all_deltas = []
    for entry in phi_results:
        print(f"## {entry['name']} (L={entry['L']})\n")
        print("| prefix | φ batch | φ stream | |Δφ| | tier |")
        print("|" + "|".join(["---"] * 5) + "|")

        for step in entry["steps"]:
            phi_b = step["phi_batch"]
            phi_s = step["phi_stream"]
            delta = abs(phi_b - phi_s) if phi_b is not None and phi_s is not None else None
            tier = step.get("tier", "—")

            def fmt(v):
                return f"{v:.4f}" if v is not None else "—"

            print(f"| {step['prefix']} | {fmt(phi_b)} | {fmt(phi_s)} | {fmt(delta)} | {tier} |")
            if delta is not None:
                all_deltas.append(delta)

        deltas = [
            abs(s["phi_batch"] - s["phi_stream"])
            for s in entry["steps"]
            if s["phi_batch"] is not None and s["phi_stream"] is not None
        ]
        if deltas:
            print(f"\nMean |Δφ|: {sum(deltas)/len(deltas):.4f}, max: {max(deltas):.4f}\n")

    if all_deltas:
        print(
            f"**Overall φ fidelity ({len(all_deltas)} steps): "
            f"mean |Δφ|={sum(all_deltas)/len(all_deltas):.4f}, "
            f"max={max(all_deltas):.4f}**\n"
        )


# ---------------------------------------------------------------------------
# Adaptive KLGT vs fixed-k comparison
# ---------------------------------------------------------------------------

def compare_klgt(fixtures, causal=True):
    """Compare adaptive vs fixed-k KLGT upper bounds."""
    from glassbox.cheeger import compute_improved_cheeger_upper
    from glassbox.svd import compute_degree_normalized_M

    results = []
    for fix in fixtures:
        Q, K, L = fix["Q"], fix["K"], fix["L"]
        scale = 1.0 / (Q.shape[1] ** 0.5)
        scores = Q @ K.T * scale
        if causal:
            scores = scores.masked_fill(
                ~torch.tril(torch.ones(L, L, dtype=torch.bool, device=scores.device)),
                float("-inf"),
            )
        A = torch.softmax(scores, dim=-1)
        M, _, _ = compute_degree_normalized_M(A)
        M_sym = (M + M.T) / 2
        eigvals = torch.linalg.eigvalsh(M_sym).flip(0)

        fixed_4 = compute_improved_cheeger_upper(eigvals, k=4, adaptive=False)
        fixed_6 = compute_improved_cheeger_upper(eigvals, k=6, adaptive=False)
        adaptive = compute_improved_cheeger_upper(eigvals, adaptive=True)
        naive = (
            (2.0 * max(1.0 - float(eigvals[1]), 0.0)) ** 0.5
            if len(eigvals) > 1 else None
        )

        results.append({
            "name": fix["name"],
            "L": L,
            "naive_upper": naive,
            "klgt_k4": fixed_4,
            "klgt_k6": fixed_6,
            "klgt_adaptive": adaptive,
        })

    return results


def print_klgt_comparison(klgt_results):
    """Print KLGT comparison table to stdout."""
    print("# KLGT Upper Bound: Naive vs Fixed-k vs Adaptive\n")

    header = (
        "| Name | L | √(2μ₂) | KLGT k=4 | KLGT k=6 | KLGT adaptive | "
        "adaptive improvement |"
    )
    sep = "|" + "|".join(["---"] * 7) + "|"
    print(header)
    print(sep)

    for r in klgt_results[:20]:
        name = r["name"][:25]

        def fmt(v):
            return f"{v:.4f}" if v is not None else "—"

        improvement = ""
        if r["naive_upper"] is not None and r["klgt_adaptive"] is not None:
            if r["naive_upper"] > 1e-8:
                pct = (1.0 - r["klgt_adaptive"] / r["naive_upper"]) * 100
                improvement = f"{pct:.1f}%"

        print(
            f"| {name} | {r['L']} | {fmt(r['naive_upper'])} | "
            f"{fmt(r['klgt_k4'])} | {fmt(r['klgt_k6'])} | "
            f"{fmt(r['klgt_adaptive'])} | {improvement} |"
        )

    n = len(klgt_results)
    if n > 0:
        improvements = []
        for r in klgt_results:
            if (r["naive_upper"] is not None and r["klgt_adaptive"] is not None
                    and r["naive_upper"] > 1e-8):
                improvements.append(1.0 - r["klgt_adaptive"] / r["naive_upper"])
        if improvements:
            mean_imp = sum(improvements) / len(improvements)
            print(
                f"\n**Mean adaptive improvement over √(2μ₂): "
                f"{mean_imp*100:.1f}%** ({n} fixtures)"
            )
    print()


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def find_pareto_front(all_sweeps):
    """Identify Pareto-optimal settings: lowest recompute rate for given drift."""
    candidates = []
    for param_name, results in all_sweeps.items():
        for r in results:
            candidates.append({
                "setting": f"{param_name}={r['value']}",
                "sigma2_drift": r["sigma2_drift"],
                "phi_drift": r["phi_drift"],
                "recompute_rate": r["recompute_rate"],
                "bracket_width": r["bracket_width"],
                "time_ms": r["time_ms"],
                "bounds_valid": r["bounds_valid"],
            })
    candidates.sort(key=lambda c: (c["recompute_rate"], c["sigma2_drift"]))
    return candidates[:10]


def generate_report(
    all_sweeps, fixtures, output_path,
    mode_results=None, klgt_results=None, phi_head2head=None,
):
    """Write detailed markdown report."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = ["# Cheeger Diagnostic Benchmark Report\n"]
    lines.append(f"Fixtures: {len(fixtures)} Q/K pairs\n")

    # --- Mode comparison ---
    if mode_results:
        lines.append("\n## Mode Comparison: Batch vs Streaming vs Light\n")
        lines.append(
            "| Name | L | batch ms | light ms | stream amort ms | "
            "recomp% | σ₂ drift (light) | σ₂ drift (stream) | "
            "φ drift (stream) | dual gap drift (light) |"
        )
        lines.append("|" + "|".join(["---"] * 10) + "|")

        def _fmt(v, prec=4):
            return f"{v:.{prec}f}" if v is not None else "—"

        for r in mode_results:
            lines.append(
                f"| {r['name'][:25]} | {r['L']} | "
                f"{r['batch_ms']:.1f} | {r['light_ms']:.1f} | "
                f"{r['stream_amort_ms']:.1f} | "
                f"{r['stream_recompute_pct']:.0%} | "
                f"{_fmt(r['sigma2_drift_light'])} | "
                f"{_fmt(r['sigma2_drift_stream'])} | "
                f"{_fmt(r['phi_drift_stream'])} | "
                f"{_fmt(r['dual_gap_drift_light'])} |"
            )

        n = len(mode_results)
        def _mean_or(key, default=0.0):
            vals = [r[key] for r in mode_results if r[key] is not None]
            return sum(vals) / len(vals) if vals else default

        lines.append("")
        lines.append(f"**Summary ({n} fixtures):**")
        lines.append(f"- Batch: mean {_mean_or('batch_ms'):.1f} ms/window")
        lines.append(f"- Light: mean {_mean_or('light_ms'):.1f} ms/window "
                      f"({_mean_or('light_ms')/_mean_or('batch_ms', 1.0):.1f}x faster)")
        lines.append(
            f"- Streaming: mean {_mean_or('stream_amort_ms'):.1f} ms/step amortised, "
            f"{_mean_or('stream_recompute_pct'):.0%} full recompute rate"
        )
        lines.append(f"- σ₂ drift (light vs batch): {_mean_or('sigma2_drift_light'):.4f}")
        lines.append(f"- σ₂ drift (streaming vs batch): {_mean_or('sigma2_drift_stream'):.4f}")
        lines.append(f"- φ drift (streaming vs batch): {_mean_or('phi_drift_stream'):.4f}")
        lines.append(f"- dual_gap drift (light vs batch): {_mean_or('dual_gap_drift_light'):.4f}")

    # --- φ head-to-head ---
    if phi_head2head:
        lines.append("\n## φ Head-to-Head: Batch vs Streaming per Decode Step\n")
        lines.append(
            "Shows φ* from batch (ground truth at each prefix length) vs streaming "
            "φ* at the same prefix. Streaming reuses prior state; batch recomputes fresh.\n"
        )

        for entry in phi_head2head:
            lines.append(f"### {entry['name']} (L={entry['L']})\n")
            lines.append(
                "| prefix | φ batch | φ stream | |Δφ| | stream tier |"
            )
            lines.append("|" + "|".join(["---"] * 5) + "|")
            for step in entry["steps"]:
                phi_b = step["phi_batch"]
                phi_s = step["phi_stream"]
                delta = abs(phi_b - phi_s) if phi_b is not None and phi_s is not None else None
                tier = step.get("tier", "—")
                lines.append(
                    f"| {step['prefix']} | "
                    f"{phi_b:.4f}" if phi_b is not None else "—"
                )
                lines[-1] = (
                    f"| {step['prefix']} | "
                    f"{_fmt(phi_b)} | {_fmt(phi_s)} | {_fmt(delta)} | {tier} |"
                )
            # per-fixture summary
            deltas = [
                abs(s["phi_batch"] - s["phi_stream"])
                for s in entry["steps"]
                if s["phi_batch"] is not None and s["phi_stream"] is not None
            ]
            if deltas:
                lines.append(
                    f"\nMean |Δφ|: {sum(deltas)/len(deltas):.4f}, "
                    f"max: {max(deltas):.4f}\n"
                )

    # --- KLGT comparison ---
    if klgt_results:
        lines.append("\n## KLGT Upper Bound: √(2μ₂) vs Fixed-k vs Adaptive\n")
        lines.append(
            "| Name | L | √(2μ₂) | KLGT k=4 | KLGT k=6 | KLGT adaptive | improvement |"
        )
        lines.append("|" + "|".join(["---"] * 7) + "|")
        for r in klgt_results:
            imp = ""
            if r["naive_upper"] is not None and r["klgt_adaptive"] is not None:
                if r["naive_upper"] > 1e-8:
                    imp = f"{(1.0 - r['klgt_adaptive']/r['naive_upper'])*100:.1f}%"
            lines.append(
                f"| {r['name'][:25]} | {r['L']} | "
                f"{_fmt(r['naive_upper'])} | {_fmt(r['klgt_k4'])} | "
                f"{_fmt(r['klgt_k6'])} | {_fmt(r['klgt_adaptive'])} | {imp} |"
            )

        improvements = [
            1.0 - r["klgt_adaptive"] / r["naive_upper"]
            for r in klgt_results
            if r["naive_upper"] is not None and r["klgt_adaptive"] is not None
            and r["naive_upper"] > 1e-8
        ]
        if improvements:
            lines.append(
                f"\n**Mean adaptive improvement over √(2μ₂): "
                f"{sum(improvements)/len(improvements)*100:.1f}%**"
            )

    # --- Streaming parameter sweep ---
    if all_sweeps:
        lines.append("\n---\n")
        lines.append("## Streaming Parameter Sweep\n")
        lines.append(f"Seed sizes: {SEED_SIZES}\n")
        lines.append("Simulation: growing prefix (models autoregressive decode)\n")

        pareto = find_pareto_front(all_sweeps)
        if pareto:
            lines.append("\n### Pareto Front (recompute rate vs fidelity)\n")
            lines.append(
                "| Setting | σ₂ drift | φ drift | Recompute % | "
                "Bracket width | ms/window | Bounds valid |"
            )
            lines.append("|" + "|".join(["---"] * 7) + "|")
            for c in pareto:
                lines.append(
                    f"| {c['setting']} | {c['sigma2_drift']:.4f} | {c['phi_drift']:.4f} | "
                    f"{c['recompute_rate']:.1%} | {c['bracket_width']:.4f} | "
                    f"{c['time_ms']:.1f} | {c['bounds_valid']:.1%} |"
                )

        for param_name, results in all_sweeps.items():
            lines.append(f"\n### Parameter: `{param_name}`\n")
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
    lines.append("Based on Pareto analysis of growing-prefix simulation:\n")
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
    parser.add_argument(
        "--max-fixtures", type=int, default=0,
        help="Subsample to at most N fixtures (0 = all)",
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

    if args.max_fixtures > 0 and len(fixtures) > args.max_fixtures:
        import random
        random.seed(42)
        fixtures = random.sample(fixtures, args.max_fixtures)

    print(f"Loaded {len(fixtures)} Q/K fixtures")
    print(f"L range: {min(f['L'] for f in fixtures)}-{max(f['L'] for f in fixtures)}\n")

    # Phase 1: batch vs matrix-free table
    batch_vs_mf_table(fixtures)

    # Phase 2: mode comparison (batch vs streaming vs light)
    print("# Mode Comparison: Batch vs Streaming vs Light\n")
    mode_results = compare_modes(fixtures, causal=True)
    print_mode_comparison(mode_results)

    # Phase 3: φ head-to-head (batch vs streaming at each decode step)
    print("# φ Head-to-Head: Batch vs Streaming per Decode Step\n")
    phi_h2h = phi_head_to_head(
        fixtures, causal=True, n_steps=20,
        max_fixtures=min(5, len(fixtures)),
    )
    print_phi_head_to_head(phi_h2h)

    # Phase 4: KLGT adaptive vs fixed-k
    print("# KLGT Adaptive vs Fixed-k\n")
    klgt_results = compare_klgt(fixtures, causal=True)
    print_klgt_comparison(klgt_results)

    # Phase 5: streaming parameter sweep
    print("# Streaming Parameter Sweep (growing prefix)\n")

    if args.quick:
        global SWEEP_GRID, SEED_SIZES
        SWEEP_GRID = {k: v[:2] for k, v in SWEEP_GRID.items()}
        SEED_SIZES = [32]
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

    # Phase 6: report
    generate_report(
        all_sweeps, fixtures, args.report_path,
        mode_results=mode_results,
        klgt_results=klgt_results,
        phi_head2head=phi_h2h,
    )


if __name__ == "__main__":
    main()
