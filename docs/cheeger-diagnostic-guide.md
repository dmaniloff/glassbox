# Cheeger Diagnostic: User Guide

The Cheeger diagnostic detects attention bottlenecks — near-rank-1 regimes where
almost all attention mass flows through a single mode. It operates on the
degree-normalized operator `M = D_Q^{-1/2} A D_K^{-1/2}` and emits a
**Cheeger bracket**: a nested interval bounding the true bipartite conductance
`phi*`.

This guide covers how to enable the diagnostic, choose the right mode for your
workload, interpret the output, and tune the streaming controller.

---

## Quick Start

### YAML configuration

```yaml
# glassbox.yaml
cheeger:
  enabled: true
  mode: batch       # batch | streaming | light
  interval: 32
  heads: [0, 1, 2]
  causal: true
```

### Programmatic

```python
from glassbox.config import GlassboxConfig

cfg = GlassboxConfig(cheeger={
    "enabled": True,
    "mode": "batch",
    "interval": 32,
    "heads": [0, 1, 2],
})
```

### CLI

```bash
glassbox run --signal cheeger --interval 32 --output-path signals.jsonl model.py
```

---

## Understanding the Bracket

The Cheeger inequality bounds the true bipartite conductance `phi*` using the
spectral gap of `M_sym = (M + M^T) / 2`:

```
mu2 / 2  <=  phi*  <=  sqrt(2 * mu2)
```

where `mu2 = 1 - lambda2(M_sym)` is the spectral gap (second-largest eigenvalue
of the symmetric part). A small `phi*` means attention routes through a narrow
bottleneck; a large `phi*` means diverse, well-connected routing.

The bracket gives you a **certificate** without computing the exact conductance:

- **Lower bound** `cheeger_lower = mu2 / 2`: rules a bottleneck *out*. If
  `cheeger_lower > threshold`, attention is definitively well-connected.
- **Upper bound** `cheeger_upper = sqrt(2*mu2)`: rules a bottleneck *in*. If
  `cheeger_upper < threshold`, attention is definitively bottlenecked.
- When both bounds are on the same side of your threshold, you have a decision
  without needing the exact `phi*`.

### The three tiers

| Tier | What it needs | What it gives | When to use |
|------|---------------|---------------|-------------|
| **1 (always-on)** | `lambda2(M_sym)` only | `cheeger_lower`, `cheeger_upper` | High-frequency monitoring; cheapest |
| **2 (gap-guarded)** | Fiedler eigenvector + sweep | `phi_star` (exact sweep), `phi_hat` (tight upper) | When you need the exact conductance |
| **3 (fallback)** | Higher-order eigenvalues | `improved_upper` via adaptive KLGT | When spectral gap is small and tier 2 is unreliable |

**Why the upper bound is the detection-relevant side.** A closed aperture or
injection sink manifests as a low-conductance bottleneck. The lower bound
certifies conductance is *large* (rules bottleneck out); the upper bound
certifies it is *small* (fires the alarm). The failure signal lives in the
upper tail.

---

## Choosing a Mode

### Decision tree

```
Need exact phi* for offline analysis?
  YES --> batch

Unbounded autoregressive decode, need per-token updates?
  YES --> streaming

Just want spectral health at minimal cost?
  YES --> light
```

### Mode comparison (GPT-2 small, 50 attention heads, L=73-112)

| Property | batch | streaming | light |
|----------|-------|-----------|-------|
| **Cost** | 1.3 ms/window (mean) | 2.1 ms/step (amortised) | 0.5 ms/window |
| **Speedup vs batch** | 1x | 0.6x amortised | 2.6x |
| **Features** | All 13 fields | All 13 (phi_star stale between recomputes) | sigma2, bounds only |
| **phi_star** | Exact every window | Carried forward, refreshed on trigger | Not available |
| **sigma2 drift** | 0 (ground truth) | 0.045 mean | 0 (exact match) |
| **phi drift** | 0 (ground truth) | 0.017 mean | N/A |
| **Dual Cheeger** | Exact every window | lambda_min carried forward | Exact every window |
| **Recompute rate** | 100% (by definition) | 35% (GPT-2), 18% (synthetic) | N/A |

### `batch` (default)

Full SVD + bipartite sweep every window. Produces exact `phi_star`, `sigma2`,
the full bracket, dual Cheeger, and the adaptive KLGT bound.

**Use for:** offline analysis, training data collection, low-frequency
monitoring where `interval` spacing amortises the O(L^2) cost.

```yaml
cheeger:
  enabled: true
  mode: batch
  interval: 64    # fire every 64 tokens
  rank: 2
  threshold: 512  # L <= 512: materialized; above: matrix-free
```

### `streaming`

Bordered Rayleigh-Ritz (BRR) amortisation between full Lanczos recomputes. The
controller carries a Ritz basis across decode steps and triggers full recompute
on three conditions:

1. **Spectral gap collapse** — `lambda2 - lambda3 < gap_threshold`
2. **Degree shift** — mean degree-ratio change exceeds `degree_shift_threshold`
3. **Geometric stride** — periodic forced recompute (grows geometrically)

Between recomputes, the cheap BRR path updates the bracket eigenvalues in ~5
matvec calls (vs ~20-50 for full Lanczos). The `phi_star` value is stale between
recomputes but the spectral bounds `cheeger_lower/upper` track live.

**Use for:** unbounded autoregressive decode where the window grows per token.

```yaml
cheeger:
  enabled: true
  mode: streaming
  interval: 1          # fire every token (cheap BRR most of the time)
  gap_threshold: 0.05
  degree_shift_threshold: 0.1
  geometric_base: 2.0
  ritz_rank: 3
  n_explore: 2
  lanczos_iters: 20
```

### `light`

Quick `lambda2(M_sym)` estimation via a single eigenvalue call (Lanczos k=2 for
large L, `eigvalsh` for small L). Produces tier-1 gap-free bounds and the dual
Cheeger bracket. No sweep cut, no `phi_star`.

**Use for:** high-frequency spectral health checks where you only need to know
whether the bracket is wide or narrow.

```yaml
cheeger:
  enabled: true
  mode: light
  interval: 8   # can fire frequently since cost is low
```

---

## Output Fields

Every Cheeger snapshot emits a `CheegerFeatures` object with up to 13 fields.

### Core bracket

| Field | Formula | Available in | Meaning |
|-------|---------|--------------|---------|
| `sigma2` | `lambda2(M_sym)` | all modes | Second-largest eigenvalue of `M_sym`. Spectral gap measure |
| `cheeger_lower` | `(1 - sigma2) / 2` | all modes | Lower bound on `phi*` |
| `cheeger_upper` | `sqrt(2 * (1 - sigma2))` | all modes | Upper bound on `phi*` |
| `bracket_width` | `cheeger_upper - cheeger_lower` | all modes | Confidence signal: narrow = informative, wide = uncertain |
| `phi_star` | Bipartite sweep | batch, streaming | Exact sweep conductance. The tight upper bound on `phi*` |
| `phi_hat` | Same as `phi_star` when gap-healthy | streaming | `None` when gap collapses and sweep is unreliable |
| `improved_upper` | `min_k k * mu2 / sqrt(mu_{k+1})` | batch, streaming | Adaptive KLGT bound. Tighter than `sqrt(2*mu2)` when higher eigenvalues are available |

### Streaming metadata

| Field | Available in | Meaning |
|-------|--------------|---------|
| `spectral_gap` | streaming | `lambda2 - lambda3` from M_sym. Gap health for Fiedler vector stability |
| `recomputed` | streaming | `true` if a full Lanczos recompute happened this step |

### Dual Cheeger (bipartiteness)

| Field | Formula | Available in | Meaning |
|-------|---------|--------------|---------|
| `lambda_min` | `lambda_min(M_sym)` | all modes | Smallest eigenvalue of `M_sym` |
| `dual_gap` | `1 + lambda_min` | all modes | 0 = perfectly bipartite routing |
| `dual_cheeger_lower` | `dual_gap / 2` | all modes | Lower bound on dual Cheeger constant beta |
| `dual_cheeger_upper` | `sqrt(2 * dual_gap)` | all modes | Upper bound on beta |

**Interpreting dual Cheeger:** A small `dual_gap` means the attention graph is
nearly bipartite — tokens split into two groups with most attention flowing
*between* groups rather than *within*. This can indicate adversarial prompt
structures that force attention into a ping-pong pattern.

---

## Interpreting Results

### Reading the bracket

```
Features: phi_star=0.12, sigma2=0.89, cheeger_lower=0.05, cheeger_upper=0.47
```

This means:
- `mu2 = 1 - 0.89 = 0.11` (small spectral gap)
- The true conductance `phi*` lies in `[0.05, 0.47]`
- The exact sweep found `phi_star = 0.12`, close to the lower bound
- This head has a **moderate bottleneck** — attention concentrates but not completely

### Thresholding for detection

```python
features = snapshot.features  # CheegerFeatures

# Conservative: only alert if the upper bound certifies low conductance
if features.cheeger_upper < 0.15:
    alert("Confirmed bottleneck — upper bound certifies phi* < 0.15")

# Aggressive: alert if the sweep cut finds low conductance
elif features.phi_star is not None and features.phi_star < 0.10:
    alert("Likely bottleneck — sweep found phi* < 0.10")

# Use bracket width as confidence
if features.bracket_width > 0.6:
    log("Wide bracket — spectral gap is small, consider full recompute")
```

### Tracking over decode steps (streaming)

In streaming mode, watch for:

1. **Bracket width expanding** — spectral gap is collapsing, Fiedler vector
   losing stability. The controller will auto-trigger a recompute.
2. **phi_star going stale** — between recomputes, `phi_star` holds the last
   sweep value. Compare with `cheeger_lower/upper` to gauge staleness.
3. **Recompute bursts** — many consecutive `recomputed=True` steps indicate
   rapidly shifting attention structure. Consider lowering
   `degree_shift_threshold` or `geometric_base` for tighter tracking.

---

## Streaming Controller Tuning

### Parameter reference

| Parameter | Default | Effect of increasing | Effect of decreasing |
|-----------|---------|---------------------|---------------------|
| `gap_threshold` | 0.05 | More recomputes, better accuracy | Fewer recomputes, more drift |
| `degree_shift_threshold` | 0.1 | Fewer recomputes (less sensitive to structural change) | More recomputes (catches smaller shifts) |
| `geometric_base` | 2.0 | Longer between forced recomputes | Shorter between forced recomputes |
| `ritz_rank` | 3 | More eigenvalues tracked (marginal benefit above 3) | Cheaper BRR but less spectral coverage |
| `n_explore` | 2 | BRR explores more directions (marginal benefit above 2) | Cheaper BRR but slower convergence |
| `lanczos_iters` | 20 | More accurate full recompute (diminishing above 20) | Faster recompute but less precise |

### Benchmark-informed recommendations

Parameter sensitivity measured on 50 GPT-2 attention heads (5 prompts x 12
layers x ~1 head, L=73-112), sweeping each parameter while holding others at
defaults.

**`gap_threshold`** — strongest effect on recompute rate:

| Value | Recompute % | sigma2 drift | phi drift | ms/step |
|-------|-------------|-------------|-----------|---------|
| 0.01 | 15% | 0.192 | 0.039 | 3.0 |
| 0.05 | 39% | 0.154 | 0.038 | 5.5 |
| 0.10 | 68% | 0.108 | 0.038 | 8.7 |
| 0.20 | 96% | 0.055 | 0.036 | 11.9 |

At 0.05 (default), the controller recomputes ~39% of steps on real attention
patterns, achieving a good balance between drift and cost. Lowering to 0.01
saves compute but allows sigma2 to drift by ~0.19. Raising to 0.10 nearly
doubles cost for only marginal phi improvement.

**`geometric_base`** — second strongest lever:

| Value | Recompute % | sigma2 drift | phi drift | ms/step |
|-------|-------------|-------------|-----------|---------|
| 1.2 | 45% | 0.138 | 0.037 | 6.4 |
| 1.5 | 42% | 0.146 | 0.037 | 5.8 |
| 2.0 | 39% | 0.154 | 0.038 | 5.5 |
| 3.0 | 35% | 0.165 | 0.041 | 5.1 |

The default 2.0 provides a good compromise. Lower values (1.2) force more
frequent recomputes and reduce drift. Higher values (3.0) are cheaper but allow
more stale phi_star values.

**`ritz_rank` and `n_explore`** — minimal sensitivity:

Both parameters show minimal impact on accuracy or cost. The defaults
(`ritz_rank=3`, `n_explore=2`) sit in the sweet spot. Increasing beyond these
values adds compute without measurably improving tracking.

**`lanczos_iters`** — diminishing returns above 20:

| Value | Recompute % | sigma2 drift | ms/step |
|-------|-------------|-------------|---------|
| 10 | 21% | 0.186 | 3.2 |
| 20 | 39% | 0.155 | 5.6 |
| 30 | 40% | 0.154 | 8.4 |
| 50 | 40% | 0.153 | 10.9 |

Going from 10 to 20 iterations halves sigma2 drift. Going from 20 to 50
provides negligible improvement at 2x the cost. The default of 20 is well-placed.

### Recommended configurations

**Low-latency (inference guard rails):**
```yaml
cheeger:
  mode: streaming
  gap_threshold: 0.01
  geometric_base: 3.0
  lanczos_iters: 10
  # ~15% recompute rate, ~3 ms/step, sigma2 drift ~0.19
```

**Balanced (default):**
```yaml
cheeger:
  mode: streaming
  gap_threshold: 0.05
  geometric_base: 2.0
  lanczos_iters: 20
  # ~39% recompute rate, ~5.5 ms/step, sigma2 drift ~0.15
```

**High-fidelity (research/debugging):**
```yaml
cheeger:
  mode: streaming
  gap_threshold: 0.1
  geometric_base: 1.5
  lanczos_iters: 30
  # ~68% recompute rate, ~8.5 ms/step, sigma2 drift ~0.11
```

---

## Phi Head-to-Head: How Accurate is Streaming?

The key accuracy question: how well does the streaming controller's `phi_star`
track the true per-prefix `phi*`?

Measured on GPT-2 attention heads with growing-prefix simulation (models
autoregressive decode):

| Property | Value |
|----------|-------|
| Mean \|delta phi\| | 0.017 |
| Max \|delta phi\| | 0.07 |
| Recompute rate | 35% of steps |

At recompute steps (`tier=materialized`), streaming phi matches batch phi
exactly. Between recomputes (`tier=bordered_ritz`), phi_star is the stale value
from the last full sweep. The stale value drifts by 0.02-0.04 on average, with
occasional spikes to 0.07 when the attention pattern shifts abruptly.

**Practical impact:** For bottleneck detection with a threshold around 0.10-0.15,
a drift of 0.02-0.04 means the streaming mode may delay detection by a few
tokens compared to batch. For monitoring dashboards and trend analysis, this
drift is negligible.

---

## KLGT Adaptive Upper Bound

The adaptive KLGT bound uses higher-order eigenvalues to provide a tighter
upper bound than the classical `sqrt(2*mu2)`. It searches over all available k
to minimise `k * mu2 / sqrt(mu_{k+1})`.

Measured on 50 GPT-2 attention heads:

| Property | Value |
|----------|-------|
| Mean improvement over sqrt(2*mu2) | 28% tighter |
| Range | 0.1% to 39.5% tighter |

The adaptive bound is always at least as tight as the classical bound and
often substantially tighter, especially in heads with rich spectral structure
(many distinct eigenvalues). It is computed automatically in `batch` and
`streaming` modes at no extra cost (reuses eigenvalues already computed for
the bracket).

---

## Materialized vs Matrix-Free

The `threshold` parameter controls whether the diagnostic materialises the
full L x L attention matrix or uses matrix-free matvec operations:

- `L <= threshold`: materialised path (dense `eigvalsh`, `svd`)
- `L > threshold`: matrix-free path (Lanczos, randomised SVD)

Measured on synthetic Q/K pairs:

| L | Batch (materialized) ms | Batch (matrix-free) ms | Light ms |
|---|------------------------|----------------------|---------|
| 48 | 0.8 | — | 0.2 |
| 128 | 2.1 | — | 0.5 |
| 256 | 8.8 | — | 2.3 |

The crossover point where matrix-free becomes faster depends on hardware.
On CPU (Apple Silicon), the default `threshold=512` is reasonable. On GPU
with large batch sizes, lower to 256 or 0 (always matrix-free) to avoid L x L
allocations.

Both paths produce identical bracket bounds (confirmed by the benchmark:
sigma2 drift between materialized and matrix-free is zero for light mode).
The sweep conductance `phi_star` may differ slightly between paths due to
numerical differences in the Fiedler vector computation.

---

## Running the Benchmarks

Reproduce the numbers in this guide:

```bash
# Synthetic data (no dependencies)
.venv/bin/python benchmarks/bench_cheeger.py --synthetic

# GPT-2 attention (requires transformers)
pip install transformers
python benchmarks/extract_gpt2_qk.py          # extract Q/K fixtures
.venv/bin/python benchmarks/bench_cheeger.py   # run full sweep

# Quick mode (2-3 minutes instead of 10+)
.venv/bin/python benchmarks/bench_cheeger.py --synthetic --quick

# Control output location
.venv/bin/python benchmarks/bench_cheeger.py --report-path results/my_report.md
```

Reports are written to `benchmarks/results/cheeger_streaming_report.md` (synthetic)
or `benchmarks/results/cheeger_gpt2_report.md` (GPT-2).

---

## Implementation Reference

| Component | Location | Purpose |
|-----------|----------|---------|
| Sweep + bounds | `glassbox/cheeger.py` | Bipartite sweep conductance, KLGT bound, dual Cheeger |
| Three-mode controller | `glassbox/diagnostics/cheeger.py` | batch/streaming/light dispatch, trigger logic |
| BRR + Lanczos | `glassbox/svd.py` | `bordered_rayleigh_ritz`, `hermitian_lanczos`, `matvec_Msym_blocked` |
| Config | `glassbox/config.py` | `CheegerConfig` with all parameters |
| Results model | `glassbox/results.py` | `CheegerFeatures` (13 fields) |
| Tests | `tests/test_cheeger*.py`, `tests/test_streaming_cheeger.py` | 123 tests covering all modes |
| Benchmark | `benchmarks/bench_cheeger.py` | Mode comparison, phi head-to-head, parameter sweep |
