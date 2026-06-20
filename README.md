# glassbox

*Grab vLLM's attention.*

`glassbox` is a vLLM plugin for extracting model internals during inference and turning them into compact, structured signals for downstream reliability systems. These signals include spectral features and flow-based features that provide a routing-oriented view of model behavior.

The main use case is online or offline analysis of failure modes in LLM generation: hallucination detection, task drift detection, uncertainty quantification, routing analysis, and other forms of model-behavior monitoring.

The primary implementation is a custom vLLM attention backend in `glassbox/backends/`. There is also an experimental `torch.compile` / FX instrumentation path in `glassbox/passes/`, but the custom backend is the working path.

## What It Extracts

At configurable intervals during inference, `glassbox` computes features from different stages of the attention computation:

1. **spectral** — SVD of the pre-softmax scores matrix `S = QK^T`
2. **routing** — SVD + Hodge decomposition of the degree-normalized post-softmax matrix `M = D_Q^{-1/2} A D_K^{-1/2}` (Dahlem et al., upcoming)
3. **tracker** — Span-independent features from the raw post-softmax attention matrix `A = softmax(QK^T / sqrt(d))` — [AttentionTracker](https://arxiv.org/abs/2411.00348) (arXiv:2411.00348)
4. **selfattn** — Self-attention features from the diagonal of `A` — [LLM-Check](https://github.com/GaurangSriramanan/LLM_Check_Hallucination_Detection) (NeurIPS 2024)
5. **laplacian** — Spectral features from the in-degree graph Laplacian `L = D_in - A` — [LapEigvals](https://github.com/graphml-lab-pwr/lapeigvals) (EMNLP 2025, [arXiv:2502.17598](https://arxiv.org/abs/2502.17598))
6. **cheeger** — Bipartite sweep conductance from Fiedler vectors of `M`, with three configurable modes (batch, streaming, light) — Dahlem et al. (upcoming)

For each tracked `(request, layer, head, step)`, it emits a JSONL record with:

- request metadata
- layer and head identifiers
- sequence length `L`
- top singular values
- derived spectral features
- optional routing / Hodge-style features for the normalized operator
- optional attention tracker features for the raw attention matrix

Those snapshots are represented by `SVDSnapshot` in `glassbox/results.py`.

## Why This Exists

Transformer internals can reveal a great deal about model behavior. They are useful for monitoring, debugging, and failure analysis, but most of the underlying objects are too large to inspect directly in a practical inference setting.

Raw activations and full attention matrices are expensive to retain, and modern attention systems are specifically engineered to avoid materializing the full `L x L` object during efficient inference. Because of that, many tools for inspecting transformer internals live in research harnesses around HuggingFace models rather than in production-grade inference stacks.

That creates a gap between interpretability results in papers and practical deployment in real systems. `glassbox` is built to close that gap by efficiently extracting compact signals of how attention is behaving:

- Is one mode dominating, or are multiple modes active?
- Is routing bottlenecked through a narrow channel?
- Is behavior becoming more asymmetric or circulatory over time?
- Do certain layers or heads shift sharply when the model starts to drift or hallucinate?


## How It Integrates With vLLM

The package registers itself through vLLM's plugin entrypoint and exposes a `CUSTOM` attention backend:

- Entry point: `glassbox.vllm_plugin:register_svd_backend`
- Backend: `glassbox.backends.svd_backend.SVDTritonAttentionBackend`

At runtime, the backend:

1. Calls the normal Triton attention implementation unchanged.
2. Captures and accumulates `Q` slices across prefill and decode for the active sequence, subject to the configured buffer policy.
3. Extracts `K` from vLLM's paged KV cache when a snapshot is due.
4. Dispatches to `Diagnostic.reduce()` for each enabled signal, optionally calling `Diagnostic.accumulate()` to merge local results into a running global state.
5. Emits snapshots via the handler pipeline (JSONL, OTel, or custom).

This lets you observe attention structure during real generation rather than in a separate offline re-run.

## Streaming Diagnostics

For long or unbounded sequences, retaining the full Q buffer is impractical (O(L·H·d) per layer). Glassbox supports windowed Q-buffer management via two policies, controlled by `q_buffer_max_tokens` and `q_buffer_mode`:

### Sliding window (default)

```yaml
q_buffer_max_tokens: 512
q_buffer_mode: sliding
```

The buffer keeps the last `W` tokens. Each decode step appends one token and trims the oldest. Diagnostics fire per their configured `interval`, so consecutive observations overlap by `W - interval` tokens. This is the natural mode for continuous monitoring.

### Tumbling window

```yaml
q_buffer_max_tokens: 512
q_buffer_mode: tumbling
```

The buffer accumulates until it reaches `W` tokens, then fires all enabled signals and flushes. The window size is the cadence — per-signal `interval` is ignored. Consecutive windows are non-overlapping, which gives **window independence**: each `reduce()` call sees a disjoint block of the sequence.

This matters for streaming accumulation proofs. Many merge strategies (running means, sketches, decomposition additivity) require that local statistics come from independent blocks. Tumbling mode provides that guarantee at the backend level.

### The Diagnostic protocol

Each signal implements the `Diagnostic` protocol (`glassbox/diagnostic.py`):

| Method | Purpose |
|--------|---------|
| `reduce(Qh, Kh, L)` | Compute local scalar features from the current window |
| `witness(Qh, Kh, L)` | Per-token localization vector (optional, `emit_witness=True`) |
| `accumulate(local, state)` | Merge a local `reduce()` result into a running global state |

`reduce()` is always correct for the window it sees — the SVD, Hodge decomposition, or Laplacian eigenvalues of the windowed attention submatrix are exact. The question is whether accumulated global statistics are meaningful, which depends on the accumulation strategy.

Currently `accumulate()` returns the latest local result (no merge). The actual merge logic — proving that specific local→global strategies are correct for each signal — is the contribution of companion streaming papers. The backend provides the windowing modes and accumulation plumbing those papers need.

### When to use which mode

| Scenario | Mode | Why |
|----------|------|-----|
| Online monitoring, sliding-window anomaly detection | `sliding` | Overlapping windows give smoother signal, per-signal intervals allow different cadences |
| Training streaming accumulators, local→global correctness proofs | `tumbling` | Non-overlapping windows give independence, simplifies merge math |
| Unbounded (full sequence) | Either with `q_buffer_max_tokens=0` | Buffer grows with sequence length, statistics are global |

## How We Avoid Materializing The Full `L x L` Matrix

The key design goal is to avoid building full score or attention matrices whenever sequence length grows.

### 1. Matrix-free multiplies for `S = QK^T`

For any vector `v`:

- `Sv = Q(K^T v)`
- `S^T u = K(Q^T u)`

That means applying `S` or `S^T` only requires two thin `L x d` multiplies instead of constructing an `L x L` matrix. In code, this is implemented by:

- `matvec_S()` in `glassbox/svd.py`
- `matvec_ST()` in `glassbox/svd.py`

Both the randomized SVD and Lanczos implementations consume only these matvecs.

### 2. Randomized SVD and Lanczos operate on operators, not matrices

`glassbox/svd.py` provides:

- `randomized_svd()`
- `svd_via_lanczos()`

Both operate on callables `matvec` and `matvec_t`, so they never require the full matrix to exist in memory.

### 3. Blocked row streaming for post-softmax attention

For the degree-normalized operator, some quantities depend on softmaxed attention. When sequence length is large, `glassbox` uses blocked streaming:

- `apply_A_blocked()`
- `apply_AT_blocked()`
- `compute_dk_blocked()`
- `compute_logsumexp_blocked()`
- `matvec_M_blocked()`
- `matvec_MT_blocked()`

These compute the effect of `A` or `M` in row blocks, keeping memory bounded by the block size instead of `L^2`.

### 4. Two-tier execution for the normalized operator

For shorter sequences, `glassbox` will materialize the normalized operator because it is simple and exact. For longer sequences, it switches to the matrix-free path.

In practice:

- if `L <= threshold`, use a materialized path
- if `L > threshold`, use blocked matrix-free operators
- if `threshold = 0`, **always** use the matrix-free path (no `L×L` allocation)

The `threshold` setting is a speed knob, not a correctness one — both tiers produce equivalent results within numerical tolerance. Set `threshold=0` when you need a hard guarantee that no full attention matrix is ever materialized (e.g., memory-constrained deployments or streaming with large windows).

That behavior is implemented in each threshold-based diagnostic's `reduce()` method (routing, tracker, selfattn, laplacian).

### 5. On-demand entry lookup for curl estimates

Some routing metrics need access to selected entries of `M`. Instead of building all of `M`, the matrix-free path computes only sampled entries on demand with `get_M_entries_batch()` in `glassbox/svd.py`.

## Features We Compute

> **Which operator?** Each diagnostic runs on the attention matrix its mathematics requires —
> conductance/Cheeger on the degree-normalized **M**, Hodge asymmetry (G/Γ/C) on the row-stochastic
> post-softmax **P**, and cyclic-triangle / tournament signals on the pre-softmax scores **S = QKᵀ**.
> These are not interchangeable. See [docs/operator-choice.md](docs/operator-choice.md) for the
> rationale (with paper + formal-proof references) and the causal-masking caveats.

### 1. Spectral signal — scores matrix features (pre-softmax scores)

These come from the singular values of the pre-softmax scores matrix `S = QK^T`.

| Feature | Formula | Meaning |
|---|---|---|
| `sv1` | `σ₁(S)` | Leading singular value of the scores matrix. Strength of the dominant attention mode |
| `sv_ratio` | `σ₁(S) / σ₂(S)` | Spectral sharpness. High values suggest near-rank-1 structure; low values suggest multiple competing modes |
| `sv_entropy` | `-Σ pᵢ log pᵢ`, with `pᵢ = σᵢ / Σⱼ σⱼ` | Entropy of the normalized singular-value distribution. Measures how concentrated or diffuse the spectrum is |

### 2. Routing signal — degree-normalized matrix features (post-softmax attention) — Dahlem et al. (upcoming)

These come from the singular values and routing decomposition of the normalized operator `M`.

| Feature | Formula | Meaning |
|---|---|---|
| `sv1` | `σ₁(M)` | Leading singular value of `M`. Dominance of the top routing mode |
| `sv_ratio` | `σ₁(M) / σ₂(M)` | Separation between the top routing mode and the rest |
| `sv_entropy` | `-Σ pᵢ log pᵢ`, with `pᵢ = σᵢ / Σⱼ σⱼ` | Entropy of the normalized singular-value distribution. Spread of routing mass across modes |
| `sigma2` | `σ₂(M)` | Second singular value of `M`. Raw spectral-gap measure and persistence of non-dominant routing structure |
| `phi_hat` | `1 - σ₂(M)` | Conductance-like bottleneck score. High `φ̂` means attention concentrates through a single dominant mode; low `φ̂` means multiple competing routing paths |
| `G` | `‖M_asym‖_F / ‖M‖_F` | Total asymmetry. Fraction of `M`'s energy in the antisymmetric part, where `M_asym = (M - Mᵀ) / 2` |
| `Gamma` | `√(G² - C²)` | Gradient coefficient. The portion of asymmetry that is potential-driven rather than circulatory |
| `C` | `curl_RMS / (√2 · ‖M‖_F)` | Curl coefficient. The portion of asymmetry due to irreversible circulation, estimated by triangle sampling in the matrix-free path |
| `curl_ratio` | `C / (G + ε)` | Curl fraction. What share of total asymmetry is circulatory versus gradient-driven |
| `sigma2_asym` | `σ₂(M_asym)` | Second singular value of the antisymmetric part. Captures whether the irreversible component has multiple significant modes |
| `commutator_norm` | `‖[M_sym, M_asym]‖_F / ‖M‖_F` | Commutator norm. Measures how much the symmetric and antisymmetric parts interfere with each other, where `[A, B] = AB - BA` |

The routing and Hodge-style metrics live in `glassbox/hodge.py`. The feature schemas are defined in `glassbox/results.py`.

### 3. Tracker signal — AttentionTracker features (raw post-softmax attention) — [arXiv:2411.00348](https://arxiv.org/abs/2411.00348)

These come from the raw post-softmax attention matrix `A = softmax(QK^T / sqrt(d))`, without degree normalization. Based on the AttentionTracker paper, which uses these features for mechanistic classification of failure modes (prompt injection vs hallucination).

Currently implements the span-independent features only. Span-aware features (`focus_score`, `cut_flow`) require instruction/data span boundaries and are planned for a future release.

| Feature | Formula | Meaning |
|---|---|---|
| `sigma2` | `σ₂(A)` | Second singular value of the raw attention matrix. Persistence of non-dominant attention structure |
| `sigma2_asym` | `σ₂(A_asym)` | Second singular value of `A_asym = (A - Aᵀ) / 2`. Whether the irreversible (asymmetric) component has multiple significant modes |
| `commutator_norm` | `‖[A_sym, A_asym]‖_F / ‖A‖_F` | Coupling between symmetric and antisymmetric parts of attention, where `[X, Y] = XY - YX` |

The computation reuses the same two-tier approach as the degree-normalized operator: materialized for `L <= threshold`, matrix-free (blocked-streaming matvecs) above. The matrix-free path reuses existing matvec infrastructure by treating `A` as the special case of `M` where `D_K^{-1/2} = I`.

The feature computation lives in `glassbox/attention_tracker.py`.

### 4. SelfAttn signal — attention diagonal features (LLM-Check, NeurIPS 2024 + LapEigvals, EMNLP 2025)

These come from the diagonal of the post-softmax attention matrix `A = softmax(QK^T / sqrt(d))`.

| Feature | Formula | Meaning |
|---|---|---|
| `attn_diag_logmean` | `mean_i(log(A[i,i]))` | Mean log self-attention weight. Higher values indicate stronger self-attention, which correlates with model confidence and factuality |
| `eigvals` | `topk(diag(A))` | Top-k diagonal values of A (descending). For causal attention these are A's eigenvalues; used as a baseline in LapEigvals |

The matrix-free path avoids materializing `A` by computing `log(A[i,i]) = s_ii - logsumexp_i` where `s_ii = Q[i]·K[i]/sqrt(d)` (O(Ld)) and `logsumexp` is computed blockwise.

Implementation: `glassbox/attention_diagonal.py`. References: [LLM-Check](https://github.com/GaurangSriramanan/LLM_Check_Hallucination_Detection), [LapEigvals](https://github.com/graphml-lab-pwr/lapeigvals).

Two additional LLM-Check extractors require signals the attention backend cannot see:

- **LLMCheckLogitsExtractor** — entropy and perplexity from output logits. Needs the final lm_head output, not Q/K.
- **HiddenStateCovarianceExtractor** — SVD of centered covariance of hidden states. Needs per-layer hidden states before projection.

These will require either `torch.compile` passes (see `glassbox/passes/`) or vLLM observation hooks.

### 5. Laplacian signal — eigenvalue features (LapEigvals, EMNLP 2025) — [arXiv:2502.17598](https://arxiv.org/abs/2502.17598)

These come from the in-degree graph Laplacian of the attention matrix: `L = D_in - A`, where `D_in[i,i] = Σ_j A[j,i]` (column sums of `A`). Treating attention as a weighted directed graph, the Laplacian diagonal captures how much attention each token receives from other tokens (in-degree minus self-attention).

For causal (lower-triangular) attention matrices, `L` is also triangular, so its eigenvalues are its diagonal entries — no eigendecomposition needed. The features are the top-k largest diagonal values.

| Feature | Formula | Meaning |
|---|---|---|
| `eigvals` | `topk(diag(D_in - A))` | Top-k Laplacian diagonal values (descending). Profile of how attention in-degree is distributed across tokens |

The two-tier approach reuses existing blocked-streaming infrastructure: `apply_AT_blocked` for column sums and `compute_logsumexp_blocked` for the attention diagonal. No SVD or eigendecomposition is performed.

The paper also constructs a **multi-layer graph** where layers are connected by vertical edges weighted by the next layer's self-attention diagonal. This gives richer features but requires cross-layer aggregation; the current implementation covers the single-layer case. The multi-layer variant is planned as a post-processing step over emitted per-layer features.

Implementation: `glassbox/laplacian_eigvals.py`. Reference: [LapEigvals](https://github.com/graphml-lab-pwr/lapeigvals).

### 6. Cheeger signal — bipartite sweep conductance (post-softmax attention) — Dahlem et al. (upcoming)

The Cheeger diagnostic measures how close the attention routing is to a bottleneck — a near-rank-1 regime where almost all attention mass flows through a single mode. It works on the same degree-normalized operator `M = D_Q^{-1/2} A D_K^{-1/2}` as the routing signal, but instead of reporting singular values directly, it computes the **Cheeger bracket**: a nested interval that bounds the true bipartite conductance `φ*`.

**The bracket.** Let `λ₂` be the second-largest eigenvalue of the symmetric part `M_sym = (M + M^T)/2`, and `μ₂ = 1 − λ₂` the spectral gap. The classical bipartite Cheeger inequality gives:

```
μ₂/2  ≤  φ*  ≤  √(2μ₂)
```

The diagnostic emits this bracket at three tiers of increasing cost and tightness:

| Tier | Requires | Produces | Cost |
|------|----------|----------|------|
| **1 (always-on)** | `λ₂(M_sym)` only | `cheeger_lower`, `cheeger_upper` from the classical inequality | One eigenproblem or Lanczos call (k=2) |
| **2 (gap-guarded)** | Fiedler eigenvector + sweep | `phi_star` (exact sweep conductance), `phi_hat` (tight upper) | Full SVD + O(L) sweep |
| **3 (fallback)** | Higher-order eigenvalues | `improved_upper` via KLGT bound: `O(k) · μ₂ / √μ_{k+1}` | Reuses Lanczos eigenvalues |

**Why the upper bound matters.** The lower bound `μ₂/2 ≤ φ*` certifies that conductance is *large* — it rules a bottleneck *out*. The upper bound certifies conductance is *small* — it's the direction that fires an alarm. A closed aperture or injection sink is a low-conductance bottleneck, so the failure signal lives in the upper tail.

| Feature | Formula | Meaning |
|---|---|---|
| `phi_star` | Bipartite sweep on Fiedler vector of `M` | Exact sweep conductance. `None` in light mode |
| `sigma2` | `λ₂(M_sym)` | Second-largest eigenvalue of `M_sym` — spectral gap measure |
| `cheeger_lower` | `(1 − σ₂) / 2` | Tier 1 lower bound on φ* |
| `cheeger_upper` | `√(2(1 − σ₂))` | Tier 1 upper bound on φ* |
| `phi_hat` | Sweep conductance when gap-healthy | Tier 2 tight upper (equals `phi_star` when available) |
| `improved_upper` | `min_k k · μ₂ / √μ_{k+1}` (KLGT, adaptive) | Tier 3 fallback upper — adaptive k selection minimizes the bound |
| `bracket_width` | `cheeger_upper − cheeger_lower` | Confidence signal — narrow = informative, wide = recompute needed |
| `spectral_gap` | `λ₂ − λ₃` of `M_sym` | Gap health for Davis-Kahan stability of the Fiedler vector |
| `recomputed` | boolean | Whether a full recompute was triggered this window |
| `lambda_min` | `λ_min(M_sym)` | Smallest eigenvalue of `M_sym` — drives bipartiteness diagnostic |
| `dual_gap` | `1 + λ_min` | Bipartiteness spectral gap (Bauer-Jost 2013). 0 = perfectly bipartite |
| `dual_cheeger_lower` | `dual_gap / 2` | Lower bound on dual Cheeger constant β |
| `dual_cheeger_upper` | `√(2 · dual_gap)` | Upper bound on dual Cheeger constant β |

**Three modes.** The `mode` config parameter selects the cost/accuracy tradeoff:

- **`batch`** (default) — Full SVD + sweep every window. Produces `phi_star`, `sigma2`, and the full bracket. The most accurate option; use for offline analysis or when interval spacing amortizes the cost.
- **`streaming`** — Bordered Rayleigh-Ritz (BRR) amortization between full Lanczos recomputes. Carries a Ritz basis across decode steps and triggers full recompute on spectral-gap collapse, degree-shift, or geometric stride. Best for unbounded autoregressive decode where the window grows per token.
- **`light`** — Quick `σ₂` estimation → tier-1 gap-free bounds only. No sweep cut, no `phi_star`. Cheapest option for high-frequency spectral health monitoring.

**Streaming controller parameters** (only used when `mode="streaming"`):

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `gap_threshold` | 0.05 | `λ₂ − λ₃` floor — trigger full recompute when gap narrows below this |
| `degree_shift_threshold` | 0.1 | Mean degree-ratio shift between consecutive windows — trigger on structural change |
| `geometric_base` | 2.0 | Geometric stride multiplier — forces periodic recompute even if triggers don't fire |
| `ritz_rank` | 3 | Number of Ritz vectors carried forward in BRR |
| `n_explore` | 2 | Random exploration vectors added to BRR basis each step |
| `lanczos_iters` | 20 | Krylov iterations for full Lanczos recompute |
| `improved_cheeger_k` | 4 | Max eigenvalues for KLGT bound — adaptive k selection picks the tightest |

Implementation: `glassbox/cheeger.py` (sweep + bounds), `glassbox/diagnostics/cheeger.py` (three-mode controller), `glassbox/svd.py` (`bordered_rayleigh_ritz`, `hermitian_lanczos`).

### More features coming soon

#### Span-aware AttentionTracker features (raw post-softmax attention)

The AttentionTracker paper also defines `focus_score` (attention mass from data tokens to instruction span) and `cut_flow` (net directed attention flow between instruction and data regions). These require instruction/data span boundaries, which need a mechanism to pass into the backend (config, per-request metadata, or marker-token detection).

#### Transport features from the Degree-normalized matrix (post-softmax attention, value-weighted routing)

These move beyond pure attention geometry and start incorporating what is actually being transported through the head.

#### Curl spectrum features from the Degree-normalized matrix (post-softmax attention, per-dimension value analysis)

These summarize how curl-like behavior is distributed across important value dimensions.

#### LayerNorm-weighted features

These modulate routing features by the effective LayerNorm gain, with the goal of emphasizing heads and layers whose routed signal is more strongly amplified by the surrounding network.

## Signal Emission Architecture

Extracted signals flow through a pluggable handler system with two tiers, designed for two downstream use cases: **offline training** of detection models and **real-time inference-time detection**.

```
                    ┌─────────────────────┐
                    │   Attention Backend  │
                    │   (SVDSnapshot)      │
                    └────────┬────────────┘
                             │
                 ┌───────────┴───────────┐
                 │                       │
          Tier 2: Full stream     Tier 1: Real-time
          (offline / bulk)        (inference-time)
                 │                       │
         ┌───────┴───────┐          OtelHandler
         │               │         glassbox.* spans
    JsonlHandler    Custom handler       │
    .jsonl file    (Kafka, Redis,   OTel Collector
                    webhook, etc.)       │
                                   Jaeger / detector
                                   / alerting
```

**Tier 2 — Full feature stream (training / bulk analysis):**
`JsonlHandler` writes every snapshot as a JSON line. Custom handlers can implement the `SnapshotHandler` protocol to forward snapshots to Kafka, Redis Streams, or any other sink.

**Tier 1 — OTel integration (real-time detection):**
`OtelHandler` emits each snapshot as a short-lived OpenTelemetry span with `glassbox.*` attributes (signal type, layer, head, step, and all derived features). It piggybacks on vLLM's global `TracerProvider` — when vLLM is started with `--otlp-traces-endpoint`, Glassbox spans flow through the same collector (Jaeger, Tempo, Datadog, etc.) with zero additional configuration. The `heads`, `interval`, and signal selection should be configured to match what your trained detection model expects.

Multiple handlers can be active simultaneously (e.g. JSONL for archival + OTel for real-time). When neither `output` nor `otel` is configured, a `LoggingHandler` logs snapshots to stderr.

### Custom handlers

Any object with `handle(snapshot)` and `close()` methods satisfies the `SnapshotHandler` protocol:

```python
from glassbox import SnapshotHandler, SVDSnapshot
from glassbox.backends.svd_backend import SVDTritonAttentionImpl

class MyHandler:
    def handle(self, snapshot: SVDSnapshot) -> None:
        # forward to Kafka, Redis, a classifier, etc.
        ...
    def close(self) -> None:
        ...

# Register before engine creation
SVDTritonAttentionImpl._handlers.append(MyHandler())
```

## Output Format

The backend emits one JSON object per observation. Each row contains:

- `signal`: `spectral`, `routing`, `tracker`, `selfattn`, or `laplacian`
- `request_id`
- `layer` and `layer_idx`
- `head`
- `step`
- `L`
- `singular_values`
- `tier`: `materialized` or `matrix_free` for signals that use the two-tier approach
- `features`: derived metrics for that observation

This format is designed to feed downstream systems directly. You can:

- train hallucination detectors on snapshot features
- compare truthful vs hallucinated generations
- monitor drift across prompts or tasks
- aggregate by head, layer, request, or dataset

The `glassbox-extract` CLI can also write a wide Parquet file.

## Example Downstream Uses

- Hallucination detection: compare spectral and routing signatures between factual and hallucinated responses
- Task drift detection: identify layer/head regimes that diverge when the model loses the task
- Uncertainty quantification: use spectral concentration and routing asymmetry as auxiliary confidence signals
- Failure mode analysis: inspect how internal structure changes across prompts, models, or checkpoints

## Installation

This repository expects vLLM to be installed separately.

```bash
# using a Ubuntu 24.04 Deep Learning AMI, I just needed to
source /opt/pytorch/bin/activate
pip install vllm==0.15.1
pip install huggingface-hub==0.36.0
pip install pydantic-settings==2.12.0
```

Once you have vLLM installed, you can install the package:

```bash
pip install -e .
```

For local development:

```bash
pip install -e .[dev]
```

For OpenTelemetry support (already present when running inside vLLM):

```bash
pip install -e .[otel]
```

## Configuration

Configuration is defined in `glassbox/config.py` and can be provided programmatically or through `glassbox.yaml`.

Example:

```yaml
spectral:
  enabled: true
  interval: 32
  rank: 4
  method: randomized
  heads: [0]

routing:
  enabled: true
  interval: 32
  rank: 4
  method: randomized
  heads: [0]
  threshold: 2048
  block_size: 256
  hodge_target_cv: 0.05
  hodge_curl_seed: 42

tracker:
  enabled: true
  interval: 32
  rank: 4
  method: randomized
  heads: [0]
  threshold: 512
  block_size: 256

selfattn:
  enabled: true
  interval: 32
  heads: [0]
  top_k: 10
  threshold: 512
  block_size: 256

laplacian:
  enabled: true
  interval: 32
  heads: [0]
  top_k: 10
  threshold: 512
  block_size: 256

output:
  path: experiments/results/svd_features.jsonl

emit:
  otel: false

# Q-buffer windowing (streaming diagnostics)
q_buffer_max_tokens: 0      # 0 = unbounded (full sequence)
q_buffer_mode: sliding       # "sliding" or "tumbling"
emit_witness: false          # attach per-token localization vectors
```

Important knobs:

| Setting | Description |
|---|---|
| `spectral.interval` | Snapshot cadence for `S = QK^T` |
| `spectral.rank` | Number of singular values to keep |
| `spectral.method` | `randomized` or `lanczos` |
| `spectral.heads` | Heads to analyze |
| `routing.enabled` | Turn on routing analysis |
| `routing.threshold` | Sequence length cutoff for materialized vs matrix-free (0 = always matrix-free) |
| `routing.block_size` | Row-block size for blocked operators |
| `tracker.enabled` | Turn on raw attention matrix analysis |
| `tracker.threshold` | Sequence length cutoff for materialized vs matrix-free (0 = always matrix-free) |
| `selfattn.enabled` | Turn on attention diagonal analysis |
| `selfattn.interval` | Snapshot cadence for diagonal features |
| `selfattn.heads` | Heads to analyze |
| `selfattn.top_k` | Number of top diagonal values to keep (0 = omit eigvals) |
| `selfattn.threshold` | Sequence length cutoff for materialized vs matrix-free (0 = always matrix-free) |
| `laplacian.enabled` | Turn on Laplacian eigenvalue analysis |
| `laplacian.interval` | Snapshot cadence for Laplacian features |
| `laplacian.heads` | Heads to analyze |
| `laplacian.top_k` | Number of top eigenvalues to keep |
| `laplacian.threshold` | Sequence length cutoff for materialized vs matrix-free (0 = always matrix-free) |
| `output.path` | JSONL output path (feature logging pipeline) |
| `emit.otel` | Emit snapshots as OpenTelemetry spans (inference pipeline) |
| `q_buffer_max_tokens` | Max Q-buffer tokens per layer (0 = unbounded). Bounds memory to O(W·H·d) per layer |
| `q_buffer_mode` | `sliding` (overlapping, trim oldest) or `tumbling` (non-overlapping, flush after fire) |
| `emit_witness` | Attach per-token localization vectors to snapshots (O(L) per snapshot when on) |

## Running Glassbox

There are three ways to run Glassbox, depending on your use case:

| Mode | Use case | Typical emission |
|------|----------|-----------------|
| `vllm serve` | Production / inference-time detection | OTel spans via `glassbox.yaml` |
| `glassbox-run` | Development / single-prompt testing | OTel (`--otel`) or JSONL (`--output`) |
| `glassbox-extract` | Offline feature extraction for training detection models | JSONL + Parquet |

### `vllm serve` — production inference

```bash
vllm serve facebook/opt-125m --attention-backend CUSTOM --enforce-eager
```

Glassbox registers itself via the `vllm.general_plugins` entry point — vLLM loads it automatically. Configure via a `glassbox.yaml` in the working directory:

```yaml
spectral:
  enabled: true
  interval: 32
  heads: [0]
emit:
  otel: true
```

When vLLM is started with `--otlp-traces-endpoint`, Glassbox spans flow through the same OTel collector with zero additional configuration. The `heads`, `interval`, and signal selection should match what your trained detection model expects.

### `glassbox-run` — single-prompt testing

```bash
glassbox-run \
  --model facebook/opt-125m \
  --signal spectral \
  --interval 16 \
  --rank 4 \
  --heads 0 \
  --otel \
  --prompt "The future of artificial intelligence is"
```

Launches vLLM with `attention_backend="CUSTOM"` and `enforce_eager=True`. Supports all CLI flags (`--signal`, `--interval`, `--rank`, `--heads`, `--otel`, `--output`, `--q-buffer-max-tokens`, `--q-buffer-mode`). The `--output` flag writes JSONL and is available for debugging/archival, but the typical runner use case is `--otel`. Settings can also be loaded from a `glassbox.yaml` in the working directory.

### `glassbox-extract` — offline feature extraction

```bash
glassbox-extract \
  --model Qwen/Qwen2-7B-Instruct \
  --dataset halueval_hallucination \
  --max-samples 200 \
  --signal spectral,routing,tracker,selfattn,laplacian \
  --parquet
```

Runs two-phase prefill (question-only + full prompt+response) on labeled datasets and produces:

- `samples.jsonl` — per-request sample metadata
- `svd_features.jsonl` — snapshot features
- `features.parquet` — optional wide Parquet for downstream training (with `--parquet`)

Output goes to `experiments/results/{timestamp}/` by default, or override with `--outdir`. Also supports `--otel` (though the typical use case for extraction is JSONL or parquet).

## Benchmarks

Coming soon.
