# Streaming modes: which are mathematically sound per diagnostic

A diagnostic can be evaluated over a stream in several modes. They are **not**
interchangeable — whether a *global* streaming statistic is correct depends on the
statistic's algebra and on the window configuration. This document is the reference matrix;
`GlassboxConfig` enforces the windowing invariants below (see `validate_window_modes`), so an
unsound combination raises at construction rather than silently mis-reporting a number.

For *which operator* each diagnostic runs on, see [operator-choice.md](operator-choice.md).

## The modes

| Mode | Config | Window | What it reports |
|---|---|---|---|
| **Local block** | default (no streaming flag) | `q_buffer_max_tokens=W` (sliding or tumbling) | the statistic of the **current window** only — per-fire, bounded memory, no accumulation |
| **Full recompute** | default | `q_buffer_max_tokens=0` (unbounded) | the **exact full-sequence** statistic, recomputed from scratch each fire (O(L)-cost per fire) |
| **Block-diagonal global** | `streaming=True` | `q_buffer_mode="tumbling"`, `W>0` | the statistic of the **block-diagonal** operator `blockdiag(W₁…W_k)` over the stream — accumulated cheaply from per-window sufficient statistics (bounded memory). **Drops cross-window structure.** |
| **Exact-full global** | `incremental=True` | `q_buffer_max_tokens=0` (unbounded) | the **exact full-sequence** statistic, maintained by an O(Δ) per-token update (cheap; no per-fire recompute) |

## Soundness criterion

The only subtle mode is **block-diagonal global** (`streaming=True`): accumulating per-window
results into one number is correct **iff the statistic is additive over a disjoint partition**
— i.e. the global value over `blockdiag(W₁…W_k)` equals a simple aggregate (a sum) of the
per-window values, with **no cross-window terms**.

- **Additive** — sums over matrix entries / independent contributions: Frobenius norms and
  sums-of-squares (e.g. the asymmetry `‖A_asym‖²`, `‖P‖²`, gradient energy `2‖r‖²/L`). Here
  `streaming=True` is sound, and the accumulated object is the exact `G/Γ/C` of the
  block-diagonal operator over the processed stream.
- **NOT additive** — anything with cross-window coupling:
  - **Spectral** functionals (SVD singular values, `σ₂`, Laplacian eigenvalues, magnetic
    frustration `λ₁`): the spectrum of a block-diagonal is the *union* of block spectra, so a
    leading value / ratio of the whole is **not** a sum of per-window values.
  - **Combinatorial cross-window** (`|T_cyc|`): a cyclic triple `{i,j,k}` with vertices in
    different windows is invisible to every per-window count, so `Σ per-window < global`.

For non-additive statistics there is **no valid block-diagonal-global mode** — use *local
block* (per-window) or *exact-full global* (incremental, when an O(Δ) update exists).

## Windowing invariants (enforced by `GlassboxConfig`)

- `q_buffer_mode="tumbling"` ⇒ `q_buffer_max_tokens > 0` (tumbling = fixed non-overlapping
  windows; meaningless unbounded).
- `streaming=True` ⇒ `q_buffer_mode="tumbling"` and `q_buffer_max_tokens > 0` (disjoint
  windows; sliding double-counts the overlap, unbounded is not a block-diagonal partition).
- `incremental=True` ⇒ `q_buffer_max_tokens = 0` (the exact-full update needs the whole
  prefix; a bounded buffer trims priors and breaks exactness).

## The matrix

✓ = sound & available · ✗ = mathematically unsound · — = sound in principle but no streaming
update implemented (use *local block* or *full recompute*).

| Diagnostic | Statistic | Local block | Block-diagonal global (`streaming`) | Exact-full global (`incremental`) |
|---|---|---|---|---|
| `spectral` | scores SVD | ✓ | ✗ spectral | — |
| `routing` | M SVD + Hodge `G` | ✓ | ✗ spectral / SVD | — |
| `asymmetry` *(in flight, #58/#60/#62)* | Frobenius `G/Γ/C` | ✓ | ✓ **additive** | ✓ |
| `cyclic` *(in flight, #42)* | `\|T_cyc\|` | ✓ | ✗ non-additive | ✓ |
| `tracker` | A SVD | ✓ | ✗ spectral | — |
| `selfattn` | diagonal stats | ✓ | — | — |
| `laplacian` | Laplacian eigvals | ✓ | ✗ spectral | — |
| `cheeger` *(planned, #38)* | conductance `σ₂`/`φ` | ✓ | ✗ spectral | — |
| `magnetic` *(planned, #41)* | frustration `λ₁` | ✓ | ✗ spectral | — |

**Rule of thumb:** additive (Frobenius/sum) statistics get all four modes; spectral and
non-additive-combinatorial statistics get *local block* + *full recompute* (and *exact-full
incremental* where an O(Δ) update exists, e.g. `|T_cyc|`), but **never** block-diagonal-global.
