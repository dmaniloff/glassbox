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
| `routing` | M SVD + Hodge `G` + conductance (`φ̂`, `σ₂`) | ✓ | ✗ spectral / SVD | — |
| **conductance / Cheeger** (`φ̂`, `σ₂` bracket on M) | conductance | ✓ | ✗ spectral¹ | —² |
| `asymmetry` *(in flight, #58/#60/#62)* | Frobenius `G/Γ/C` | ✓ | ✓ **additive** | ✓ |
| `cyclic` *(in flight, #42)* | `\|T_cyc\|` | ✓ | ✗ non-additive | ✓ |
| `tracker` | A SVD | ✓ | ✗ spectral | — |
| `selfattn` | diagonal stats | ✓ | — | — |
| `laplacian` | Laplacian eigvals | ✓ | ✗ spectral | — |
| `magnetic` *(#41)* | frustration `λ₁` (spectral) | ✓ | ✗ spectral | —³ |
| `magnetic` *(#68/#69)* | phase-curl energy `‖θ‖²−2‖r_θ‖²/L` | ✓ | ✓ **additive** | ✓ |

¹ Conductance is **doubly** unsound as a block-diagonal-global: `σ₂` is an order statistic of
the *union* of block spectra (not a sum), and the conductance `φ` of a block-diagonal (i.e.
disconnected) graph is degenerate (the inter-block cut is free, `φ→0`). Use *local block* per
window. ² No *exact* incremental update; the streaming-Cheeger line (#38/#53) maintains an
**approximate** `σ₂` via bordered Rayleigh–Ritz — a separate approximate regime outside this
exact-modes matrix.

³ The magnetic frustration `λ₁` has **no exact incremental update**: a new token borders `L_φ`
*and* shifts every prior degree, a rank-`t` PSD change, so interlacing/secular tricks don't
apply (only approximate subspace tracking, like `σ₂` above). The **faithful streamable
frustration** is the **phase-curl energy** (next row) — the Hodge curl of the phase field `θ`,
computed by the same row-sum identity as the asymmetry curl, hence additive and fully
streamable. `λ₁` and `phase_curl` are both `0 ⟺ balanced`, and `phase_curl` brackets `λ₁`
(magnetic Cheeger); `λ₁` itself stays batch (or approximate). See issue #68.

Conductance is **already emitted today** by the `routing` signal (`phi_hat`, `sigma2`) on M;
#38/#53 is the dedicated streaming version. The Cheeger σ₂ bracket is the M-operator family in
the [operator taxonomy](operator-choice.md) (Cheeger→M, Hodge→P, orientation→pre-softmax S).

**Rule of thumb:** additive (Frobenius/sum) statistics get all four modes (e.g. the asymmetry
`G/Γ/C` and the magnetic **phase-curl energy** — both Hodge sums-of-squares via the same row-sum
identity); spectral statistics (SVD `σ`, `σ₂`, Laplacian/magnetic `λ₁`) get *local block* +
*full recompute* (+ *approximate* subspace tracking), **never** block-diagonal-global; and
non-additive-combinatorial statistics (`|T_cyc|`) get *local block* + *exact-full incremental*
(an O(Δ) update) but **never** block-diagonal-global. When a spectral quantity has an additive
Hodge proxy (frustration `λ₁` → phase-curl energy), **that proxy is the faithful stream**.
