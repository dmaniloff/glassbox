# Magnetic-Laplacian frustration diagnostic

The `magnetic` signal measures whether an attention head's **directional preference structure
can be reconciled into a coherent ranking, or whether it contains irreducible loops
("frustration")**. It is the spectral member of the *orientation* operator family (alongside the
discrete cyclic-triangle count `|T_cyc|`), and the only diagnostic that reads the head's latent
preference geometry **robustly to causal masking and to softmax/degree normalization**.

See also: [operator taxonomy](operator-choice.md) · [streaming modes](streaming-modes.md) ·
research note (issue #68).

---

## The operator

It is built on the **unmasked pre-softmax scores** `S = QKᵀ/√d` — *not* the post-softmax
attention. A causal post-softmax matrix is lower-triangular ⇒ its orientation is transitive ⇒
frustration is trivially zero. The preference structure lives in the antisymmetric part of `S`
(`qᵢ·kⱼ` vs `qⱼ·kᵢ`), which the causal mask never touches.

From `S`, glassbox forms the Hermitian **magnetic Laplacian** (formally verified in shade-formal
`MagneticFrustration.lean`):

| symbol | definition | role |
|---|---|---|
| `W_ij` | `(|S_ij| + |S_ji|)/2` | symmetric magnitude (preference *strength*), ≥ 0, `W_ii=0` |
| `θ_ij` | `arctan((S_ij − S_ji)/(S_ij + S_ji))` | antisymmetric phase (preference *direction*), `θ_ji=−θ_ij` |
| `A_θ` | `W ⊙ e^{iθ}` | Hermitian transport matrix |
| `D` | `diag(Σ_j W_ij)` | real degree |
| `L_φ` | `D − A_θ` | Hermitian, positive-semidefinite |

There is **no charge parameter** (effectively `g=1`). The construction is **gauge-invariant**: a
pure-gradient phase shift is a diagonal-unitary conjugation that preserves the whole spectrum, so
degree normalization (`M` vs the row-stochastic `P`) leaves the frustration unchanged — that is
precisely why it sits on `S` rather than `M`/`P`.

---

## The components (what gets emitted)

`MagneticFeatures` carries up to four numbers per (layer, head, fire):

| field | what it is | `0` means | cost |
|---|---|---|---|
| `frustration` (`λ₁`) | smallest eigenvalue of `L_φ` | a coherent global ranking exists (balanced) | batch eigensolve |
| `phase_curl` | unweighted Hodge curl energy of `θ`: `‖θ‖² − 2‖r_θ‖²/L` | `θ` is a pure gauge gradient (balanced) | streamable, O(t)/token |
| `phase_curl_w` | magnitude-weighted curl: `Σ W_ij θ_ij² − 2 Σ b_i²/d_i` | balanced | streamable, O(t)/token |
| `witness` | per-token magnitudes of the bottom eigenvector of `L_φ` | — | batch (with `λ₁`) |

where `r_θ = θ·1`, `b_i = Σ_j W_ij θ_ij`, `d_i = Σ_j W_ij`.

**How they relate.** All three are `0` exactly when the orientation is *balanced* (no
frustration). Away from zero:

- **`λ₁`** is a *min-eigenvalue* — the global spectral floor, weighted by preference strength. It
  is large only when **no coherent ranking survives even approximately** (pervasive frustration).
  It is the most *specific* severity measure, and it carries the eigenvector **witness** (which
  tokens form the frustrated mode). It has **no exact streaming update**, so it is a batch /
  on-demand quantity.
- **`phase_curl`** is a *total-energy* aggregate (the sum of all squared triangle holonomies
  `Σ Φ_ijk²`). It rises whenever there is more circulation anywhere — *sensitive*, and the
  formally-cleanest pure-phase measure, but it can be inflated by weak / near-symmetric edges
  whose `arctan` phase is noise (it tracks `λ₁` only at Spearman ρ≈0.68).
- **`phase_curl_w`** is the **faithful streamable `λ₁` proxy**: the Jacobi (diagonal) weighted
  Hodge curl. It downweights weak edges by magnitude `W` and tracks `λ₁` at **ρ≈0.97** —
  matching the exact batch weighted-Hodge — while remaining additive and fully streamable. It
  reduces to `phase_curl` when `W` is uniform, so it is the exact generalization, not an ad-hoc
  reweighting. **This is the one to monitor in real time.**

> **`λ₁ = 0 ⇒ |T_cyc| = 0`, but not conversely.** The magnetic frustration is a *strictly finer*
> balance condition than the sign-level cyclic-triangle count: it also catches the case where the
> tokens *are* orderable by sign yet the preference *strengths* don't form a consistent potential.

---

## How to use it

### Enable the signal

```python
from glassbox.config import GlassboxConfig

# Batch λ₁ + both phase-curls (default mode)
cfg = GlassboxConfig(magnetic={"enabled": True, "interval": 32, "heads": [0, 1]})

# Streaming frustration only (eigensolver-free phase-curl, λ₁ left None)
cfg = GlassboxConfig(
    magnetic={"enabled": True, "incremental": True},
    q_buffer_max_tokens=0,   # incremental needs the full prefix (unbounded buffer)
)
```

Or from the CLI: `--signal magnetic` (combine with `--threshold`, `--block-size`).

### `MagneticConfig` fields

| field | default | meaning |
|---|---|---|
| `threshold` | 512 | `L ≤ threshold` → dense eigh; above → matrix-free complex Lanczos |
| `block_size` | 256 | block width for the matrix-free / streaming paths |
| `incremental` | `False` | report the streamable phase-curl folded across fires; **leaves `λ₁` as `None`** |

### Two modes of operation

- **Batch (default, `incremental=False`).** Each fire computes `λ₁` (dense eigh for
  `L ≤ threshold`, else complex Lanczos), the eigenvector witness, and **both** phase-curls.
  Use for forensics, calibration, and the spatial (which-tokens) witness.
- **Streaming (`incremental=True`).** Maintains the curl sufficient statistics across fires and
  reports the **exact full-sequence `phase_curl` / `phase_curl_w`** with **no eigensolve**. Use
  for real-time monitoring. Requires the unbounded Q-buffer (`q_buffer_max_tokens=0`); the
  windowing invariant is enforced by the streaming-modes validator (see
  [streaming-modes](streaming-modes.md)).

### Which component to read

| you want… | use | why |
|---|---|---|
| real-time frustration monitor / trigger | **`phase_curl_w`** (streaming) | cheap, additive, tracks `λ₁` at ρ≈0.97 |
| the *specific* "ranking has collapsed" severity | `λ₁` (batch, on demand) | min-eigenvalue; pervasive-frustration measure |
| which tokens form the frustrated mode | `witness` (batch) | bottom eigenvector localization |
| pure phase topology, magnitude-agnostic | `phase_curl` | formally-cleanest, sign/strength-blind |

The intended pattern (mirrors conductance `σ₂` vs the Cheeger sweep): **stream `phase_curl_w`
continuously; fire `λ₁` on demand** — when the stream trips, and periodically for calibration —
for the specific severity and the mode witness. Do **not** run the eigensolve per token.

---

## What it tells you about failure modes

A well-functioning head (induction, retrieval, previous-token, syntactic) implements a near-
consistent preference — it can rank tokens by relevance — so it lives near `λ₁ ≈ 0`. Frustration
is *irreducible directional conflict*, which the diagnostic is positioned to flag:

- **In-context contradiction / ambiguity** — contradictory context admits no consistent salience
  ordering ⇒ frustration rises.
- **Prompt injection / adversarial competition** — an injected instruction installs a *competing*
  preference order; the conflict shows up as frustration, and the eigenvector **witness localizes
  which tokens** create it. Mask- and gauge-invariance make this robust.
- **A head going out-of-distribution** — one that should rank but turns circulatory.

Because `phase_curl_w` is additive and streamable, you can watch the frustration **trajectory**
live and read the per-token increment as a **temporal witness** ("token *t* spiked frustration"),
with additive attribution guarantees — something `λ₁` cannot give cheaply.

### How it complements the other operators

| operator | axis | the magnetic operator adds… |
|---|---|---|
| conductance `σ₂` (M) | *can* information mix | direction: is the **flow** consistent, not just connected |
| Hodge `G/Γ/C` (P) | how much asymmetric **mass**, gradient vs curl of the *realized* routing | the **orientation** of the *latent* geometry, mask- & gauge-invariant |
| `|T_cyc|` (S, sign) | topological **count** of preference cycles | magnitude-weighted **spectral severity**; a strictly finer balance test |

---

## Faithfulness & status

The construction and its key properties — Hermiticity, PSD (`λ₁ ≥ 0`), gauge invariance, and
`λ₁ = 0 ⟺ balanced` — are **machine-checked** in shade-formal (`MagneticFrustration.lean`,
sorry-free). The row-sum Hodge identity behind `phase_curl` is verified in `CurlRatioAnalysis.lean`
(the same identity the asymmetry curl uses). `phase_curl_w` is the diagonal (Jacobi) approximation
of the weighted Hodge projection — exact for uniform `W`, and empirically ρ≈0.97 to `λ₁`.

**Status: proposed, not yet empirically validated.** The failure-mode mappings above are
hypotheses the diagnostic *enables testing*, not established results. Treat the per-head values as
relative-to-baseline; calibrate before acting on them.

## References (SHADE papers)

- *directed-attention-geometry* — magnetic Laplacian of attention, frustration index, gauge
  invariance of `λ₁`.
- *structural-streaming-attention* — the streaming diagnostic suite; magnetic frustration as the
  spectral orientation member on pre-softmax `S`.
