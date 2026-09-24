# Operator choice: which attention matrix each diagnostic uses

Glassbox computes diagnostics on three matrices: pre-softmax scores S, post-softmax
attention P, and degree-normalized attention M. **Each diagnostic family must run on the
one its mathematics requires** — they are not interchangeable. Using the wrong operator silently
changes (or destroys) the quantity being measured. The choices below are grounded in the
SHADE papers (see References).

## TL;DR

| Diagnostic family | Matrix | glassbox signal | Why | What it gives |
|---|---|---|---|---|
| Conductance / bottleneck | **M** = degree-normalized post-softmax | `cheeger`, `routing` | Cheeger σ₂ bracket is a theorem about the *normalized* operator | transport bottleneck bracket `(1−σ₂)/2 ≤ φ ≤ √(2(1−σ₂))`; plus `routing`'s `asym_index` transpose-sensitivity scalar |
| Normalized asymmetry index | **M** = degree-normalized post-softmax | `routing` (`asym_index`) | measures asymmetry of the degree-normalized operator | normalized asymmetry `‖M_asym‖_F/‖M‖_F` |
| Hodge asymmetry / gradient–curl | **P** = row-stochastic post-softmax | `asymmetry` | decomposes the original attention flow | total asymmetry G, gradient (hierarchical) vs curl (circulatory) split, per-token witness |
| Orientation / tournament (discrete) | **S = QKᵀ** pre-softmax (unmasked) | `cyclic` (`\|T_cyc\|`) | causal post-softmax is transitive ⇒ `\|T_cyc\|=0`; the real tournament is in the raw scores | count of non-transitive (cyclic) attention triangles |
| Orientation / frustration (spectral) | **S = QKᵀ** pre-softmax (unmasked) | `magnetic` (λ₁ + phase-curl) | same post-softmax vacuity; magnetic Laplacian `L_φ=D−A⊙e^{iθ}` encodes the preference orientation as a U(1) phase | spectral frustration `λ₁` (0 ⟺ balanced) + streamable phase-curl energy |
| Score geometry / rank | **S = QKᵀ** pre-softmax | `spectral` | pre-activation spectrum | singular-value structure of the scores |

## The 3 matrices

- **Pre-softmax scores** `S = QKᵀ/√d` — full, **not** causally masked (the mask is applied inside
  the softmax, not to the raw scores). The directional asymmetry `sgn(qᵢ·kⱼ − qⱼ·kᵢ)` and the
  cyclic tournament structure live here and survive causal masking.
- **Post-softmax attention** `P = softmax(masked S)` — row-stochastic (`P·1 = 1`); lower-triangular
  for causal decoders. The actual information-routing operator.
- **Degree-normalized** `M = D_Q^{-1/2} P D_K^{-1/2} = P·D_K^{-1/2}` (`D_Q = I` for row-stochastic
  P; `D_K` = key degrees). The normalized operator whose spectrum reflects conductance independent
  of degree.

## Conductance / Cheeger → M

The Cheeger σ₂ bracket is a statement about the **symmetric normalized** operator. The
normalization is load-bearing: it removes degree heterogeneity so σ₂ bounds conductance. On the
raw matrix the bound is degree-distorted.

## Hodge asymmetry G / Γ / C → P

The asymmetry/Hodge family runs on the **row-stochastic post-softmax attention P**, with
`A = (P − Pᵀ)/2`, **not** on the degree-normalized M. Reasons:

1. **Degree normalization mixes symmetric and antisymmetric components of P.** For row-stochastic P
   (`D_Q = I`), `M = P·D_K^{-1/2}`, and entry-wise

   ```
   M_asym(i,j) = P_asym(i,j)·σ(i,j) + P_sym(i,j)·δ(i,j)
   σ(i,j) = (dᵢ^{-1/2} + dⱼ^{-1/2}) / 2      (symmetric weight)
   δ(i,j) = (dⱼ^{-1/2} − dᵢ^{-1/2}) / 2      (antisymmetric weight)
   ```

   where `d` = positive key degrees. When key degrees differ (`dᵢ ≠ dⱼ`), `P_sym`
   contributes to `M_asym` through δ. A Hodge split on M therefore describes the normalized
   operator's flow; the `asymmetry` signal uses P to describe the original attention flow.
2. **Clean interpretation on P.** The Hodge gradient on P is exactly the in-degree imbalance:
   `A_grad(i,j) = m(i) − m(j)`, `m(i) ∝ (1 − cᵢ)`, `cᵢ = Σⱼ Pⱼᵢ`. The curl is the divergence-free
   residual — the **solenoidal / row-mean projection** (*Beyond Hodge*: `A = A_pot + A_sol`,
   `A_pot = d₀(-m)`, with the Pythagorean energy split `G² = Γ² + C²`), **not** a
   triangle-circulation RMS.
3. **Paper alignment.** *streaming-asym-operators* decomposes the row-stochastic P.

**What it gives:** total asymmetry `G = ‖P_asym‖_F/‖P‖_F`; the gradient (hierarchical) vs curl
(circulatory) split (`G² = Γ² + C²`); and a per-token asymmetry witness.

**Caveat — causal masking.** Under causal masking, post-softmax attention is lower-triangular, so
its asymmetry is **largely a triangular-mask artifact**, not learned structure (`zero-shot-cheeger`
found exactly this for GPT-2 / Pythia; the sign tournament is transitive ⇒ `|T_cyc| = 0`). The
asymmetry axis therefore carries limited diagnostic power for decoder-only models — interpret G
accordingly. It is genuinely informative for *non-causal* attention (encoder / cross-attention).

**G on M vs P.** The `routing` signal reports `asym_index = ‖M_asym‖_F/‖M‖_F`,
the normalized asymmetry of M. The `asymmetry` signal reports `G = ‖P_asym‖_F/‖P‖_F`
and its gradient/curl decomposition (`Γ`, `C`) on P.

## Orientation family → pre-softmax S

Both orientation diagnostics live on the **unmasked pre-softmax scores** and are vacuous on the
causal post-softmax operator (a causal P is lower-triangular ⇒ its tournament is the transitive
position order ⇒ no cycles, no frustration). They are the **discrete** and **spectral** readouts
of the same antisymmetric preference structure `qᵢ·kⱼ − qⱼ·kᵢ`:

- **`|T_cyc|` (discrete, #42)** — the count of non-transitive / cyclic triangles via
  `sgn(qᵢ·kⱼ − qⱼ·kᵢ)`. Streamable exactly (Kendall–Babington-Smith out-degree update). Matches
  `streaming-cyclic-triangles`.
- **`magnetic` frustration `λ₁` (spectral, #41)** — the smallest eigenvalue of the Hermitian
  magnetic Laplacian `L_φ = D − A⊙e^{iθ}`, `W=(|S_ij|+|S_ji|)/2`,
  `θ=arctan((S_ij−S_ji)/(S_ij+S_ji))`. `λ₁ = 0 ⟺` the orientation is balanced (a pure gauge
  gradient); `λ₁ > 0 ⟺` frustration (preference loops that cannot be gauged away). Gauge-invariant,
  so degree normalization (M vs P) leaves it unchanged — which is *why* it sits on S, not on M/P.
  The exact `λ₁` is batch-only; its **faithful streamable companion is the phase-curl energy**
  `‖θ‖²−2‖r_θ‖²/L` (the Hodge curl of the phase field θ, computed by the same row-sum identity as
  the asymmetry curl — additive, exact-streamable; `0 ⟺ balanced`; brackets `λ₁`, #68). See
  [streaming-modes](streaming-modes.md).

## References

SHADE papers:

- *beyond-hodge* — Hodge decomposition of attention operators, normalization invariance
  (symmetric scaling `D·A·D` preserves antisymmetry; asymmetric scaling need not —
  `rem:asymmetric_scaling`), and the gradient/curl energy split `G² = Γ² + C²`.
- *streaming-asym-operators* — Hodge decomposition on the row-stochastic P.
- *zero-shot-cheeger* — conductance on M; asymmetry-is-mask-artifact under causal masking.
- *streaming-cyclic-triangles* — cyclic-triangle tournament on the pre-softmax q·k scores.
- *directed-attention-geometry* — magnetic Laplacian of attention, frustration index, and the
  gauge invariance of `λ₁` (why frustration is normalization-independent).
- *structural-streaming-attention* — the streaming diagnostic suite; magnetic frustration as the
  spectral orientation member on pre-softmax S.
