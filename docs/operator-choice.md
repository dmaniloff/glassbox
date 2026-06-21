# Operator choice: which attention matrix each diagnostic uses

Attention exposes three matrices, and **each diagnostic family must run on the one its
mathematics requires** — they are not interchangeable. Using the wrong operator silently
changes (or destroys) the quantity being measured. The choices below are grounded in the
SHADE papers (see References).

## TL;DR

| Diagnostic family | Operator | glassbox signal | Why | What it gives |
|---|---|---|---|---|
| Conductance / bottleneck | **M** = degree-normalized post-softmax | `cheeger` | Cheeger σ₂ bracket is a theorem about the *normalized* operator | transport bottleneck bracket `(1−σ₂)/2 ≤ φ ≤ √(2(1−σ₂))` |
| Hodge asymmetry / gradient–curl | **P** = row-stochastic post-softmax | `asymmetry`, `routing` (Hodge part) | degree normalization is an *asymmetric* scaling that inflates the antisymmetric rank; P keeps the clean structure | total asymmetry G, gradient (hierarchical) vs curl (circulatory) split, per-token witness |
| Cyclic triangles / tournament | **S = QKᵀ** pre-softmax (unmasked) | `\|T_cyc\|` (planned) | causal post-softmax is transitive ⇒ `\|T_cyc\|=0`; the real tournament is in the raw scores | count of non-transitive (cyclic) attention triangles |
| Score geometry / rank | **S = QKᵀ** pre-softmax | `spectral` | pre-activation spectrum | singular-value structure of the scores |

## The three operators

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

1. **Asymmetric normalization distorts the antisymmetric structure.** `M = D_Q^{-1/2}(·)D_K^{-1/2}`
   with `D_Q ≠ D_K` is an *asymmetric* scaling. *Beyond Hodge* shows that symmetric scaling
   `D·A·D` preserves antisymmetry, whereas asymmetric `D₁AD₂` (`D₁ ≠ D₂`) does not, and can inflate
   the rank of the antisymmetric part (`rem:asymmetric_scaling`). P (no normalization) keeps the
   clean rank-2 gradient.
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

**G on M vs P.** `G = ‖M_asym‖_F/‖M‖_F` on M is the well-posed *Beyond Hodge* circulation ratio
(Thm 3.1) — fine as a transpose-sensitivity *feature* in the conductance bundle (this is
what `zero-shot-cheeger` and the `routing` signal report). The dedicated **`asymmetry` signal
computes G on P**, the operator of the Hodge/circulation program.

## Cyclic triangles |T_cyc| → pre-softmax S (planned)

`|T_cyc|` (the count of non-transitive / cyclic attention triangles) must be computed on the
**pre-softmax scores** `sgn(qᵢ·kⱼ − qⱼ·kᵢ)`, never on post-softmax causal attention: a causal P is
lower-triangular ⇒ its sign tournament is the transitive position order ⇒ `|T_cyc| = 0` identically.
The pre-softmax q·k tournament is never causally masked, so it survives. Matches
`streaming-cyclic-triangles`.

## References

SHADE papers:

- *beyond-hodge* — Hodge decomposition of attention operators, normalization invariance, the
  circulation ratio (Thm 3.1), and the gradient/curl energy split `G² = Γ² + C²`.
- *streaming-asym-operators* — Hodge decomposition on the row-stochastic P.
- *zero-shot-cheeger* — conductance on M; asymmetry-is-mask-artifact under causal masking.
- *streaming-cyclic-triangles* — cyclic-triangle tournament on the pre-softmax q·k scores.
