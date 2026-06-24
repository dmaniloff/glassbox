# `asymmetry` signal

Hodge gradient/curl decomposition of the routing asymmetry on the **row-stochastic post-softmax
operator** `P` — the clean operator for the circulation program (degree normalization is an
asymmetric scaling that inflates the antisymmetric rank, so this lives on `P`, not `M`).

**Operator:** row-stochastic `P`. **Measures:** how much of the directed routing is a consistent
hierarchy (gradient) vs circulatory (curl).

## Status

**In flight** — PRs #58 (G via Hutchinson), #60 (incremental), #62 (gradient/curl split).

## Outputs (`AsymmetryFeatures`)

- `G` — total asymmetry `‖P_asym‖_F / ‖P‖_F`.
- `Gamma` — gradient (hierarchical) coefficient.
- `C` — curl (circulatory) coefficient; genuine Pythagorean `G² = Γ² + C²` via the row-sum Hodge
  identity `‖A_grad‖² = 2‖r‖²/L`, `r = P_asym·1`.

## Streaming

Frobenius sums-of-squares are **additive**, so `asymmetry` supports all modes: local-block,
block-diagonal global (`streaming`, tumbling), and exact-full (`incremental`). See
[streaming-modes](../streaming-modes.md).

## See also

[operator-choice](../operator-choice.md) (Hodge → `P`). The same row-sum identity powers the
routing curl and the magnetic [phase-curl](magnetic.md).
