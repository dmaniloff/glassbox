# `cyclic` signal

Count of **non-transitive (cyclic) attention triangles** in the sign tournament of the
**unmasked pre-softmax scores** — `sgn(qᵢ·kⱼ − qⱼ·kᵢ)`.

**Operator:** pre-softmax `S` (sign tournament). **Measures:** how non-transitive the discrete
preference structure is. Vacuous on causal post-softmax (a triangular `P` is transitive ⇒
`|T_cyc| = 0`), so it must be read on `S`.

## Status

**In flight** — PR #63 (issue #42).

## Outputs (`CyclicTrianglesFeatures`)

- `T_cyc` — number of cyclic triples `{i,j,k}` (Kendall–Babington-Smith identity
  `|T_cyc| = C(n,3) − Σ_i C(s_i, 2)`, `s_i` = out-degree).

## Streaming

`|T_cyc|` is **not additive** across windows (cross-window triangles), so no block-diagonal
global mode — but it has an **exact O(Δ) incremental** update (the out-degree fold), so it
supports local-block + exact-full streaming. See [streaming-modes](../streaming-modes.md).

## Relation to `magnetic`

`cyclic` is the **discrete** orientation member; [`magnetic`](magnetic.md) is the **spectral**,
magnitude-weighted one. `λ₁ = 0 ⇒ |T_cyc| = 0` but not conversely — magnetic frustration is a
strictly finer balance condition. See [operator-choice](../operator-choice.md).
