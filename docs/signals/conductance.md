# `conductance` / Cheeger signal

The transport-bottleneck bracket on the **degree-normalized operator** `M` — Cheeger's
inequality `(1 − σ₂)/2 ≤ φ ≤ √(2(1 − σ₂))`.

**Operator:** degree-normalized `M` (post-softmax). **Measures:** whether information can mix
across the sequence, or is partitioned into weakly-connected clusters.

## Status

**Emitted today** by the [`routing`](routing.md) signal as `phi_hat` (sweep-cut conductance) and
`sigma2` (spectral side) on `M`. A **dedicated streaming** version (bordered Rayleigh–Ritz σ₂
tracking) is in flight (#38/#53).

## Outputs

- `phi_hat` — Cheeger conductance via the bipartite sweep cut.
- `sigma2` — second singular value of `M` (the spectral bracket side).

## Interpretation

Low `σ₂` / low `φ̂` ⇒ a bottleneck (poorly-mixed routing, near-disconnected token clusters).
This is the `M`-operator member of the [operator taxonomy](../operator-choice.md)
(Cheeger→`M`, Hodge→`P`, orientation→pre-softmax `S`).

## Streaming

Conductance is **spectral**, so it is local-block + full-recompute only (no exact block-diagonal
or incremental update — a block-diagonal graph is disconnected, `φ→0`). See
[streaming-modes](../streaming-modes.md); the streaming-Cheeger line maintains an *approximate*
`σ₂`.
