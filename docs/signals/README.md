# Signals reference

Each glassbox **signal** is a `Diagnostic` that reads a layer's `(Q, K)` and emits a structured
feature record per `(request, layer, head, step)`. Enable signals via `GlassboxConfig` (or
`--signal <name>` on the CLI); see each signal's page for its config fields and outputs.

| Signal | Operator | Measures | Status | Doc |
|---|---|---|---|---|
| `spectral` | pre-softmax `S = QKᵀ` | score-geometry rank / spectrum | merged | [spectral](spectral.md) |
| `routing` | degree-normalized `M` (post-softmax) | SVD + Hodge `G/Γ/C` + conductance | merged | [routing](routing.md) |
| `conductance` | `M` (post-softmax) | transport bottleneck (`φ̂`, `σ₂` bracket) | emitted via `routing`; streaming in flight (#38/#53) | [conductance](conductance.md) |
| `tracker` | raw post-softmax `A` | AttentionTracker spectral features | merged | [tracker](tracker.md) |
| `selfattn` | `diag(A)` | self-attention magnitude / eigenvalues | merged | [selfattn](selfattn.md) |
| `laplacian` | attention graph Laplacian `L = D_in − A` | Laplacian eigenvalues | merged | [laplacian](laplacian.md) |
| `asymmetry` | row-stochastic `P` (post-softmax) | Hodge gradient/curl split `G/Γ/C` | in flight (#58/#60/#62) | [asymmetry](asymmetry.md) |
| `cyclic` | pre-softmax `S` (sign tournament) | cyclic-triangle count `\|T_cyc\|` | in flight (#42/#63) | [cyclic](cyclic.md) |
| `magnetic` | pre-softmax `S` (magnetic Laplacian) | orientation frustration `λ₁` + phase-curl | in flight (#41/#67, #68/#69) | [magnetic](magnetic.md) |

**Cross-cutting references**

- [operator-choice](../operator-choice.md) — *why* each diagnostic lives on its operator
  (Cheeger→`M`, Hodge→`P`, orientation→pre-softmax `S`).
- [streaming-modes](../streaming-modes.md) — which signals support local-block, block-diagonal
  global (`streaming`), and exact-full (`incremental`) accumulation, and why.

All signals share the orchestration fields `enabled`, `interval` (fire every N tokens), and
`heads` (which heads to instrument).
