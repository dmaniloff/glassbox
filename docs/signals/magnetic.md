# `magnetic` signal

Magnetic-Laplacian **frustration** of the **unmasked pre-softmax tournament** — whether the
head's directional preferences reconcile into a coherent ranking, or contain irreducible loops.

**Operator:** pre-softmax `S`; the Hermitian magnetic Laplacian `L_φ = D − W⊙e^{iθ}`
(`W=(|S_ij|+|S_ji|)/2`, `θ=arctan((S_ij−S_ji)/(S_ij+S_ji))`). Gauge-invariant, so it is robust
to causal masking and to `M`-vs-`P` normalization.

## Status

**In flight** — PR #67 (`λ₁`), PR #69 (streamable phase-curl). Issue #68 (research note).

## Outputs (`MagneticFeatures`)

- `frustration` (`λ₁`) — smallest eigenvalue of `L_φ`; `0 ⟺ balanced` (a coherent global
  ranking exists), `> 0 ⟺ frustration`. Batch / on-demand (no exact streaming update).
- `phase_curl` — unweighted Hodge curl energy of `θ` (`‖θ‖² − 2‖r_θ‖²/L`); streamable, pure-phase.
- `phase_curl_w` — magnitude-weighted curl (`Σ Wθ² − 2 Σ b²/d`); the **faithful streamable `λ₁`
  proxy** (ρ≈0.97). Recommended for real-time monitoring.
- `witness` — bottom-eigenvector per-token magnitudes (which tokens form the frustrated mode).

## Enable

```python
cfg = GlassboxConfig(magnetic={"enabled": True})                       # batch λ₁ + phase-curls
cfg = GlassboxConfig(magnetic={"enabled": True, "incremental": True},  # streaming phase-curl
                     q_buffer_max_tokens=0)
```

Config: `threshold`, `block_size`, `incremental`.

## Full documentation

See **[magnetic-laplacian.md](../magnetic-laplacian.md)** for the components, the
batch-vs-streaming modes, the which-component-to-read table, and the failure-mode story
(in-context conflict, prompt injection, OOD heads). Pattern: stream `phase_curl_w`, fire `λ₁`
on demand. See also [operator-choice](../operator-choice.md) and
[streaming-modes](../streaming-modes.md).
