# `tracker` signal

Spectral features of the **raw post-softmax attention matrix** `A` (not degree-normalized) —
the AttentionTracker family.

**Operator:** raw post-softmax `A`. **Measures:** span-independent spectral structure of the
realized attention weights.

## Outputs (`TrackerFeatures`)

- `singular_values`, `sv1`, `sv_ratio`, `sv_entropy` — SVD spectrum of `A`.
- `sigma2` — second singular value of `A`.
- `sigma2_asym` — second singular value of `A_asym = (A − Aᵀ)/2`.
- `commutator_norm` — `‖[A_sym, A_asym]‖_F / ‖A‖_F`, coupling of the symmetric and
  antisymmetric parts.

## Enable

```python
cfg = GlassboxConfig(tracker={"enabled": True, "rank": 4, "threshold": 512})
```

Config: `rank`, `method`, `threshold`, `block_size`, `causal`.

## Interpretation

A direct spectral readout of `A` itself (contrast with [`routing`](routing.md), which normalizes
to the transport operator `M`). Useful as a normalization-free baseline.

## Reference

AttentionTracker, arXiv:2411.00348.
