# `spectral` signal

SVD of the **pre-softmax scores** `S = QKᵀ/√d` — the score-geometry / rank structure of the
raw attention logits, before softmax and masking.

**Operator:** pre-softmax `S` (the pre-activation spectrum). **Measures:** how concentrated vs
spread the score energy is across singular directions.

## Outputs (`SpectralFeatures`)

- `singular_values` — top-`rank` singular values of `S` (descending).
- `sv1` — leading singular value.
- `sv_ratio` — `σ₁/σ₂` (dominance of the top direction).
- `sv_entropy` — entropy of the normalized singular-value distribution (spread).

## Enable

```python
from glassbox.config import GlassboxConfig
cfg = GlassboxConfig(spectral={"enabled": True, "rank": 4, "method": "randomized"})
```

Config: `rank` (number of singular values), `method` (`"randomized"` | `"lanczos"`).
`spectral` is the only signal enabled by default.

## Interpretation

High `sv_ratio` / low `sv_entropy` ⇒ a single dominant score direction (focused attention
geometry); flat spectrum ⇒ diffuse. This is the unnormalized pre-softmax view; for the
transport operator use [`routing`](routing.md).

## See also

[operator-choice](../operator-choice.md) (score geometry → `S`).
