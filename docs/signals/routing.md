# `routing` signal

SVD + Hodge decomposition + conductance of the **degree-normalized post-softmax operator**
`M = D_Q^{-1/2} A D_K^{-1/2}` — the transport view of attention routing.

**Operator:** degree-normalized `M`. **Measures:** spectral structure, transport bottleneck
(conductance), and the gradient/curl decomposition of the routing asymmetry.

## Outputs (`RoutingFeatures`)

- `singular_values`, `sv1`, `sv_ratio`, `sv_entropy` — SVD spectrum of `M`.
- `phi_hat` — Cheeger conductance via the bipartite sweep cut (transport bottleneck).
- `sigma2` — second singular value of `M` (the Cheeger spectral side).
- `G` — total asymmetry `‖M_asym‖_F / ‖M‖_F`.
- `Gamma`, `C` — gradient (hierarchical) vs curl (circulatory) split of the asymmetry,
  `G² = Γ² + C²` (row-sum Hodge identity).
- `curl_ratio` — `C / (G + ε)`, the circulatory share.
- `sigma2_asym`, `commutator_norm` — antisymmetric spectrum and sym/asym coupling.

## Enable

```python
cfg = GlassboxConfig(routing={"enabled": True, "rank": 4, "threshold": 512})
```

Config: `rank`, `method`, `threshold` (dense `M` for `L ≤ threshold`, else matrix-free),
`block_size`, `causal`.

## Interpretation

Low `sigma2` / low `phi_hat` ⇒ a transport bottleneck (poorly-mixed routing). High `curl_ratio`
⇒ the asymmetric routing is circulatory rather than a clean hierarchy. `phi_hat`/`sigma2` are
the live conductance bracket (see [conductance](conductance.md)).

## See also

[operator-choice](../operator-choice.md) · [streaming-modes](../streaming-modes.md).
