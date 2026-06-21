# `selfattn` signal

Direct scalar/vector statistics from the **attention matrix diagonal** `diag(A)` — no SVD.

**Operator:** `diag(A)` (self-attention weights). **Measures:** how strongly each token attends
to itself.

## Outputs (`SelfAttnFeatures`)

- `attn_diag_logmean` — `mean_i(log A[i,i])`; higher ⇒ stronger self-attention (LLM-Check).
- `eigvals` — top-`top_k` diagonal values of `A`, descending. For causal (lower-triangular)
  attention these are the eigenvalues of `A` (the LapEigvals baseline).

## Enable

```python
cfg = GlassboxConfig(selfattn={"enabled": True, "top_k": 10})
```

Config: `top_k`, `threshold`, `block_size`, `causal`.

## Interpretation

Elevated `attn_diag_logmean` indicates tokens hoarding attention on themselves (a pattern linked
to hallucination in LLM-Check). Cheap to compute (diagonal only).

## References

LLM-Check (NeurIPS 2024); LapEigvals (EMNLP 2025, arXiv:2502.17598).
