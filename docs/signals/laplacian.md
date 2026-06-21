# `laplacian` signal

Eigenvalue features of the **attention graph Laplacian** `L = D_in − A`, where
`D_in[i,i] = Σ_j A[j,i]` (column sums) — attention as a weighted directed graph.

**Operator:** in-degree graph Laplacian `L = D_in − A`. **Measures:** the spectral structure of
the attention graph.

## Outputs (`LaplacianFeatures`)

- `eigvals` — top-`top_k` Laplacian diagonal values, descending. For causal (lower-triangular)
  attention the diagonal entries of `L` are its eigenvalues.

## Enable

```python
cfg = GlassboxConfig(laplacian={"enabled": True, "top_k": 10})
```

Config: `top_k`, `threshold`, `block_size`, `causal`.

## Interpretation

The Laplacian spectrum summarizes connectivity/clustering of the attention graph; used as a
spectral hallucination signal in the reference work.

## Reference

Binkowski et al., "Hallucination Detection in LLMs Using Spectral Features of Attention Maps",
EMNLP 2025 (arXiv:2502.17598).
