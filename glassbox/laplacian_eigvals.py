"""
Laplacian eigenvalue features from attention graphs (LapEigvals, EMNLP 2025).

Treats the attention matrix A as a weighted directed graph adjacency matrix
and computes the in-degree graph Laplacian L = D_in - A, where
D_in[i,i] = sum_j A[j,i] (column sums of A).

For causal (lower-triangular) attention matrices, L is also triangular,
so its eigenvalues are its diagonal entries --- no eigendecomposition needed.
The features are the top-k largest diagonal values, sorted descending.

Two paths controlled by a sequence-length threshold:
  - Materialized (L <= threshold): dense softmax, column sums, diagonal.
  - Matrix-free  (L >  threshold): blocked column sums + diagonal, O(block_size * L) memory.

Reference:
    Binkowski et al., "Hallucination Detection in LLMs Using Spectral
    Features of Attention Maps", EMNLP 2025 (arXiv:2502.17598).
    https://github.com/graphml-lab-pwr/lapeigvals
"""

from __future__ import annotations

import torch

from glassbox.results import LaplacianFeatures
from glassbox.svd import apply_AT_blocked, compute_logsumexp_blocked


def compute_laplacian_eigvals_materialized(
    A: torch.Tensor,
    top_k: int = 10,
) -> LaplacianFeatures:
    """Laplacian diagonal features from a materialized attention matrix.

    Args:
        A: Attention matrix of shape (L, L), already softmaxed.
        top_k: Number of largest diagonal values to keep.

    Returns:
        LaplacianFeatures with sorted eigvals.
    """
    d_col = A.sum(dim=0)  # in-degree: column sums [L]
    diag_A = A.diag()  # self-attention [L]
    diag_L = d_col - diag_A  # Laplacian diagonal [L]

    k = min(top_k, diag_L.shape[0])
    topk_vals, _ = torch.topk(diag_L, k)

    return LaplacianFeatures(eigvals=topk_vals.tolist())


def compute_laplacian_eigvals_matrix_free(
    Q: torch.Tensor,
    K: torch.Tensor,
    scale: float,
    top_k: int = 10,
    block_size: int = 256,
    causal: bool = False,
) -> LaplacianFeatures:
    """Laplacian diagonal features via blocked computation.

    Avoids materializing the L x L attention matrix by computing:
      - diag(A)[i] = exp(s_ii - lse[i])  where s_ii = Q[i]·K[i] * scale
      - col_sums = A^T @ 1  via blocked streaming (apply_AT_blocked)

    Args:
        Q: Query tensor of shape (L, d).
        K: Key tensor of shape (L, d).
        scale: Attention scale factor (1 / sqrt(d)).
        top_k: Number of largest diagonal values to keep.
        block_size: Block size for blocked computation.
        causal: Apply causal mask (token i attends only to j <= i).

    Returns:
        LaplacianFeatures with sorted eigvals.
    """
    L = Q.shape[0]

    # Diagonal of A: diag(A)[i] = exp(s_ii - lse[i])
    s_diag = (Q * K).sum(dim=-1) * scale  # [L]
    lse = compute_logsumexp_blocked(Q, K, scale, block_size, causal=causal)  # [L]
    diag_A = torch.exp(s_diag - lse)  # [L]

    # Column sums of A: d_col[j] = sum_i A[i,j] = (A^T @ 1)[j]
    ones = torch.ones(L, device=Q.device, dtype=Q.dtype)
    d_col = apply_AT_blocked(Q, K, ones, scale, block_size, causal=causal)  # [L]

    # Laplacian diagonal
    diag_L = d_col - diag_A  # [L]

    k = min(top_k, L)
    topk_vals, _ = torch.topk(diag_L, k)

    return LaplacianFeatures(eigvals=topk_vals.tolist())
