"""
Attention diagonal features from LLM-Check (NeurIPS 2024).

Computes mean log self-attention weight: mean_i(log(A[i,i])),
where A = softmax(QK^T / sqrt(d)).

This captures how strongly each token attends to itself. Higher values
indicate stronger self-attention, which correlates with model confidence
and factuality per the LLM-Check paper.

Two paths controlled by a sequence-length threshold:
  - Materialized (L <= threshold): dense softmax, extract diagonal.
  - Matrix-free  (L >  threshold): blocked logsumexp, O(block_size * L) memory.

The matrix-free path avoids materializing the L×L attention matrix by
computing log(diag(A)[i]) = s_ii - logsumexp_i, where s_ii is the
diagonal of the scores matrix (cheap: O(Ld)) and logsumexp_i is the
row-wise log-sum-exp (via existing compute_logsumexp_blocked).

Reference:
    LLM-Check (NeurIPS 2024)
    https://github.com/GaurangSriramanan/LLM_Check_Hallucination_Detection
"""

from __future__ import annotations

import torch

from glassbox.results import AttentionDiagonalFeatures
from glassbox.svd import compute_logsumexp_blocked

EPSILON = 1e-10


def compute_attention_diagonal_features_materialized(
    A: torch.Tensor,
) -> AttentionDiagonalFeatures:
    """Attention diagonal features from a materialized attention matrix.

    Args:
        A: Attention matrix of shape (L, L), already softmaxed.

    Returns:
        AttentionDiagonalFeatures with attn_diag_logmean.
    """
    diag_A = A.diag()
    log_diag = torch.log(diag_A + EPSILON)
    attn_diag_logmean = log_diag.mean().item()

    return AttentionDiagonalFeatures(attn_diag_logmean=attn_diag_logmean)


def compute_attention_diagonal_features_matrix_free(
    Q: torch.Tensor,
    K: torch.Tensor,
    scale: float,
    block_size: int = 256,
) -> AttentionDiagonalFeatures:
    """Attention diagonal features via blocked computation.

    Avoids materializing L×L by computing:
        log(diag(A)[i]) = s_ii - logsumexp_i
    where s_ii = Q[i]·K[i] * scale (O(Ld)) and logsumexp is blocked.

    Args:
        Q: Query tensor of shape (L, d).
        K: Key tensor of shape (L, d).
        scale: Attention scale factor (1 / sqrt(d)).
        block_size: Block size for blocked logsumexp.

    Returns:
        AttentionDiagonalFeatures with attn_diag_logmean.
    """
    # Diagonal of scores matrix: s_ii = Q[i] · K[i] * scale
    s_diag = (Q * K).sum(dim=-1) * scale  # [L]

    # Row-wise logsumexp via existing blocked implementation
    lse = compute_logsumexp_blocked(Q, K, scale, block_size)  # [L]

    # log(diag(A)[i]) = s_ii - logsumexp_i (numerically stable, no exp needed)
    log_diag = s_diag - lse  # [L]
    attn_diag_logmean = log_diag.mean().item()

    return AttentionDiagonalFeatures(attn_diag_logmean=attn_diag_logmean)
