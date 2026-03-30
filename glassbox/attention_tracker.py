"""
AttentionTracker features from raw post-softmax attention matrix A.

Computes span-independent features from A = softmax(QK^T / sqrt(d)):
  - sigma2: second singular value of A
  - sigma2_asym: second singular value of A_asym = (A - A^T) / 2
  - commutator_norm: ||[A_sym, A_asym]||_F / ||A||_F

Two paths controlled by a sequence-length threshold:
  - Materialized (L <= threshold): dense tensor ops on the L x L matrix A.
  - Matrix-free  (L >  threshold): blocked-streaming matvecs, O(Ld) memory.

The matrix-free path reuses existing M-family matvecs from svd.py and hodge.py
by passing d_k_inv_sqrt = ones, which makes M = A (no degree normalization).

Reference:
    AttentionTracker (arXiv:2411.00348)
"""

from __future__ import annotations

import torch

from glassbox.hodge import (
    EPSILON,
    compute_sigma2_asym_matrix_free,
    estimate_commutator_norm_matrix_free,
)
from glassbox.results import TrackerFeatures
from glassbox.svd import (
    apply_A_blocked,
    apply_AT_blocked,
    compute_M_fro_norm_blocked,
    randomized_svd,
    svd_via_lanczos,
)


def compute_attention_tracker_features_materialized(
    A: torch.Tensor,
    rank: int,
) -> TrackerFeatures:
    """All AttentionTracker features from a materialized attention matrix.

    Args:
        A: Attention matrix of shape (L, L), already softmaxed.
        rank: Number of singular values to retain.

    Returns:
        TrackerFeatures with sigma2, sigma2_asym, commutator_norm.
    """
    sigma = torch.linalg.svdvals(A)
    k = min(rank, len(sigma))
    sigma2 = sigma[1].item() if len(sigma) > 1 else 0.0

    A_sym = (A + A.T) / 2.0
    A_asym = (A - A.T) / 2.0

    sigma_asym = torch.linalg.svdvals(A_asym)
    sigma2_asym = sigma_asym[1].item() if len(sigma_asym) > 1 else 0.0

    comm = A_sym @ A_asym - A_asym @ A_sym
    A_fro = torch.linalg.norm(A, "fro").item()
    commutator_norm = torch.linalg.norm(comm, "fro").item() / (A_fro + EPSILON)

    return TrackerFeatures(
        singular_values=sigma[:k].cpu().tolist(),
        sigma2=sigma2,
        sigma2_asym=sigma2_asym,
        commutator_norm=commutator_norm,
    )


def compute_attention_tracker_features_matrix_free(
    Q: torch.Tensor,
    K: torch.Tensor,
    scale: float,
    rank: int,
    method: str = "randomized",
    block_size: int = 256,
) -> TrackerFeatures:
    """All AttentionTracker features via matrix-free blocked operations.

    Reuses existing matvec infrastructure by passing d_k_inv_sqrt = ones,
    which makes M = A (no degree normalization).

    Args:
        Q: Query tensor of shape (L, d).
        K: Key tensor of shape (L, d).
        scale: Attention scale factor (1 / sqrt(d)).
        rank: Number of singular values to compute.
        method: SVD algorithm ("randomized" or "lanczos").
        block_size: Block size for blocked-streaming matvecs.

    Returns:
        TrackerFeatures with sigma2, sigma2_asym, commutator_norm.
    """
    L = Q.shape[0]
    device = Q.device
    ones = torch.ones(L, device=device, dtype=Q.dtype)

    # --- SVD of A for sigma2 ---
    k = min(max(rank, 2), L - 1)

    def matvec(v):
        return apply_A_blocked(Q, K, v, scale, block_size)

    def matvec_t(u):
        return apply_AT_blocked(Q, K, u, scale, block_size)

    if method == "lanczos":
        _, S, _ = svd_via_lanczos(matvec, matvec_t, L, k, max(2 * k + 2, 20), str(device))
    else:
        _, S, _ = randomized_svd(matvec, matvec_t, L, k, device=str(device))

    S_sorted, _ = torch.sort(S, descending=True)
    sigma2 = S_sorted[1].item() if len(S_sorted) > 1 else 0.0

    # --- sigma2_asym via existing hodge.py (d_k_inv_sqrt=ones -> M=A) ---
    sigma2_asym = compute_sigma2_asym_matrix_free(Q, K, ones, scale, block_size, method)

    # --- ||A||_F via existing svd.py (d_k_inv_sqrt=ones -> ||M||_F = ||A||_F) ---
    A_fro = compute_M_fro_norm_blocked(Q, K, ones, scale, block_size).item()

    # --- commutator_norm via existing hodge.py (d_k_inv_sqrt=ones -> M=A) ---
    commutator_norm = estimate_commutator_norm_matrix_free(Q, K, ones, scale, A_fro, block_size)

    return TrackerFeatures(
        singular_values=S_sorted[:k].cpu().tolist(),
        sigma2=sigma2,
        sigma2_asym=sigma2_asym,
        commutator_norm=commutator_norm,
    )
