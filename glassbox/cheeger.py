"""Bipartite sweep conductance for the degree-normalized cross-operator M.

Computes the Cheeger conductance φ* via sweep cut over the Fiedler-like
singular vectors (u₂, v₂) of M.  Two paths:

  - Materialized: ``bipartite_sweep_conductance(u2, v2, M)``
  - Matrix-free:  ``bipartite_sweep_conductance_matrix_free(u2, v2, Q, K, ...)``
    streams through M in row blocks, O(block_size × L) peak memory.

The spectral gap ``1 - σ₂`` only *bounds* conductance via the Cheeger inequality:

    (1 - σ₂) / 2  ≤  φ*  ≤  √(2(1 - σ₂))

References:
    Cheeger (1970): A lower bound for the smallest eigenvalue of the Laplacian
    Bauer & Jost (2013): Bipartite and neighborhood graphs and the spectrum
        of the normalized graph Laplacian
"""

from __future__ import annotations

import torch


@torch.no_grad()
def bipartite_sweep_conductance(
    u2: torch.Tensor,
    v2: torch.Tensor,
    M: torch.Tensor,
    eps: float = 1e-12,
) -> float:
    """Minimum conductance via bipartite sweep cut on materialized M.

    Args:
        u2: Second left singular vector of M, shape (nL,).
        v2: Second right singular vector of M, shape (nR,).
        M: Degree-normalized cross-operator, shape (nL, nR), non-negative.
        eps: Numerical stability epsilon.

    Returns:
        Minimum conductance φ* in [0, 1].
    """
    target_dtype = torch.float32 if M.device.type == "mps" else torch.float64
    M = M.to(target_dtype).clamp_min(0)
    u2 = u2.to(target_dtype)
    v2 = v2.to(target_dtype)

    nL, nR = M.shape

    degL = M.sum(dim=1).clamp_min(eps)
    degR = M.sum(dim=0).clamp_min(eps)
    vol_total = (degL.sum() + degR.sum()).item()

    if vol_total <= eps:
        return 0.0

    li, rj = (M > 0).nonzero(as_tuple=True)
    if li.numel() == 0:
        return 0.0
    w = M[li, rj]

    return _sweep_both_orientations(
        u2, v2, degL, degR, vol_total, li, rj, w, nL, nR, target_dtype, M.device, eps
    )


@torch.no_grad()
def bipartite_sweep_conductance_matrix_free(
    u2: torch.Tensor,
    v2: torch.Tensor,
    Q: torch.Tensor,
    K: torch.Tensor,
    d_k_inv_sqrt: torch.Tensor,
    scale: float,
    block_size: int = 256,
    causal: bool = False,
    eps: float = 1e-12,
) -> float:
    """Minimum conductance via bipartite sweep cut, matrix-free.

    Streams through M in row blocks to avoid materializing the full L×L
    matrix, following the same blocked pattern as ``compute_M_fro_norm_blocked``
    in ``svd.py``.

    Args:
        u2: Second left singular vector from SVD of M, shape (L,).
        v2: Second right singular vector from SVD of M, shape (L,).
        Q: Query matrix, shape (L, d).
        K: Key matrix, shape (L, d).
        d_k_inv_sqrt: Inverse sqrt of key degrees, shape (L,).
        scale: Attention scale factor (1 / sqrt(d)).
        block_size: Row block size for streaming.
        causal: Whether to apply causal masking.
        eps: Numerical stability epsilon.

    Returns:
        Minimum conductance φ* in [0, 1].
    """
    L = Q.shape[0]
    device = Q.device
    target_dtype = torch.float32 if device.type == "mps" else torch.float64

    u2 = u2.to(target_dtype)
    v2 = v2.to(target_dtype)

    nL = nR = L
    N = nL + nR

    # Compute degrees by streaming through M blocks
    degL = torch.zeros(L, device=device, dtype=target_dtype)
    degR = torch.zeros(L, device=device, dtype=target_dtype)

    # Also pre-compute ranks (independent of M)
    mean_u2 = u2.mean()
    mean_v2 = v2.mean()
    if (mean_u2 + mean_v2) < 0:
        u2_canon = -u2
        v2_canon = -v2
    else:
        u2_canon = u2
        v2_canon = v2

    x = torch.cat([u2_canon, v2_canon], dim=0)
    order = torch.argsort(x)
    rank = torch.empty_like(order)
    rank[order] = torch.arange(N, device=device, dtype=order.dtype)

    # Pass 1: compute degrees and accumulate cut_delta in one streaming pass
    cut_delta = torch.zeros(N, device=device, dtype=target_dtype)
    vol_delta = torch.zeros(N, device=device, dtype=target_dtype)

    for i0 in range(0, L, block_size):
        i1 = min(i0 + block_size, L)
        scores = Q[i0:i1] @ K.T * scale
        if causal:
            from glassbox.svd import _mask_causal

            scores = _mask_causal(scores, i0)
        attn = torch.softmax(scores, dim=-1)
        M_block = (attn * d_k_inv_sqrt.unsqueeze(0)).to(target_dtype)

        # Accumulate row degrees for this block
        degL[i0:i1] = M_block.sum(dim=1)
        # Accumulate column degrees
        degR += M_block.sum(dim=0)

        bs = i1 - i0
        for r in range(bs):
            global_i = i0 + r
            m_row = M_block[r]
            mask = m_row > 0
            if not mask.any():
                continue
            cols = mask.nonzero(as_tuple=True)[0]
            w_edges = m_row[cols]

            rL = rank[global_i].expand(cols.shape[0])
            rR = rank[nL + cols]
            rmin = torch.minimum(rL, rR)
            rmax = torch.maximum(rL, rR)

            cut_delta.scatter_add_(0, rmin, w_edges)
            cut_delta.scatter_add_(0, rmax, -w_edges)

    degL = degL.clamp_min(eps)
    degR = degR.clamp_min(eps)
    vol_total = (degL.sum() + degR.sum()).item()

    if vol_total <= eps:
        return 0.0

    # Compute volumes
    vol_delta.scatter_add_(0, rank[:nL], degL)
    vol_delta.scatter_add_(0, rank[nL:], degR)
    volS = torch.cumsum(vol_delta, dim=0)

    # Compute cut
    cut = torch.cumsum(cut_delta, dim=0)

    # Conductance for first orientation
    den = torch.minimum(volS, vol_total - volS)
    mask = den > eps
    if not mask.any():
        phi1 = 1.0
    else:
        phi_all = cut[mask] / den[mask]
        phi1 = float(phi_all.min().item())

    sign_confidence = abs(float(mean_u2 + mean_v2))

    if phi1 >= 0 and sign_confidence > 0.1:
        return max(0.0, min(phi1, 1.0))

    # Second orientation: recompute with flipped signs
    # Re-stream M blocks for flipped cut_delta
    x_flip = torch.cat([-u2_canon, -v2_canon], dim=0)
    order_flip = torch.argsort(x_flip)
    rank_flip = torch.empty_like(order_flip)
    rank_flip[order_flip] = torch.arange(N, device=device, dtype=order_flip.dtype)

    cut_delta_flip = torch.zeros(N, device=device, dtype=target_dtype)
    vol_delta_flip = torch.zeros(N, device=device, dtype=target_dtype)

    for i0 in range(0, L, block_size):
        i1 = min(i0 + block_size, L)
        scores = Q[i0:i1] @ K.T * scale
        if causal:
            from glassbox.svd import _mask_causal

            scores = _mask_causal(scores, i0)
        attn = torch.softmax(scores, dim=-1)
        M_block = (attn * d_k_inv_sqrt.unsqueeze(0)).to(target_dtype)

        bs = i1 - i0
        for r in range(bs):
            global_i = i0 + r
            m_row = M_block[r]
            mask_r = m_row > 0
            if not mask_r.any():
                continue
            cols = mask_r.nonzero(as_tuple=True)[0]
            w_edges = m_row[cols]

            rL = rank_flip[global_i].expand(cols.shape[0])
            rR = rank_flip[nL + cols]
            rmin = torch.minimum(rL, rR)
            rmax = torch.maximum(rL, rR)

            cut_delta_flip.scatter_add_(0, rmin, w_edges)
            cut_delta_flip.scatter_add_(0, rmax, -w_edges)

    vol_delta_flip.scatter_add_(0, rank_flip[:nL], degL)
    vol_delta_flip.scatter_add_(0, rank_flip[nL:], degR)
    volS_flip = torch.cumsum(vol_delta_flip, dim=0)
    cut_flip = torch.cumsum(cut_delta_flip, dim=0)

    den_flip = torch.minimum(volS_flip, vol_total - volS_flip)
    mask_flip = den_flip > eps
    if not mask_flip.any():
        phi2 = 1.0
    else:
        phi_all_flip = cut_flip[mask_flip] / den_flip[mask_flip]
        phi2 = float(phi_all_flip.min().item())

    return _pick_best_phi(phi1, phi2)


def _sweep_both_orientations(
    u2: torch.Tensor,
    v2: torch.Tensor,
    degL: torch.Tensor,
    degR: torch.Tensor,
    vol_total: float,
    li: torch.Tensor,
    rj: torch.Tensor,
    w: torch.Tensor,
    nL: int,
    nR: int,
    target_dtype: torch.dtype,
    device: torch.device,
    eps: float,
) -> float:
    """Try both sign orientations and return best conductance."""
    N = nL + nR

    # Canonicalize sign
    mean_u2 = (u2 * degL).sum() / (degL.sum() + eps)
    mean_v2 = (v2 * degR).sum() / (degR.sum() + eps)
    sign_confidence = abs(float(mean_u2 + mean_v2))

    if (mean_u2 + mean_v2) < 0:
        u2 = -u2
        v2 = -v2

    phi1 = _sweep_one_orientation(
        u2, v2, degL, degR, vol_total, li, rj, w, nL, nR, N, target_dtype, device, eps
    )

    if phi1 >= 0 and sign_confidence > 0.1:
        return float(phi1)

    phi2 = _sweep_one_orientation(
        -u2, -v2, degL, degR, vol_total, li, rj, w, nL, nR, N, target_dtype, device, eps
    )

    return _pick_best_phi(phi1, phi2)


def _sweep_one_orientation(
    u2: torch.Tensor,
    v2: torch.Tensor,
    degL: torch.Tensor,
    degR: torch.Tensor,
    vol_total: float,
    li: torch.Tensor,
    rj: torch.Tensor,
    w: torch.Tensor,
    nL: int,
    nR: int,
    N: int,
    target_dtype: torch.dtype,
    device: torch.device,
    eps: float,
) -> float:
    """Vectorized sweep for a single sign orientation."""
    x = torch.cat([u2, v2], dim=0)
    order = torch.argsort(x)
    rank = torch.empty_like(order)
    rank[order] = torch.arange(N, device=device, dtype=order.dtype)

    rL = rank[li]
    rR = rank[nL + rj]
    rmin = torch.minimum(rL, rR)
    rmax = torch.maximum(rL, rR)

    cut_delta = torch.zeros(N, device=device, dtype=target_dtype)
    cut_delta.scatter_add_(0, rmin, w)
    cut_delta.scatter_add_(0, rmax, -w)
    cut = torch.cumsum(cut_delta, dim=0)

    vol_delta = torch.zeros(N, device=device, dtype=target_dtype)
    vol_delta.scatter_add_(0, rank[:nL], degL)
    vol_delta.scatter_add_(0, rank[nL:], degR)
    volS = torch.cumsum(vol_delta, dim=0)

    den = torch.minimum(volS, vol_total - volS)
    mask = den > eps
    if not mask.any():
        return 1.0

    phi_all = cut[mask] / den[mask]
    return float(phi_all.min().item())


def _pick_best_phi(phi1: float, phi2: float) -> float:
    """Select best conductance from two orientations, clamped to [0, 1]."""
    if phi1 < 0 and phi2 >= 0:
        result = phi2
    elif phi2 < 0 and phi1 >= 0:
        result = phi1
    elif phi1 >= 0 and phi2 >= 0:
        result = min(phi1, phi2)
    else:
        result = max(0.0, min(phi1, phi2))
    return max(0.0, min(result, 1.0))
