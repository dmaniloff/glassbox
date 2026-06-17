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

import math
from dataclasses import dataclass

import torch

from glassbox.results import CheegerFeatures
from glassbox.svd import (
    compute_dk_blocked,
    compute_degree_normalized_M,
    matvec_M_blocked,
    matvec_MT_blocked,
    randomized_svd,
    svd_via_lanczos,
)


@dataclass
class SweepResult:
    """Result of a bipartite sweep conductance computation."""

    phi_star: float
    cut_membership: torch.Tensor  # [2L] int: 1 if in S, 0 if not


@torch.no_grad()
def bipartite_sweep_conductance(
    u2: torch.Tensor,
    v2: torch.Tensor,
    M: torch.Tensor,
    eps: float = 1e-12,
) -> SweepResult:
    """Minimum conductance via bipartite sweep cut on materialized M.

    Args:
        u2: Second left singular vector of M, shape (nL,).
        v2: Second right singular vector of M, shape (nR,).
        M: Degree-normalized cross-operator, shape (nL, nR), non-negative.
        eps: Numerical stability epsilon.

    Returns:
        SweepResult with phi_star and cut_membership [nL + nR].
    """
    target_dtype = torch.float32 if M.device.type == "mps" else torch.float64
    M = M.to(target_dtype).clamp_min(0)
    u2 = u2.to(target_dtype)
    v2 = v2.to(target_dtype)

    nL, nR = M.shape
    N = nL + nR

    degL = M.sum(dim=1).clamp_min(eps)
    degR = M.sum(dim=0).clamp_min(eps)
    vol_total = (degL.sum() + degR.sum()).item()

    if vol_total <= eps:
        return SweepResult(phi_star=0.0, cut_membership=torch.zeros(N, dtype=torch.long))

    li, rj = (M > 0).nonzero(as_tuple=True)
    if li.numel() == 0:
        return SweepResult(phi_star=0.0, cut_membership=torch.zeros(N, dtype=torch.long))
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
) -> SweepResult:
    """Minimum conductance via bipartite sweep cut, matrix-free.

    Streams through M in row blocks to avoid materializing the full L×L
    matrix. Fully vectorized — no Python per-row loops.

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
        SweepResult with phi_star and cut_membership [2L].
    """
    L = Q.shape[0]
    device = Q.device
    target_dtype = torch.float32 if device.type == "mps" else torch.float64

    u2 = u2.to(target_dtype)
    v2 = v2.to(target_dtype)

    nL = nR = L
    N = nL + nR

    # Canonicalize sign orientation
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
    rank = torch.empty(N, device=device, dtype=torch.long)
    rank[order] = torch.arange(N, device=device, dtype=torch.long)

    # Streaming pass: compute degrees and cut_delta in one pass (vectorized)
    degL = torch.zeros(L, device=device, dtype=target_dtype)
    degR = torch.zeros(L, device=device, dtype=target_dtype)
    cut_delta = torch.zeros(N, device=device, dtype=target_dtype)

    rank_R = rank[nL:]  # [L] — ranks of right (key) nodes

    for i0 in range(0, L, block_size):
        i1 = min(i0 + block_size, L)
        scores = Q[i0:i1] @ K.T * scale
        if causal:
            from glassbox.svd import _mask_causal

            scores = _mask_causal(scores, i0)
        attn = torch.softmax(scores, dim=-1)
        M_block = (attn * d_k_inv_sqrt.unsqueeze(0)).to(target_dtype)  # [bs, L]

        # Accumulate degrees
        degL[i0:i1] = M_block.sum(dim=1)
        degR += M_block.sum(dim=0)

        # Vectorized cut_delta scatter — no per-row loop
        bs = i1 - i0
        rank_L_block = rank[i0:i1]  # [bs]
        rL_expanded = rank_L_block.unsqueeze(1).expand(bs, L)  # [bs, L]
        rR_expanded = rank_R.unsqueeze(0).expand(bs, L)  # [bs, L]
        rmin = torch.minimum(rL_expanded, rR_expanded)  # [bs, L]
        rmax = torch.maximum(rL_expanded, rR_expanded)  # [bs, L]

        flat_w = M_block.reshape(-1)
        mask = flat_w > eps
        cut_delta.scatter_add_(0, rmin.reshape(-1)[mask], flat_w[mask])
        cut_delta.scatter_add_(0, rmax.reshape(-1)[mask], -flat_w[mask])

    degL = degL.clamp_min(eps)
    degR = degR.clamp_min(eps)
    vol_total = (degL.sum() + degR.sum()).item()

    if vol_total <= eps:
        return SweepResult(phi_star=0.0, cut_membership=torch.zeros(N, dtype=torch.long))

    # Compute volumes via cumsum
    vol_delta = torch.zeros(N, device=device, dtype=target_dtype)
    vol_delta.scatter_add_(0, rank[:nL], degL)
    vol_delta.scatter_add_(0, rank[nL:], degR)
    volS = torch.cumsum(vol_delta, dim=0)

    # Compute cut via cumsum
    cut = torch.cumsum(cut_delta, dim=0)

    # Conductance for first orientation
    den = torch.minimum(volS, vol_total - volS)
    valid = den > eps
    if not valid.any():
        phi1 = 1.0
        argmin1 = 0
    else:
        phi_all = torch.where(valid, cut / den, torch.tensor(float("inf"), device=device))
        argmin1 = int(phi_all.argmin().item())
        phi1 = float(phi_all[argmin1].item())

    sign_confidence = abs(float(mean_u2 + mean_v2))

    if phi1 >= 0 and sign_confidence > 0.1:
        membership = torch.zeros(N, dtype=torch.long, device=device)
        membership[order[: argmin1 + 1]] = 1
        return SweepResult(phi_star=max(0.0, min(phi1, 1.0)), cut_membership=membership)

    # Second orientation: reverse order
    order_flip = order.flip(0)
    rank_flip = torch.empty(N, device=device, dtype=torch.long)
    rank_flip[order_flip] = torch.arange(N, device=device, dtype=torch.long)

    cut_delta_flip = torch.zeros(N, device=device, dtype=target_dtype)
    rank_R_flip = rank_flip[nL:]

    for i0 in range(0, L, block_size):
        i1 = min(i0 + block_size, L)
        scores = Q[i0:i1] @ K.T * scale
        if causal:
            from glassbox.svd import _mask_causal

            scores = _mask_causal(scores, i0)
        attn = torch.softmax(scores, dim=-1)
        M_block = (attn * d_k_inv_sqrt.unsqueeze(0)).to(target_dtype)

        bs = i1 - i0
        rank_L_block = rank_flip[i0:i1]
        rL_expanded = rank_L_block.unsqueeze(1).expand(bs, L)
        rR_expanded = rank_R_flip.unsqueeze(0).expand(bs, L)
        rmin = torch.minimum(rL_expanded, rR_expanded)
        rmax = torch.maximum(rL_expanded, rR_expanded)

        flat_w = M_block.reshape(-1)
        mask = flat_w > eps
        cut_delta_flip.scatter_add_(0, rmin.reshape(-1)[mask], flat_w[mask])
        cut_delta_flip.scatter_add_(0, rmax.reshape(-1)[mask], -flat_w[mask])

    vol_delta_flip = torch.zeros(N, device=device, dtype=target_dtype)
    vol_delta_flip.scatter_add_(0, rank_flip[:nL], degL)
    vol_delta_flip.scatter_add_(0, rank_flip[nL:], degR)
    volS_flip = torch.cumsum(vol_delta_flip, dim=0)
    cut_flip = torch.cumsum(cut_delta_flip, dim=0)

    den_flip = torch.minimum(volS_flip, vol_total - volS_flip)
    valid_flip = den_flip > eps
    if not valid_flip.any():
        phi2 = 1.0
        argmin2 = 0
    else:
        phi_all_flip = torch.where(
            valid_flip, cut_flip / den_flip, torch.tensor(float("inf"), device=device)
        )
        argmin2 = int(phi_all_flip.argmin().item())
        phi2 = float(phi_all_flip[argmin2].item())

    # Pick best orientation
    if phi1 <= phi2:
        membership = torch.zeros(N, dtype=torch.long, device=device)
        membership[order[: argmin1 + 1]] = 1
        best_phi = phi1
    else:
        membership = torch.zeros(N, dtype=torch.long, device=device)
        membership[order_flip[: argmin2 + 1]] = 1
        best_phi = phi2

    return SweepResult(phi_star=max(0.0, min(best_phi, 1.0)), cut_membership=membership)


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
) -> SweepResult:
    """Try both sign orientations and return best conductance with cut membership."""
    N = nL + nR

    mean_u2 = (u2 * degL).sum() / (degL.sum() + eps)
    mean_v2 = (v2 * degR).sum() / (degR.sum() + eps)
    sign_confidence = abs(float(mean_u2 + mean_v2))

    if (mean_u2 + mean_v2) < 0:
        u2 = -u2
        v2 = -v2

    phi1, argmin1, order1 = _sweep_one_orientation(
        u2, v2, degL, degR, vol_total, li, rj, w, nL, nR, N, target_dtype, device, eps
    )

    if phi1 >= 0 and sign_confidence > 0.1:
        membership = torch.zeros(N, dtype=torch.long, device=device)
        membership[order1[: argmin1 + 1]] = 1
        return SweepResult(phi_star=max(0.0, min(float(phi1), 1.0)), cut_membership=membership)

    phi2, argmin2, order2 = _sweep_one_orientation(
        -u2, -v2, degL, degR, vol_total, li, rj, w, nL, nR, N, target_dtype, device, eps
    )

    if phi1 <= phi2:
        membership = torch.zeros(N, dtype=torch.long, device=device)
        membership[order1[: argmin1 + 1]] = 1
        best_phi = phi1
    else:
        membership = torch.zeros(N, dtype=torch.long, device=device)
        membership[order2[: argmin2 + 1]] = 1
        best_phi = phi2

    return SweepResult(
        phi_star=max(0.0, min(float(best_phi), 1.0)), cut_membership=membership
    )


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
) -> tuple[float, int, torch.Tensor]:
    """Vectorized sweep for a single sign orientation.

    Returns (phi, argmin_position, order).
    """
    x = torch.cat([u2, v2], dim=0)
    order = torch.argsort(x)
    rank = torch.empty(N, device=device, dtype=torch.long)
    rank[order] = torch.arange(N, device=device, dtype=torch.long)

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
    valid = den > eps
    if not valid.any():
        return 1.0, 0, order

    phi_all = torch.where(valid, cut / den, torch.tensor(float("inf"), device=device))
    argmin_pos = int(phi_all.argmin().item())
    phi = float(phi_all[argmin_pos].item())
    return phi, argmin_pos, order


# ---------------------------------------------------------------------------
# High-level entry points for CheegerDiagnostic
# ---------------------------------------------------------------------------


def compute_cheeger_features_materialized(M: torch.Tensor, rank: int = 2) -> CheegerFeatures:
    """Compute CheegerFeatures from a materialized M matrix."""
    U_mat, sigma, Vt = torch.linalg.svd(M, full_matrices=False)
    k = min(rank, len(sigma))
    sigma2 = sigma[1].item() if k > 1 else 0.0
    u2 = U_mat[:, 1] if k > 1 else torch.zeros(M.shape[0], device=M.device)
    v2 = Vt[1, :] if k > 1 else torch.zeros(M.shape[1], device=M.device)

    result = bipartite_sweep_conductance(u2, v2, M)
    gap = 1.0 - sigma2
    return CheegerFeatures(
        phi_star=result.phi_star,
        sigma2=sigma2,
        cheeger_lower=gap / 2.0,
        cheeger_upper=math.sqrt(max(2.0 * gap, 0.0)),
    )


def compute_cheeger_features_matrix_free(
    Q: torch.Tensor,
    K: torch.Tensor,
    d_k_inv_sqrt: torch.Tensor,
    scale: float,
    rank: int = 2,
    svd_method: str = "randomized",
    block_size: int = 256,
    causal: bool = False,
) -> CheegerFeatures:
    """Compute CheegerFeatures via matrix-free SVD + vectorized sweep."""
    L = Q.shape[0]
    device = Q.device
    k = min(max(rank, 2), L - 1)

    def matvec(v):
        return matvec_M_blocked(Q, K, v, d_k_inv_sqrt, scale, block_size, causal=causal)

    def matvec_t(u):
        return matvec_MT_blocked(Q, K, u, d_k_inv_sqrt, scale, block_size, causal=causal)

    if svd_method == "lanczos":
        U_svd, S, V_svd = svd_via_lanczos(
            matvec, matvec_t, L, k, max(2 * k + 2, 20), str(device)
        )
    else:
        U_svd, S, V_svd = randomized_svd(matvec, matvec_t, L, k, device=str(device))

    S_sorted, sort_idx = torch.sort(S, descending=True)
    sigma2 = S_sorted[1].item() if len(S_sorted) > 1 else 0.0
    idx2 = sort_idx[1] if len(sort_idx) > 1 else 0
    u2 = U_svd[:, idx2]
    v2 = V_svd[:, idx2]

    result = bipartite_sweep_conductance_matrix_free(
        u2, v2, Q, K, d_k_inv_sqrt, scale, block_size, causal=causal
    )
    gap = 1.0 - sigma2
    return CheegerFeatures(
        phi_star=result.phi_star,
        sigma2=sigma2,
        cheeger_lower=gap / 2.0,
        cheeger_upper=math.sqrt(max(2.0 * gap, 0.0)),
    )


def compute_cheeger_witness_materialized(M: torch.Tensor, rank: int = 2) -> torch.Tensor:
    """Compute cut membership witness from materialized M. Returns Tensor[nL] in {-1, +1}."""
    U_mat, sigma, Vt = torch.linalg.svd(M, full_matrices=False)
    k = min(rank, len(sigma))
    u2 = U_mat[:, 1] if k > 1 else torch.zeros(M.shape[0], device=M.device)
    v2 = Vt[1, :] if k > 1 else torch.zeros(M.shape[1], device=M.device)

    result = bipartite_sweep_conductance(u2, v2, M)
    nL = M.shape[0]
    query_membership = result.cut_membership[:nL]
    return query_membership * 2 - 1  # map {0,1} -> {-1, +1}


def compute_cheeger_witness_matrix_free(
    Q: torch.Tensor,
    K: torch.Tensor,
    d_k_inv_sqrt: torch.Tensor,
    scale: float,
    rank: int = 2,
    svd_method: str = "randomized",
    block_size: int = 256,
    causal: bool = False,
) -> torch.Tensor:
    """Compute cut membership witness via matrix-free path. Returns Tensor[L] in {-1, +1}."""
    L = Q.shape[0]
    device = Q.device
    k = min(max(rank, 2), L - 1)

    def matvec(v):
        return matvec_M_blocked(Q, K, v, d_k_inv_sqrt, scale, block_size, causal=causal)

    def matvec_t(u):
        return matvec_MT_blocked(Q, K, u, d_k_inv_sqrt, scale, block_size, causal=causal)

    if svd_method == "lanczos":
        U_svd, S, V_svd = svd_via_lanczos(
            matvec, matvec_t, L, k, max(2 * k + 2, 20), str(device)
        )
    else:
        U_svd, S, V_svd = randomized_svd(matvec, matvec_t, L, k, device=str(device))

    S_sorted, sort_idx = torch.sort(S, descending=True)
    idx2 = sort_idx[1] if len(sort_idx) > 1 else 0
    u2 = U_svd[:, idx2]
    v2 = V_svd[:, idx2]

    result = bipartite_sweep_conductance_matrix_free(
        u2, v2, Q, K, d_k_inv_sqrt, scale, block_size, causal=causal
    )
    query_membership = result.cut_membership[:L]
    return query_membership * 2 - 1  # map {0,1} -> {-1, +1}
