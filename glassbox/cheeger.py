"""Cheeger diagnostics for the degree-normalized cross-operator M.

Features: φ* (sweep-cut conductance), σ₂_asym (antisymmetric spectrum),
commutator_norm (symmetric/antisymmetric entanglement).

Two paths:
  - Materialized: dense tensor ops on L×L matrix M.
  - Matrix-free:  blocked-streaming matvecs, O(Ld) memory.

The spectral gap ``1 - σ₂`` only *bounds* conductance via the Cheeger inequality:

    (1 - σ₂) / 2  ≤  φ*  ≤  √(2(1 - σ₂))

References:
    Cheeger (1970): A lower bound for the smallest eigenvalue of the Laplacian
    Bauer & Jost (2013): Bipartite and neighborhood graphs and the spectrum
        of the normalized graph Laplacian
"""

from __future__ import annotations

from typing import Any

import torch

from glassbox.svd import (
    matvec_commutator_blocked,
    matvec_Masym_blocked,
    randomized_svd,
    svd_via_lanczos,
)

EPSILON = 1e-10


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


# ---------------------------------------------------------------------------
# Spectral diagnostics (moved from hodge.py)
# ---------------------------------------------------------------------------


def compute_sigma2_asym_matrix_free(
    Q, K, d_k_inv_sqrt, scale, block_size=256, svd_method="randomized",
    causal=False,
):
    """Second singular value of M_asym = (M - M^T) / 2, computed matrix-free."""
    L = Q.shape[0]
    device = Q.device

    def matvec_asym(v):
        return matvec_Masym_blocked(
            Q, K, v, d_k_inv_sqrt, scale, block_size, causal=causal
        )

    def matvec_asym_t(v):
        return -matvec_Masym_blocked(
            Q, K, v, d_k_inv_sqrt, scale, block_size, causal=causal
        )

    k = min(2, L - 1)
    if k < 2:
        return 0.0

    if svd_method == "lanczos":
        _, S, _ = svd_via_lanczos(
            matvec_asym, matvec_asym_t, L, k,
            max(2 * k + 2, 20), str(device),
        )
    else:
        _, S, _ = randomized_svd(
            matvec_asym, matvec_asym_t, L, k, device=str(device)
        )

    S_sorted, _ = torch.sort(S, descending=True)
    return S_sorted[1].item() if len(S_sorted) > 1 else 0.0


def estimate_commutator_norm_matrix_free(
    Q, K, d_k_inv_sqrt, scale, M_fro_norm, block_size=256,
    n_hutchinson=10, seed=42, causal=False,
):
    """Estimate ||[M_sym, M_asym]||_F / ||M||_F via Hutchinson trace."""
    L = Q.shape[0]
    device = Q.device
    dtype = Q.dtype

    gen = torch.Generator(device=device).manual_seed(seed)

    trace_est = torch.tensor(0.0, device=device, dtype=dtype)
    for _ in range(n_hutchinson):
        z = torch.where(
            torch.rand(L, device=device, generator=gen) < 0.5,
            torch.ones(L, device=device, dtype=dtype),
            -torch.ones(L, device=device, dtype=dtype),
        )
        w = matvec_commutator_blocked(
            Q, K, z, d_k_inv_sqrt, scale, block_size, causal=causal
        )
        trace_est = trace_est + w.dot(w)

    trace_est = trace_est / n_hutchinson
    comm_fro = torch.sqrt(trace_est.clamp(min=0.0))
    return (comm_fro / (M_fro_norm + EPSILON)).item()


# ---------------------------------------------------------------------------
# Registry callables — conform to routing.py dispatcher protocol
# ---------------------------------------------------------------------------


def compute_cheeger_materialized(
    shared_ctx: dict[str, Any], **kwargs,
) -> dict[str, float]:
    """Cheeger features from materialized M."""
    M = shared_ctx["M"]
    u2 = shared_ctx.get("u2")
    v2 = shared_ctx.get("v2")
    M_fro = shared_ctx["M_fro"]

    phi_hat = bipartite_sweep_conductance(u2, v2, M) if u2 is not None else 0.0

    M_asym = (M - M.T) / 2.0
    sigma_asym = torch.linalg.svdvals(M_asym)
    sigma2_asym = sigma_asym[1].item() if len(sigma_asym) > 1 else 0.0

    M_sym = (M + M.T) / 2.0
    comm = M_sym @ M_asym - M_asym @ M_sym
    commutator_norm = (
        torch.linalg.norm(comm, "fro").item() / (M_fro + EPSILON)
    )

    return {
        "phi_hat": phi_hat,
        "sigma2_asym": sigma2_asym,
        "commutator_norm": commutator_norm,
    }


def compute_cheeger_matrix_free(
    shared_ctx: dict[str, Any], **kwargs,
) -> dict[str, float]:
    """Cheeger features via matrix-free blocked streaming."""
    u2 = shared_ctx.get("u2")
    v2 = shared_ctx.get("v2")
    Q = shared_ctx["Q"]
    K = shared_ctx["K"]
    d_k_inv_sqrt = shared_ctx["d_k_inv_sqrt"]
    scale = shared_ctx["scale"]
    M_fro = shared_ctx["M_fro"]
    block_size = shared_ctx.get("block_size", 256)
    causal = shared_ctx.get("causal", False)
    svd_method = shared_ctx.get("svd_method", "randomized")
    n_hutchinson = kwargs.get("n_hutchinson", 10)
    seed = kwargs.get("seed", 42)

    if u2 is not None:
        phi_hat = bipartite_sweep_conductance_matrix_free(
            u2, v2, Q, K, d_k_inv_sqrt, scale, block_size, causal=causal,
        )
    else:
        phi_hat = 0.0

    sigma2_asym = compute_sigma2_asym_matrix_free(
        Q, K, d_k_inv_sqrt, scale, block_size, svd_method, causal=causal,
    )

    commutator_norm = estimate_commutator_norm_matrix_free(
        Q, K, d_k_inv_sqrt, scale, M_fro, block_size, n_hutchinson, seed,
        causal=causal,
    )

    return {
        "phi_hat": phi_hat,
        "sigma2_asym": sigma2_asym,
        "commutator_norm": commutator_norm,
    }


# ---------------------------------------------------------------------------
# Self-registration (lazy — triggered by routing._ensure_registered)
# ---------------------------------------------------------------------------

def _register() -> None:
    from glassbox.routing import register
    register("cheeger", compute_cheeger_materialized, compute_cheeger_matrix_free)

_register()
