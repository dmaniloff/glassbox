"""Hodge decomposition features for the degree-normalized cross-operator M.

Pure Hodge decomposition: asymmetry coefficient G, curl estimate C,
Pythagorean decomposition Gamma, and curl_ratio.

Two paths controlled by a sequence-length threshold:
  - Materialized (L <= threshold): dense tensor ops on the L×L matrix M.
  - Matrix-free  (L >  threshold): blocked-streaming matvecs, O(Ld) memory.

Triangle sampling for curl estimation uses Bernstein-bound adaptive sizing
and LRU-cached vectorized sampling (ported from shade.functional.hodge_ops).

References:
    Lim (2020): Hodge Laplacians on Graphs (SIAM Review)
    Jiang et al (2011): HodgeRank (Mathematical Programming)
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Any

import torch

from glassbox.cheeger import EPSILON
from glassbox.svd import (
    compute_logsumexp_blocked,
    get_M_entries_batch,
)

# ---------------------------------------------------------------------------
# Triangle sampling (ported from shade.functional.hodge_ops)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=64)
def sample_triangles(n: int, n_samples: int, seed: int = 42) -> torch.Tensor:
    """Generate triangle vertex indices for curl estimation (cached, CPU).

    Returns a (m, 3) int64 tensor of strictly-ordered triangle vertices
    (i < j < k) on CPU.  Cached by (n, n_samples, seed) so that repeated
    calls at the same sequence length reuse indices across heads.

    Ported from shade.functional.hodge_ops.sample_triangles.
    """
    n_tri = n * (n - 1) * (n - 2) // 6
    actual = min(n_samples, n_tri)
    if actual <= 0:
        return torch.zeros((0, 3), dtype=torch.int64)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    collected = []
    seen = set()
    remaining = actual
    while remaining > 0:
        batch_size = remaining * 3
        raw = torch.randint(0, n, (batch_size, 3), generator=gen)
        raw_sorted, _ = raw.sort(dim=1)
        valid = (raw_sorted[:, 0] < raw_sorted[:, 1]) & (
            raw_sorted[:, 1] < raw_sorted[:, 2]
        )
        for row in raw_sorted[valid]:
            key = (row[0].item(), row[1].item(), row[2].item())
            if key not in seen:
                seen.add(key)
                collected.append(row)
                remaining -= 1
                if remaining <= 0:
                    break
    return (
        torch.stack(collected)
        if collected
        else torch.zeros((0, 3), dtype=torch.int64)
    )


# ---------------------------------------------------------------------------
# Adaptive sample sizing (Bernstein bound, ported from shade)
# ---------------------------------------------------------------------------


def adaptive_curl_samples(
    n: int,
    Q: torch.Tensor | None = None,
    K: torch.Tensor | None = None,
    lse: torch.Tensor | None = None,
    d_k_inv_sqrt: torch.Tensor | None = None,
    scale: float | None = None,
    target_cv: float = 0.05,
    confidence: float = 0.95,
    pilot_size: int = 100,
    floor: int = 200,
    causal: bool = False,
) -> int:
    """Compute required triangle samples for target CV on curl RMS estimator.

    Uses the Bernstein bound (ported from shade):

        m >= (kappa_4 - 1) / (4 * eps^2) * 2 * ln(2 / delta)

    If Q/K/lse/d_k_inv_sqrt/scale are provided, kappa_4 is estimated from a
    pilot sample via matrix-free M[i,j] lookups.  Otherwise a conservative
    empirical formula kappa_4 = max(3, n/5) is used.
    """
    if n < 4:
        return 0
    n_tri = n * (n - 1) * (n - 2) // 6
    if n_tri <= floor:
        return n_tri

    delta = 1.0 - confidence
    log_factor = 2.0 * math.log(2.0 / delta)

    has_pilot = all(x is not None for x in (Q, K, lse, d_k_inv_sqrt, scale))
    if has_pilot:
        tri = sample_triangles(n, pilot_size, seed=0)
        if len(tri) < 10:
            return min(floor, n_tri)
        ii = tri[:, 0].to(Q.device)
        jj = tri[:, 1].to(Q.device)
        kk = tri[:, 2].to(Q.device)

        def _entry(a, b):
            return get_M_entries_batch(
                Q, K, lse, d_k_inv_sqrt, scale, a, b, causal=causal
            )

        circs = (
            (_entry(ii, jj) - _entry(jj, ii))
            + (_entry(jj, kk) - _entry(kk, jj))
            - (_entry(ii, kk) - _entry(kk, ii))
        )
        c2 = circs.square().to(torch.float64)
        mu2 = c2.mean()
        mu4 = c2.square().mean()
        kappa = (mu4 / (mu2.square() + 1e-30)).item()
        kappa = max(kappa, 1.0)
    else:
        kappa = max(3.0, n / 5.0)

    m = int(math.ceil((kappa - 1.0) / (4.0 * target_cv**2) * log_factor))
    return min(max(floor, m), n_tri)


# ---------------------------------------------------------------------------
# Matrix-free curl estimation
# ---------------------------------------------------------------------------


def estimate_curl_matrix_free(
    Q: torch.Tensor,
    K: torch.Tensor,
    lse: torch.Tensor,
    d_k_inv_sqrt: torch.Tensor,
    scale: float,
    M_fro_norm: float,
    target_cv: float = 0.05,
    confidence: float = 0.95,
    pilot_size: int = 100,
    min_samples: int = 200,
    seed: int = 42,
    causal: bool = False,
) -> float:
    """Triangle-sampling curl using on-the-fly M[i,j] lookups."""
    n = Q.shape[0]
    n_samp = adaptive_curl_samples(
        n,
        Q=Q,
        K=K,
        lse=lse,
        d_k_inv_sqrt=d_k_inv_sqrt,
        scale=scale,
        target_cv=target_cv,
        confidence=confidence,
        pilot_size=pilot_size,
        floor=min_samples,
        causal=causal,
    )
    if n_samp == 0:
        return 0.0

    tri = sample_triangles(n, n_samp, seed)
    if len(tri) == 0:
        return 0.0
    ii = tri[:, 0].to(Q.device)
    jj = tri[:, 1].to(Q.device)
    kk = tri[:, 2].to(Q.device)

    def _entry(a, b):
        return get_M_entries_batch(
            Q, K, lse, d_k_inv_sqrt, scale, a, b, causal=causal
        )

    circs = (
        (_entry(ii, jj) - _entry(jj, ii))
        + (_entry(jj, kk) - _entry(kk, jj))
        - (_entry(ii, kk) - _entry(kk, ii))
    )
    rms = circs.square().mean().sqrt()
    C = rms.item() / (math.sqrt(2) * (M_fro_norm + EPSILON))
    return C


# ---------------------------------------------------------------------------
# Matrix-free G (asymmetry coefficient)
# ---------------------------------------------------------------------------


def compute_G_matrix_free(
    Q, K, d_k_inv_sqrt, scale, block_size=256, causal=False,
):
    """Matrix-free G via blocked streaming.

    Computes ||M||_F^2 and <M, M^T>_F in one pass over row blocks.
    ||M_asym||_F^2 = (||M||_F^2 - <M, M^T>_F) / 2
    """
    L = Q.shape[0]
    lse = compute_logsumexp_blocked(Q, K, scale, block_size, causal=causal)
    norm_sq = torch.tensor(0.0, device=Q.device, dtype=Q.dtype)
    inner_sq = torch.tensor(0.0, device=Q.device, dtype=Q.dtype)

    for i0 in range(0, L, block_size):
        i1 = min(i0 + block_size, L)
        scores = Q[i0:i1] @ K.T * scale
        if causal:
            from glassbox.svd import _mask_causal

            scores = _mask_causal(scores, i0)
        attn = torch.softmax(scores, dim=-1)
        M_block = attn * d_k_inv_sqrt.unsqueeze(0)
        norm_sq = norm_sq + (M_block**2).sum()

        row_idx = torch.arange(i0, i1, device=Q.device)
        col_idx = torch.arange(L, device=Q.device)
        ii_exp = col_idx.unsqueeze(1).expand(L, i1 - i0).reshape(-1)
        jj_exp = row_idx.unsqueeze(0).expand(L, i1 - i0).reshape(-1)
        M_T_entries = get_M_entries_batch(
            Q, K, lse, d_k_inv_sqrt, scale, ii_exp, jj_exp, causal=causal
        )
        M_T_block = M_T_entries.reshape(L, i1 - i0).T
        inner_sq = inner_sq + (M_block * M_T_block).sum()

    M_fro = torch.sqrt(norm_sq).item()
    asym_sq = (norm_sq - inner_sq) / 2.0
    asym_sq = asym_sq.clamp(min=0.0)
    G = (torch.sqrt(asym_sq) / (torch.sqrt(norm_sq) + EPSILON)).item()
    return G, M_fro


# ---------------------------------------------------------------------------
# Materialized path
# ---------------------------------------------------------------------------


def estimate_curl_materialized(M, target_cv=0.05, seed=42):
    """Triangle-sampling curl on a materialized M tensor."""
    n = M.shape[0]
    if n < 4:
        return 0.0
    n_tri = n * (n - 1) * (n - 2) // 6
    n_samp = min(max(200, int(math.ceil(1.0 / (target_cv**2)))), n_tri)
    tri = sample_triangles(n, n_samp, seed)
    if len(tri) == 0:
        return 0.0
    ii = tri[:, 0]
    jj = tri[:, 1]
    kk = tri[:, 2]
    circs = (
        (M[ii, jj] - M[jj, ii])
        + (M[jj, kk] - M[kk, jj])
        - (M[ii, kk] - M[kk, ii])
    )
    rms = circs.square().mean().sqrt()
    M_fro_norm = torch.linalg.norm(M, "fro").item()
    C = rms.item() / (math.sqrt(2) * (M_fro_norm + EPSILON))
    return C


def compute_G_materialized(M):
    """Asymmetry coefficient from materialized M."""
    M_fro = torch.linalg.norm(M, "fro")
    M_asym = (M - M.T) / 2.0
    M_asym_fro = torch.linalg.norm(M_asym, "fro")
    G = (M_asym_fro / (M_fro + EPSILON)).item()
    return G, M_fro.item()


# ---------------------------------------------------------------------------
# Registry callables — conform to routing.py dispatcher protocol
# ---------------------------------------------------------------------------


def compute_hodge_materialized(
    shared_ctx: dict[str, Any], **kwargs,
) -> dict[str, float]:
    """Hodge features from materialized M."""
    M = shared_ctx["M"]
    target_cv = kwargs.get("target_cv", 0.05)
    seed = kwargs.get("seed", 42)

    G, _ = compute_G_materialized(M)
    C = estimate_curl_materialized(M, target_cv, seed)
    Gamma = math.sqrt(max(G**2 - C**2, 0.0))
    curl_ratio = C / (G + EPSILON)

    return {"G": G, "Gamma": Gamma, "C": C, "curl_ratio": curl_ratio}


def compute_hodge_matrix_free(
    shared_ctx: dict[str, Any], **kwargs,
) -> dict[str, float]:
    """Hodge features via matrix-free blocked streaming."""
    Q = shared_ctx["Q"]
    K = shared_ctx["K"]
    d_k_inv_sqrt = shared_ctx["d_k_inv_sqrt"]
    scale = shared_ctx["scale"]
    lse = shared_ctx["lse"]
    M_fro = shared_ctx["M_fro"]
    block_size = shared_ctx.get("block_size", 256)
    causal = shared_ctx.get("causal", False)
    target_cv = kwargs.get("target_cv", 0.05)
    confidence = kwargs.get("confidence", 0.95)
    pilot_size = kwargs.get("pilot_size", 100)
    min_samples = kwargs.get("min_samples", 200)
    seed = kwargs.get("seed", 42)

    G, _ = compute_G_matrix_free(
        Q, K, d_k_inv_sqrt, scale, block_size, causal=causal,
    )

    C = estimate_curl_matrix_free(
        Q, K, lse, d_k_inv_sqrt, scale, M_fro,
        target_cv=target_cv,
        confidence=confidence,
        pilot_size=pilot_size,
        min_samples=min_samples,
        seed=seed,
        causal=causal,
    )

    Gamma = math.sqrt(max(G**2 - C**2, 0.0))
    curl_ratio = C / (G + EPSILON)

    return {"G": G, "Gamma": Gamma, "C": C, "curl_ratio": curl_ratio}


# ---------------------------------------------------------------------------
# Self-registration (lazy — triggered by routing._ensure_registered)
# ---------------------------------------------------------------------------

def _register() -> None:
    from glassbox.routing import register
    register("hodge", compute_hodge_materialized, compute_hodge_matrix_free)

_register()
