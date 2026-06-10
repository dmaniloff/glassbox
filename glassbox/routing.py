"""Registry-based feature dispatcher for the degree-normalized cross-operator M.

Shared setup (SVD, degrees, Frobenius norm) runs once; registered feature
modules receive a shared context dict and return their feature dicts.

Two paths controlled by a sequence-length threshold:
  - Materialized (L <= threshold): ``compute_routing_features_materialized``
  - Matrix-free  (L >  threshold): ``compute_routing_features_matrix_free``

Feature modules register via ``register(signal_name, mat_fn, mf_fn)`` at
import time.  Adding a new module requires no changes here.
"""

from __future__ import annotations

from typing import Any, Callable

import torch

from glassbox.results import RoutingFeatures
from glassbox.svd import (
    compute_M_fro_norm_blocked,
    matvec_M_blocked,
    matvec_MT_blocked,
    randomized_svd,
    svd_via_lanczos,
)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, tuple[Callable[..., dict], Callable[..., dict]]] = {}


def register(
    signal_name: str,
    compute_materialized: Callable[..., dict[str, Any]],
    compute_matrix_free: Callable[..., dict[str, Any]],
) -> None:
    """Register a feature module by name.

    Args:
        signal_name: Unique signal name (e.g. "hodge", "cheeger").
        compute_materialized: ``(shared_ctx, **kw) -> dict[str, float]``
        compute_matrix_free:  ``(shared_ctx, **kw) -> dict[str, float]``
    """
    _REGISTRY[signal_name] = (compute_materialized, compute_matrix_free)


# ---------------------------------------------------------------------------
# Dispatcher — materialized path
# ---------------------------------------------------------------------------


def compute_routing_features_materialized(
    M: torch.Tensor,
    rank: int,
    signals: tuple[str, ...] = ("hodge", "cheeger"),
    svd_method: str = "randomized",
    target_cv: float = 0.05,
    seed: int = 42,
) -> RoutingFeatures:
    """Dispatch feature extraction on materialized M.

    Runs SVD once, builds shared context, then calls each registered
    module in *signals* and merges their feature dicts.
    """
    _ensure_registered()

    U_mat, sigma, Vt = torch.linalg.svd(M, full_matrices=False)
    k = min(rank, len(sigma))
    sigma2 = sigma[1].item() if len(sigma) > 1 else 0.0

    u2 = U_mat[:, 1] if len(sigma) > 1 else None
    v2 = Vt[1, :] if len(sigma) > 1 else None

    M_fro = torch.linalg.norm(M, "fro").item()

    shared_ctx: dict[str, Any] = {
        "M": M,
        "sigma2": sigma2,
        "u2": u2,
        "v2": v2,
        "M_fro": M_fro,
    }

    features: dict[str, Any] = {}
    for name in signals:
        if name not in _REGISTRY:
            continue
        mat_fn, _ = _REGISTRY[name]
        features.update(
            mat_fn(shared_ctx=shared_ctx, target_cv=target_cv, seed=seed)
        )

    return RoutingFeatures(
        singular_values=sigma[:k].cpu().tolist(),
        sigma2=sigma2,
        **features,
    )


# ---------------------------------------------------------------------------
# Dispatcher — matrix-free path
# ---------------------------------------------------------------------------


def compute_routing_features_matrix_free(
    Q: torch.Tensor,
    K: torch.Tensor,
    d_k_inv_sqrt: torch.Tensor,
    scale: float,
    lse: torch.Tensor,
    rank: int,
    signals: tuple[str, ...] = ("hodge", "cheeger"),
    svd_method: str = "randomized",
    block_size: int = 256,
    target_cv: float = 0.05,
    confidence: float = 0.95,
    pilot_size: int = 100,
    min_samples: int = 200,
    seed: int = 42,
    n_hutchinson: int = 10,
    causal: bool = False,
) -> RoutingFeatures:
    """Dispatch feature extraction via matrix-free blocked streaming.

    Runs SVD once (retaining U, V), computes M_fro once, then calls
    each registered module in *signals*.
    """
    _ensure_registered()

    L = Q.shape[0]
    device = Q.device

    k = min(max(rank, 2), L - 1)

    def matvec(v):
        return matvec_M_blocked(
            Q, K, v, d_k_inv_sqrt, scale, block_size, causal=causal
        )

    def matvec_t(u):
        return matvec_MT_blocked(
            Q, K, u, d_k_inv_sqrt, scale, block_size, causal=causal
        )

    if svd_method == "lanczos":
        U_svd, S, V_svd = svd_via_lanczos(
            matvec, matvec_t, L, k, max(2 * k + 2, 20), str(device)
        )
    else:
        U_svd, S, V_svd = randomized_svd(
            matvec, matvec_t, L, k, device=str(device)
        )

    S_sorted, sort_idx = torch.sort(S, descending=True)
    sigma2 = S_sorted[1].item() if len(S_sorted) > 1 else 0.0

    if len(S_sorted) > 1:
        idx2 = sort_idx[1]
        u2 = U_svd[:, idx2]
        v2 = V_svd[:, idx2]
    else:
        u2 = None
        v2 = None

    M_fro_norm = compute_M_fro_norm_blocked(
        Q, K, d_k_inv_sqrt, scale, block_size, causal=causal
    )
    M_fro_val = M_fro_norm.item()

    shared_ctx: dict[str, Any] = {
        "Q": Q,
        "K": K,
        "d_k_inv_sqrt": d_k_inv_sqrt,
        "scale": scale,
        "lse": lse,
        "sigma2": sigma2,
        "u2": u2,
        "v2": v2,
        "M_fro": M_fro_val,
        "block_size": block_size,
        "causal": causal,
        "svd_method": svd_method,
    }

    features: dict[str, Any] = {}
    for name in signals:
        if name not in _REGISTRY:
            continue
        _, mf_fn = _REGISTRY[name]
        features.update(
            mf_fn(
                shared_ctx=shared_ctx,
                target_cv=target_cv,
                confidence=confidence,
                pilot_size=pilot_size,
                min_samples=min_samples,
                seed=seed,
                n_hutchinson=n_hutchinson,
            )
        )

    return RoutingFeatures(
        singular_values=S_sorted[:k].cpu().tolist(),
        sigma2=sigma2,
        **features,
    )


# ---------------------------------------------------------------------------
# Lazy import trigger
# ---------------------------------------------------------------------------


_MODULES_IMPORTED = False


def _ensure_registered() -> None:
    """Import feature modules so they self-register.

    Called lazily on first dispatch to avoid import-time circular deps.
    """
    global _MODULES_IMPORTED
    if _MODULES_IMPORTED:
        return
    _MODULES_IMPORTED = True
    import glassbox.cheeger  # noqa: F401 — triggers register()
    import glassbox.hodge  # noqa: F401 — triggers register()
