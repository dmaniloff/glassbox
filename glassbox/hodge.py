"""
Hodge decomposition features for the degree-normalized cross-operator M.

Two paths controlled by a sequence-length threshold:
  - Materialized (L <= threshold): dense tensor ops on the L×L matrix M.
  - Matrix-free  (L >  threshold): blocked-streaming matvecs, O(Ld) memory.

Features: asymmetry coefficient G, the Hodge gradient/curl split (Gamma, C) computed
exactly from the row-sum identity ||A_grad||^2 = 2||r||^2/L (r = M_asym @ 1), sigma2_asym,
commutator_norm, and curl_ratio. The Pythagorean split G^2 = Gamma^2 + C^2 is genuine.

References:
    Lim (2020): Hodge Laplacians on Graphs (SIAM Review)
    Jiang et al (2011): HodgeRank (Mathematical Programming)
"""

from __future__ import annotations

import math

import torch

from glassbox.cheeger import (
    bipartite_sweep_conductance,
    bipartite_sweep_conductance_matrix_free,
)
from glassbox.results import RoutingFeatures
from glassbox.svd import (
    compute_logsumexp_blocked,
    compute_M_fro_norm_blocked,
    get_M_entries_batch,
    matvec_commutator_blocked,
    matvec_M_blocked,
    matvec_Masym_blocked,
    matvec_MT_blocked,
    randomized_svd,
    svd_via_lanczos,
)

EPSILON = 1e-10

# ---------------------------------------------------------------------------
# Matrix-free G (asymmetry coefficient)
# ---------------------------------------------------------------------------


def compute_G_matrix_free(Q, K, d_k_inv_sqrt, scale, block_size=256, causal=False):
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
        attn = torch.softmax(scores, dim=-1)  # [bs, L]
        M_block = attn * d_k_inv_sqrt.unsqueeze(0)  # [bs, L]
        norm_sq = norm_sq + (M_block**2).sum()

        # Compute M[j, i] for all (i, j) pairs in this block
        row_idx = torch.arange(i0, i1, device=Q.device)
        col_idx = torch.arange(L, device=Q.device)
        ii_exp = col_idx.unsqueeze(1).expand(L, i1 - i0).reshape(-1)
        jj_exp = row_idx.unsqueeze(0).expand(L, i1 - i0).reshape(-1)
        M_T_entries = get_M_entries_batch(
            Q, K, lse, d_k_inv_sqrt, scale, ii_exp, jj_exp, causal=causal
        )
        M_T_block = M_T_entries.reshape(L, i1 - i0).T  # [bs, L]
        inner_sq = inner_sq + (M_block * M_T_block).sum()

    M_fro = torch.sqrt(norm_sq).item()
    asym_sq = (norm_sq - inner_sq) / 2.0
    asym_sq = asym_sq.clamp(min=0.0)
    G = (torch.sqrt(asym_sq) / (torch.sqrt(norm_sq) + EPSILON)).item()
    return G, M_fro


# ---------------------------------------------------------------------------
# Matrix-free sigma2_asym
# ---------------------------------------------------------------------------


def compute_sigma2_asym_matrix_free(
    Q, K, d_k_inv_sqrt, scale, block_size=256, svd_method="randomized", causal=False
):
    """Second singular value of M_asym = (M - M^T) / 2, computed matrix-free.

    The matvec for M_asym is: M_asym @ v = (M@v - M^T@v) / 2.
    Since M_asym is antisymmetric: M_asym^T = -M_asym.
    """
    L = Q.shape[0]
    device = Q.device

    def matvec_asym(v):
        return matvec_Masym_blocked(Q, K, v, d_k_inv_sqrt, scale, block_size, causal=causal)

    def matvec_asym_t(v):
        return -matvec_Masym_blocked(Q, K, v, d_k_inv_sqrt, scale, block_size, causal=causal)

    k = min(2, L - 1)
    if k < 2:
        return 0.0

    if svd_method == "lanczos":
        _, S, _ = svd_via_lanczos(
            matvec_asym, matvec_asym_t, L, k, max(2 * k + 2, 20), str(device), dtype=Q.dtype
        )
    else:
        _, S, _ = randomized_svd(
            matvec_asym, matvec_asym_t, L, k, device=str(device), dtype=Q.dtype
        )

    S_sorted, _ = torch.sort(S, descending=True)
    return S_sorted[1].item() if len(S_sorted) > 1 else 0.0


# ---------------------------------------------------------------------------
# Matrix-free commutator norm (Hutchinson trace estimator)
# ---------------------------------------------------------------------------


def estimate_commutator_norm_matrix_free(
    Q, K, d_k_inv_sqrt, scale, M_fro_norm, block_size=256, n_hutchinson=10, seed=42, causal=False
):
    """Estimate ||[M_sym, M_asym]||_F / ||M||_F via Hutchinson trace estimator.

    ||C||_F^2 = tr(C^T C) ~ (1/m) sum_i z_i^T C^T C z_i
    where z_i ~ Rademacher(+-1) and C = M_sym @ M_asym - M_asym @ M_sym.

    All m = n_hutchinson probes are stacked into one [L, m] right-hand side and pushed through
    a single multi-RHS matvec (one blocked softmax pass for all probes, not m serial passes).
    The squared-norm accumulates in float32 for precision under fp16/bf16 Q/K.
    """
    L = Q.shape[0]
    device = Q.device
    dtype = Q.dtype
    n_hutchinson = max(1, n_hutchinson)

    gen = torch.Generator(device=device).manual_seed(seed)
    # Z: [L, m] Rademacher (+-1); W = C @ Z is [L, m] in one pass.
    Z = torch.randint(0, 2, (L, n_hutchinson), device=device, generator=gen).to(dtype) * 2 - 1
    W = matvec_commutator_blocked(Q, K, Z, d_k_inv_sqrt, scale, block_size, causal=causal)
    trace_est = W.to(torch.float32).square().sum() / n_hutchinson
    comm_fro = torch.sqrt(trace_est.clamp(min=0.0))
    return (comm_fro / (M_fro_norm + EPSILON)).item()


# ---------------------------------------------------------------------------
# Main entry point: all routing features, always matrix-free
# ---------------------------------------------------------------------------


def compute_routing_features_matrix_free(
    Q,
    K,
    d_k_inv_sqrt,
    scale,
    rank,
    svd_method="randomized",
    block_size=256,
    seed=42,
    n_hutchinson=10,
    causal=False,
):
    """Compute all Hodge routing features matrix-free.

    Returns a RoutingFeatures with singular_values, spectral
    features, and Hodge decomposition features populated.

    The Pythagorean identity G^2 = Gamma^2 + C^2 is a genuine split: G is exact (blocked
    streaming), and Gamma/C are both derived from the exact row-sum r = M_asym @ 1 (one
    matvec) via the Hodge identity ||A_grad||^2 = 2||r||^2/L. ``seed`` / ``n_hutchinson``
    feed only the commutator-norm estimator.
    """
    L = Q.shape[0]
    device = Q.device

    # --- SVD of M for sigma2, phi_hat ---
    k = min(max(rank, 2), L - 1)

    def matvec(v):
        return matvec_M_blocked(Q, K, v, d_k_inv_sqrt, scale, block_size, causal=causal)

    def matvec_t(u):
        return matvec_MT_blocked(Q, K, u, d_k_inv_sqrt, scale, block_size, causal=causal)

    if svd_method == "lanczos":
        U_svd, S, V_svd = svd_via_lanczos(
            matvec, matvec_t, L, k, max(2 * k + 2, 20), str(device), dtype=Q.dtype
        )
    else:
        U_svd, S, V_svd = randomized_svd(matvec, matvec_t, L, k, device=str(device), dtype=Q.dtype)

    S_sorted, sort_idx = torch.sort(S, descending=True)
    sigma2 = S_sorted[1].item() if len(S_sorted) > 1 else 0.0

    # Cheeger conductance via bipartite sweep cut
    if len(S_sorted) > 1:
        idx2 = sort_idx[1]
        u2 = U_svd[:, idx2]
        v2 = V_svd[:, idx2]
        phi_hat = bipartite_sweep_conductance_matrix_free(
            u2,
            v2,
            Q,
            K,
            d_k_inv_sqrt,
            scale,
            block_size,
            causal=causal,
        )
    else:
        phi_hat = 0.0

    # --- G and ||M||_F in one fused blocked pass (compute_G already returns ||M||_F) ---
    G, M_fro_val = compute_G_matrix_free(Q, K, d_k_inv_sqrt, scale, block_size, causal=causal)

    # --- Exact Hodge gradient/curl split via the row-sum r = M_asym @ 1 (one matvec; #55) ---
    ones = torch.ones(L, device=device, dtype=Q.dtype)
    r = matvec_Masym_blocked(Q, K, ones, d_k_inv_sqrt, scale, block_size, causal=causal)
    M_asym_fro_sq = (G * M_fro_val) ** 2  # ||M_asym||^2 = (G * ||M||)^2
    Gamma, C = _gradient_curl_split(M_asym_fro_sq, r, L, M_fro_val)
    curl_ratio = C / (G + EPSILON)

    # --- sigma2_asym (matrix-free) ---
    sigma2_asym = compute_sigma2_asym_matrix_free(
        Q, K, d_k_inv_sqrt, scale, block_size, svd_method, causal=causal
    )

    # --- commutator_norm (Hutchinson, matrix-free) ---
    commutator_norm = estimate_commutator_norm_matrix_free(
        Q, K, d_k_inv_sqrt, scale, M_fro_val, block_size, n_hutchinson, seed, causal=causal
    )

    return RoutingFeatures(
        singular_values=S_sorted[:k].cpu().tolist(),
        phi_hat=phi_hat,
        sigma2=sigma2,
        G=G,
        Gamma=Gamma,
        C=C,
        curl_ratio=curl_ratio,
        sigma2_asym=sigma2_asym,
        commutator_norm=commutator_norm,
    )


# ---------------------------------------------------------------------------
# Materialized path (used when L <= threshold)
# ---------------------------------------------------------------------------


def compute_G_materialized(M):
    """Asymmetry coefficient from materialized M."""
    M_fro = torch.linalg.norm(M, "fro")
    M_asym = (M - M.T) / 2.0
    M_asym_fro = torch.linalg.norm(M_asym, "fro")
    G = (M_asym_fro / (M_fro + EPSILON)).item()
    return G, M_fro.item()


def _gradient_curl_split(asym_fro_sq: float, r: torch.Tensor, n: int, fro: float) -> tuple:
    """Exact Hodge gradient/curl split of an antisymmetric part from its row-sum r = A_asym @ 1.

    ``||A_grad||^2 = 2||r||^2 / n`` (potential phi = r/n); Gamma = ||A_grad||/||M|| (gradient,
    hierarchical) and the curl is the divergence-free residual
    ``C = sqrt(||A_asym||^2 - ||A_grad||^2)/||M||`` (circulatory). Both are derived from r, so
    ``G^2 = Gamma^2 + C^2`` is a genuine split — not the tautology of defining Gamma as
    sqrt(G^2 - C^2). C is the Pythagorean residual of the exact gradient energy (no triangle
    count enters); the gradient identity is the load-bearing part, shared with the asymmetry
    signal.
    """
    grad_energy = 2.0 * float((r * r).sum().item()) / n
    den = fro + EPSILON
    gamma = math.sqrt(max(grad_energy, 0.0)) / den
    c = math.sqrt(max(asym_fro_sq - grad_energy, 0.0)) / den
    return gamma, c


def compute_routing_features_materialized(M, rank, svd_method="randomized") -> RoutingFeatures:
    """All routing features from materialized M.

    Returns a RoutingFeatures with singular_values, spectral
    features, and Hodge decomposition features populated.

    Used when L <= threshold. Dense tensor ops are much faster than
    iterative matvec approaches at small sequence lengths.
    """
    # Dense SVD/svdvals has no fp16/bf16 kernel (LAPACK on CPU, cuSOLVER on GPU); upcast to
    # float32 for all materialized linalg. Emitted features are scalars/lists, so no cast-back
    # is needed. Same float32-compute pattern as GKL bidiag / hermitian_lanczos. (#57)
    if M.dtype in (torch.float16, torch.bfloat16):
        M = M.float()
    U_mat, sigma, Vt = torch.linalg.svd(M, full_matrices=False)
    k = min(rank, len(sigma))
    sigma2 = sigma[1].item() if len(sigma) > 1 else 0.0

    # Cheeger conductance via bipartite sweep cut
    if len(sigma) > 1:
        u2 = U_mat[:, 1]
        v2 = Vt[1, :]
        phi_hat = bipartite_sweep_conductance(u2, v2, M)
    else:
        phi_hat = 0.0

    G, M_fro = compute_G_materialized(M)

    M_sym = (M + M.T) / 2.0
    M_asym = (M - M.T) / 2.0
    sigma_asym = torch.linalg.svdvals(M_asym)
    sigma2_asym = sigma_asym[1].item() if len(sigma_asym) > 1 else 0.0

    comm = M_sym @ M_asym - M_asym @ M_sym
    commutator_norm = torch.linalg.norm(comm, "fro").item() / (M_fro + EPSILON)

    # Exact Hodge gradient/curl split via the row-sum identity (issue #55), replacing the
    # mis-normalized triangle-RMS curl. r = M_asym @ 1; Gamma/C both derived from it.
    # ||M_asym||^2 is taken as (G*||M||)^2 (consistent with G) so G^2 = Gamma^2 + C^2 is exact.
    r = M_asym.sum(dim=1)
    M_asym_fro_sq = (G * M_fro) ** 2
    Gamma, C = _gradient_curl_split(M_asym_fro_sq, r, M.shape[0], M_fro)
    curl_ratio = C / (G + EPSILON)

    return RoutingFeatures(
        singular_values=sigma[:k].cpu().tolist(),
        phi_hat=phi_hat,
        sigma2=sigma2,
        G=G,
        Gamma=Gamma,
        C=C,
        curl_ratio=curl_ratio,
        sigma2_asym=sigma2_asym,
        commutator_norm=commutator_norm,
    )


# ---------------------------------------------------------------------------
# Asymmetry coefficient G = ||A_asym||_F / ||A||_F via direct Hutchinson (Route B)
# ---------------------------------------------------------------------------


def _asymmetry_probe_accumulate(
    Q, K, d_k_inv_sqrt, scale, block_size, n_hutchinson, seed, causal, per_row=False
):
    """Hutchinson accumulation of A_asym @ z over Rademacher (+-1) probes.

    Returns (s_asym, row_sq): s_asym ~ ||A_asym||_F^2 = (1/m) sum_i ||A_asym z_i||^2,
    row_sq[i] ~ per-row mass with sum(row_sq) == s_asym (None if per_row=False).
    The operator A is selected by ``d_k_inv_sqrt``: the real key-degree vector gives the
    degree-normalized M; a unit vector gives the row-stochastic attention P.  Seeding
    makes the scalar and witness paths draw identical probes, so ||witness||_2 == G.
    """
    L = Q.shape[0]
    device = Q.device
    dtype = Q.dtype
    gen = torch.Generator(device=device).manual_seed(seed)
    scalar_acc = torch.tensor(0.0, device=device, dtype=dtype)
    row_acc = torch.zeros(L, device=device, dtype=dtype) if per_row else None
    for _ in range(n_hutchinson):
        z = torch.where(
            torch.rand(L, device=device, generator=gen) < 0.5,
            torch.ones(L, device=device, dtype=dtype),
            -torch.ones(L, device=device, dtype=dtype),
        )
        w = matvec_Masym_blocked(Q, K, z, d_k_inv_sqrt, scale, block_size, causal=causal)
        scalar_acc = scalar_acc + w.dot(w)
        if per_row:
            row_acc = row_acc + w * w
    s_asym = scalar_acc / n_hutchinson
    row_sq = (row_acc / n_hutchinson) if per_row else None
    return s_asym, row_sq


def asymmetry_partials_and_witness_matrix_free(
    Q,
    K,
    d_k_inv_sqrt,
    scale,
    M_fro_norm=None,
    block_size=256,
    n_hutchinson=32,
    seed=42,
    causal=False,
):
    """Additive sufficient statistics + per-row witness for the asymmetry coefficient.

    Returns (S_asym, S_den, row_sq, r): S_asym ~ ||A_asym||_F^2 (direct Hutchinson on
    ||A_asym z||^2, Route B — non-negative, cancellation-free), S_den = ||A||_F^2
    (exact), row_sq[i] ~ per-row asymmetry mass, and r = A_asym @ 1 (exact row-sum vector,
    one matvec) for the Hodge gradient/curl split — gradient energy = 2||r||^2/L.  The
    operator A is set by ``d_k_inv_sqrt`` (a unit vector => the row-stochastic attention P).
    The sum-of-squares partials are additive over a disjoint partition of windows:
    G = sqrt(sum S_asym / sum S_den).
    """
    if M_fro_norm is None:
        M_fro_norm = compute_M_fro_norm_blocked(
            Q, K, d_k_inv_sqrt, scale, block_size, causal=causal
        ).item()
    s_asym, row_sq = _asymmetry_probe_accumulate(
        Q, K, d_k_inv_sqrt, scale, block_size, n_hutchinson, seed, causal, per_row=True
    )
    S_asym = float(s_asym.clamp(min=0.0).item())
    S_den = float(M_fro_norm) ** 2
    # r = A_asym @ 1: exact gradient row-sum (one matvec), not a Hutchinson estimate.
    ones_vec = torch.ones(Q.shape[0], device=Q.device, dtype=Q.dtype)
    r = matvec_Masym_blocked(Q, K, ones_vec, d_k_inv_sqrt, scale, block_size, causal=causal)
    return S_asym, S_den, row_sq, r
