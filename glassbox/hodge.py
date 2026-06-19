"""
Hodge decomposition features for the degree-normalized cross-operator M.

Two paths controlled by a sequence-length threshold:
  - Materialized (L <= threshold): dense tensor ops on the L×L matrix M.
  - Matrix-free  (L >  threshold): blocked-streaming matvecs, O(Ld) memory.

Features: asymmetry coefficient G, curl estimate C, Pythagorean decomposition
Gamma, sigma2_asym, commutator_norm, and curl_ratio.

Triangle sampling for curl estimation uses Bernstein-bound adaptive sizing
and LRU-cached vectorized sampling (ported from shade.functional.hodge_ops).

References:
    Lim (2020): Hodge Laplacians on Graphs (SIAM Review)
    Jiang et al (2011): HodgeRank (Mathematical Programming)
"""

from __future__ import annotations

import math
from functools import lru_cache

import torch

from glassbox.results import RoutingFeatures
from glassbox.svd import (
    _working_dtype,
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
    # Oversample to absorb rejections from degenerate and duplicate draws
    gen = torch.Generator(device="cpu").manual_seed(seed)
    collected = []
    seen = set()
    remaining = actual
    while remaining > 0:
        batch_size = remaining * 3  # oversample 3x
        raw = torch.randint(0, n, (batch_size, 3), generator=gen)
        raw_sorted, _ = raw.sort(dim=1)
        valid = (raw_sorted[:, 0] < raw_sorted[:, 1]) & (raw_sorted[:, 1] < raw_sorted[:, 2])
        for row in raw_sorted[valid]:
            key = (row[0].item(), row[1].item(), row[2].item())
            if key not in seen:
                seen.add(key)
                collected.append(row)
                remaining -= 1
                if remaining <= 0:
                    break
    return torch.stack(collected) if collected else torch.zeros((0, 3), dtype=torch.int64)


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
        return n_tri  # enumerate all triangles

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
            return get_M_entries_batch(Q, K, lse, d_k_inv_sqrt, scale, a, b, causal=causal)

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
    """Triangle-sampling curl using on-the-fly M[i,j] lookups.

    Uses Bernstein-bound adaptive sizing.  Triangle indices are cached via
    sample_triangles for reuse across heads at the same sequence length.
    """
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
        return get_M_entries_batch(Q, K, lse, d_k_inv_sqrt, scale, a, b, causal=causal)

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


def _asymmetry_probe_accumulate(
    Q, K, d_k_inv_sqrt, scale, block_size, n_hutchinson, seed, causal, per_row=False
):
    """Hutchinson accumulation of M_asym @ z over Rademacher (+-1) probes.

    Returns (s_asym, row_sq) where, with M_asym = (M - M^T) / 2:
        s_asym  ~ ||M_asym||_F^2 = (1/m) sum_i ||M_asym z_i||^2
        row_sq  ~ per-row mass; row_sq[i] = (1/m) sum_i (M_asym z_i)[i]^2,
                  so row_sq.sum() == s_asym (None if per_row=False)

    All n_hutchinson probes are drawn as one [L, m] Rademacher matrix and pushed
    through M_asym in a single multi-RHS blocked pass (one softmax recompute per
    block shared across probes -- a GEMM, not m GEMVs), then reduced.  Squares
    are accumulated in a float32 working dtype for fp16/bf16 safety.  Seeding the
    generator makes the scalar (per_row=False) and witness (per_row=True) paths
    draw the identical probe matrix, so ||witness||_2 == G exactly.
    """
    L = Q.shape[0]
    device = Q.device
    dtype = Q.dtype
    wdt = _working_dtype(dtype)

    gen = torch.Generator(device=device).manual_seed(seed)
    Z = torch.where(
        torch.rand(L, n_hutchinson, device=device, generator=gen) < 0.5,
        torch.ones(L, n_hutchinson, device=device, dtype=dtype),
        -torch.ones(L, n_hutchinson, device=device, dtype=dtype),
    )
    W = matvec_Masym_blocked(Q, K, Z, d_k_inv_sqrt, scale, block_size, causal=causal)  # [L, m]
    Wsq = W.to(wdt) ** 2
    s_asym = Wsq.sum() / n_hutchinson
    row_sq = (Wsq.sum(dim=1) / n_hutchinson) if per_row else None
    return s_asym, row_sq


def asymmetry_partials_matrix_free(
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
    """Additive sufficient statistics for G over one window.

    Returns (S_asym, S_M) where S_asym ~ ||M_asym||_F^2 (Hutchinson, Route B)
    and S_M = ||M||_F^2 (exact).  Both are sums-of-squares, hence additive over
    a disjoint partition of windows: a streaming reader maintains running totals
    and reports G_global = sqrt(sum S_asym / sum S_M) at readout.  Floats.
    """
    if M_fro_norm is None:
        M_fro_norm = compute_M_fro_norm_blocked(
            Q, K, d_k_inv_sqrt, scale, block_size, causal=causal
        ).item()
    s_asym, _ = _asymmetry_probe_accumulate(
        Q, K, d_k_inv_sqrt, scale, block_size, n_hutchinson, seed, causal, per_row=False
    )
    S_asym = float(s_asym.clamp(min=0.0).item())
    S_M = float(M_fro_norm) ** 2
    return S_asym, S_M


def compute_G_matrix_free(
    Q,
    K,
    d_k_inv_sqrt,
    scale,
    block_size=256,
    n_hutchinson=32,
    seed=42,
    M_fro_norm=None,
    causal=False,
):
    """Asymmetry coefficient G = ||M_asym||_F / ||M||_F, matrix-free (stochastic).

    Estimates the numerator ||M_asym||_F^2 DIRECTLY via a Hutchinson estimator
    (Route B), avoiding the catastrophic cancellation of the tr(M^2) route:

        ||M_asym||_F^2 = tr(M_asym^T M_asym) ~ (1/m) sum_i || M_asym z_i ||^2,
        z_i ~ Rademacher(+-1),  M_asym z = (M z - M^T z) / 2  (2 matvecs/probe).

    Cost O(n_hutchinson * L * d); no transpose block is materialized.  G is
    non-negative by construction and downward-biased by Jensen (sqrt of an
    unbiased squared-norm estimate) -- the same convention used by
    estimate_commutator_norm_matrix_free.  Returns (G, M_fro).
    """
    if M_fro_norm is None:
        M_fro_norm = compute_M_fro_norm_blocked(
            Q, K, d_k_inv_sqrt, scale, block_size, causal=causal
        ).item()
    S_asym, _ = asymmetry_partials_matrix_free(
        Q, K, d_k_inv_sqrt, scale, M_fro_norm, block_size, n_hutchinson, seed, causal=causal
    )
    G = math.sqrt(S_asym) / (M_fro_norm + EPSILON)
    return G, M_fro_norm


def compute_asymmetry_witness_materialized(M):
    """Per-row asymmetry profile from materialized M.

    witness[i] = ||M_asym[i, :]||_2 / ||M||_F, with M_asym = (M - M^T) / 2.
    Then ||witness||_2 == G (the materialized asymmetry coefficient), so the
    witness is a per-token decomposition of the scalar.  Returns Tensor[L].
    """
    M_fro = torch.linalg.norm(M, "fro")
    M_asym = (M - M.T) / 2.0
    row_norms = torch.linalg.norm(M_asym, dim=1)  # [L]
    return row_norms / (M_fro + EPSILON)


def compute_asymmetry_witness_matrix_free(
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
    """Per-row asymmetry profile, matrix-free (Hutchinson, Route B).

    Reuses the same Rademacher probes as compute_G_matrix_free (identical seed
    and n_hutchinson), estimating r_i^2 ~ (1/m) sum (M_asym z)_i^2.  Emits
    witness[i] = r_i / ||M||_F, so ||witness||_2 == G computed from the same
    probes (up to float).  Returns Tensor[L].
    """
    if M_fro_norm is None:
        M_fro_norm = compute_M_fro_norm_blocked(
            Q, K, d_k_inv_sqrt, scale, block_size, causal=causal
        ).item()
    _, row_sq = _asymmetry_probe_accumulate(
        Q, K, d_k_inv_sqrt, scale, block_size, n_hutchinson, seed, causal, per_row=True
    )
    row_norms = torch.sqrt(row_sq.clamp(min=0.0))
    return row_norms / (M_fro_norm + EPSILON)


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
    """One probe pass returning (S_asym, S_M, row_sq) for a fused scalar+witness.

    S_asym ~ ||M_asym||_F^2, S_M = ||M||_F^2, and row_sq[i] ~ per-row asymmetry
    mass with sum(row_sq) == S_asym.  Lets a caller derive both the scalar G
    (from S_asym, S_M) and the per-row witness (from row_sq, S_M) from a single
    multi-RHS Hutchinson pass instead of two.
    """
    if M_fro_norm is None:
        M_fro_norm = compute_M_fro_norm_blocked(
            Q, K, d_k_inv_sqrt, scale, block_size, causal=causal
        ).item()
    s_asym, row_sq = _asymmetry_probe_accumulate(
        Q, K, d_k_inv_sqrt, scale, block_size, n_hutchinson, seed, causal, per_row=True
    )
    S_asym = float(s_asym.clamp(min=0.0).item())
    S_M = float(M_fro_norm) ** 2
    return S_asym, S_M, row_sq


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
        _, S, _ = svd_via_lanczos(matvec_asym, matvec_asym_t, L, k, max(2 * k + 2, 20), str(device))
    else:
        _, S, _ = randomized_svd(matvec_asym, matvec_asym_t, L, k, device=str(device))

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
    Each C @ z costs 8 matvecs.  All m probes are applied as one [L, m]
    multi-RHS pass; squares accumulate in a float32 working dtype.
    """
    L = Q.shape[0]
    device = Q.device
    dtype = Q.dtype
    wdt = _working_dtype(dtype)

    gen = torch.Generator(device=device).manual_seed(seed)
    Z = torch.where(
        torch.rand(L, n_hutchinson, device=device, generator=gen) < 0.5,
        torch.ones(L, n_hutchinson, device=device, dtype=dtype),
        -torch.ones(L, n_hutchinson, device=device, dtype=dtype),
    )
    W = matvec_commutator_blocked(Q, K, Z, d_k_inv_sqrt, scale, block_size, causal=causal)  # [L, m]
    trace_est = (W.to(wdt) ** 2).sum() / n_hutchinson
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
    lse,
    rank,
    svd_method="randomized",
    block_size=256,
    target_cv=0.05,
    confidence=0.95,
    pilot_size=100,
    min_samples=200,
    seed=42,
    n_hutchinson=32,
    causal=False,
):
    """Compute all Hodge routing features matrix-free.

    Returns a RoutingFeatures with singular_values, spectral
    features, and Hodge decomposition features populated.

    The Pythagorean identity G^2 = Gamma^2 + C^2 holds approximately:
    G is a Hutchinson estimate (Route B), C is sampled (Bernstein-bound
    adaptive), and Gamma = sqrt(max(G^2 - C^2, 0)) is derived from the identity.
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
        _, S, _ = svd_via_lanczos(matvec, matvec_t, L, k, max(2 * k + 2, 20), str(device))
    else:
        _, S, _ = randomized_svd(matvec, matvec_t, L, k, device=str(device))

    S_sorted, _ = torch.sort(S, descending=True)
    sigma2 = S_sorted[1].item() if len(S_sorted) > 1 else 0.0
    phi_hat = 1.0 - sigma2

    # --- ||M||_F (exact, blocked) ---
    M_fro_norm = compute_M_fro_norm_blocked(Q, K, d_k_inv_sqrt, scale, block_size, causal=causal)
    M_fro_val = M_fro_norm.item()

    # --- G (Hutchinson, matrix-free; reuses exact ||M||_F) ---
    G, _ = compute_G_matrix_free(
        Q,
        K,
        d_k_inv_sqrt,
        scale,
        block_size=block_size,
        n_hutchinson=n_hutchinson,
        seed=seed,
        M_fro_norm=M_fro_val,
        causal=causal,
    )

    # --- C (sampled, Bernstein-bound adaptive) ---
    C = estimate_curl_matrix_free(
        Q,
        K,
        lse,
        d_k_inv_sqrt,
        scale,
        M_fro_val,
        target_cv=target_cv,
        confidence=confidence,
        pilot_size=pilot_size,
        min_samples=min_samples,
        seed=seed,
        causal=causal,
    )

    # --- Pythagorean: Gamma = sqrt(G^2 - C^2) ---
    Gamma = math.sqrt(max(G**2 - C**2, 0.0))
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
    circs = (M[ii, jj] - M[jj, ii]) + (M[jj, kk] - M[kk, jj]) - (M[ii, kk] - M[kk, ii])
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


def compute_routing_features_materialized(
    M, rank, svd_method="randomized", target_cv=0.05, seed=42
) -> RoutingFeatures:
    """All routing features from materialized M.

    Returns a RoutingFeatures with singular_values, spectral
    features, and Hodge decomposition features populated.

    Used when L <= threshold. Dense tensor ops are much faster than
    iterative matvec approaches at small sequence lengths.
    """
    sigma = torch.linalg.svdvals(M)
    k = min(rank, len(sigma))
    sigma2 = sigma[1].item() if len(sigma) > 1 else 0.0
    phi_hat = 1.0 - sigma2

    G, M_fro = compute_G_materialized(M)

    M_sym = (M + M.T) / 2.0
    M_asym = (M - M.T) / 2.0
    sigma_asym = torch.linalg.svdvals(M_asym)
    sigma2_asym = sigma_asym[1].item() if len(sigma_asym) > 1 else 0.0

    comm = M_sym @ M_asym - M_asym @ M_sym
    commutator_norm = torch.linalg.norm(comm, "fro").item() / (M_fro + EPSILON)

    C = estimate_curl_materialized(M, target_cv, seed)
    Gamma = math.sqrt(max(G**2 - C**2, 0.0))
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
