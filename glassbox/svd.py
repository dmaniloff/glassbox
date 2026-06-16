"""
Matrix-free SVD algorithms and spectral feature computation.

A = softmax(QK^T / sqrt(d))
M = Dq_inv_sqrt * A * Dk_inv_sqrt
M is NOT symmetric in general. We compute SVD, not eigen-decomposition,
using matrix-vector products with M and M^T.

All operator functions accept both 1D vectors (L,) and 2D matrices (L, cols),
enabling the SVD algorithms to batch operator calls as BLAS-3 matrix multiplies.
"""

from __future__ import annotations

import torch

from glassbox.results import SpectralFeatures


def _mask_causal(scores: torch.Tensor, row_offset: int) -> torch.Tensor:
    """Mask scores[r, j] to -inf where j > row_offset + r (future tokens).

    Args:
        scores: Score matrix of shape (bs, L_k).
        row_offset: Global row index of the first row in the block.

    Returns:
        Masked scores with -inf above the causal diagonal.
    """
    bs, L_k = scores.shape
    row_idx = torch.arange(row_offset, row_offset + bs, device=scores.device).unsqueeze(1)
    col_idx = torch.arange(L_k, device=scores.device).unsqueeze(0)
    return scores.masked_fill(col_idx > row_idx, float("-inf"))


def matvec_S(Q, K, v):
    """S v = Q (K^T v). v: (L,) or (L, cols)."""
    return Q @ (K.T @ v)


def matvec_ST(Q, K, u):
    """S^T u = K (Q^T u). u: (L,) or (L, cols)."""
    return K @ (Q.T @ u)


def apply_A_blocked(Q, K, v, scale, block_size=256, causal=False):
    """A @ v via blocked row-streaming. v: (L_k,) or (L_k, cols)."""
    L_q = Q.shape[0]
    squeeze = v.ndim == 1
    if squeeze:
        v = v.unsqueeze(1)
    result = torch.zeros(L_q, v.shape[1], device=Q.device, dtype=Q.dtype)
    for i0 in range(0, L_q, block_size):
        i1 = min(i0 + block_size, L_q)
        scores = Q[i0:i1] @ K.T * scale
        if causal:
            scores = _mask_causal(scores, i0)
        attn = torch.softmax(scores, dim=-1)
        result[i0:i1] = attn @ v
    return result.squeeze(1) if squeeze else result


def apply_AT_blocked(Q, K, u, scale, block_size=256, causal=False):
    """A^T @ u via blocked row-streaming. u: (L_q,) or (L_q, cols)."""
    L_k = K.shape[0]
    squeeze = u.ndim == 1
    if squeeze:
        u = u.unsqueeze(1)
    result = torch.zeros(L_k, u.shape[1], device=K.device, dtype=K.dtype)
    for i0 in range(0, Q.shape[0], block_size):
        i1 = min(i0 + block_size, Q.shape[0])
        scores = Q[i0:i1] @ K.T * scale
        if causal:
            scores = _mask_causal(scores, i0)
        attn = torch.softmax(scores, dim=-1)
        result += attn.T @ u[i0:i1]
    return result.squeeze(1) if squeeze else result


def _apply_A_and_AT_blocked(Q, K, v_fwd, u_rev, scale, block_size=256, causal=False):
    """Fused A @ v_fwd and A^T @ u_rev sharing one softmax pass per block.

    Both inputs must be 2D: v_fwd (L_k, cols), u_rev (L_q, cols).
    Returns (A @ v_fwd, A^T @ u_rev), both 2D.
    """
    L_q = Q.shape[0]
    Av = torch.zeros(L_q, v_fwd.shape[1], device=Q.device, dtype=Q.dtype)
    ATu = torch.zeros(K.shape[0], u_rev.shape[1], device=K.device, dtype=K.dtype)
    for i0 in range(0, L_q, block_size):
        i1 = min(i0 + block_size, L_q)
        scores = Q[i0:i1] @ K.T * scale
        if causal:
            scores = _mask_causal(scores, i0)
        attn = torch.softmax(scores, dim=-1)
        Av[i0:i1] = attn @ v_fwd
        ATu += attn.T @ u_rev[i0:i1]
    return Av, ATu


def compute_dk_blocked(Q, K, scale, block_size=256, epsilon=1e-10, causal=False):
    """Compute D_K (column sums of A) via apply_AT_blocked.

    Uses Moore-Penrose pseudoinverse: zero-degree positions get 0 instead of
    large values.
    """
    ones = torch.ones(Q.shape[0], device=Q.device, dtype=Q.dtype)
    d_k = apply_AT_blocked(Q, K, ones, scale, block_size, causal=causal)
    d_k_inv_sqrt = torch.where(d_k > epsilon, 1.0 / torch.sqrt(d_k), torch.zeros_like(d_k))
    return d_k, d_k_inv_sqrt


def compute_logsumexp_blocked(Q, K, scale, block_size=256, causal=False):
    """Precompute lse[i] = logsumexp(Q_i . K^T * scale) for all rows."""
    L_q = Q.shape[0]
    lse = torch.zeros(L_q, device=Q.device, dtype=Q.dtype)
    for i0 in range(0, L_q, block_size):
        i1 = min(i0 + block_size, L_q)
        scores = Q[i0:i1] @ K.T * scale
        if causal:
            scores = _mask_causal(scores, i0)
        lse[i0:i1] = torch.logsumexp(scores, dim=-1)
    return lse


def get_M_entries_batch(Q, K, lse, d_k_inv_sqrt, scale, ii, jj, causal=False):
    """Compute M[ii, jj] on the fly. O(N*d) cost."""
    scores = (Q[ii] * K[jj]).sum(dim=-1) * scale  # [N]
    A_ij = torch.exp(scores - lse[ii])  # [N]
    result = A_ij * d_k_inv_sqrt[jj]  # [N]
    if causal:
        result = result * (jj <= ii).to(result.dtype)
    return result


def matvec_M_blocked(Q, K, x, d_k_inv_sqrt, scale, block_size=256, causal=False):
    """M @ x = A @ (D_K^{-1/2} * x). x: (L_k,) or (L_k, cols)."""
    dk = d_k_inv_sqrt.unsqueeze(1) if x.ndim > 1 else d_k_inv_sqrt
    return apply_A_blocked(Q, K, dk * x, scale, block_size, causal=causal)


def matvec_MT_blocked(Q, K, x, d_k_inv_sqrt, scale, block_size=256, causal=False):
    """M^T @ x = D_K^{-1/2} * (A^T @ x). x: (L_q,) or (L_q, cols)."""
    result = apply_AT_blocked(Q, K, x, scale, block_size, causal=causal)
    dk = d_k_inv_sqrt.unsqueeze(1) if result.ndim > 1 else d_k_inv_sqrt
    return dk * result


def compute_M_fro_norm_blocked(Q, K, d_k_inv_sqrt, scale, block_size=256, causal=False):
    """Compute ||M||_F without materializing M.

    Uses ||M||_F^2 = sum_j d_kj^2 * sum_i A_ij^2 to avoid a full (bs, L_k)
    temporary per block.
    """
    L_q = Q.shape[0]
    dk_sq = d_k_inv_sqrt * d_k_inv_sqrt
    norm_sq = torch.tensor(0.0, device=Q.device, dtype=Q.dtype)
    for i0 in range(0, L_q, block_size):
        i1 = min(i0 + block_size, L_q)
        scores = Q[i0:i1] @ K.T * scale
        if causal:
            scores = _mask_causal(scores, i0)
        attn = torch.softmax(scores, dim=-1)
        norm_sq = norm_sq + torch.dot(attn.pow(2).sum(dim=0), dk_sq)
    return torch.sqrt(norm_sq)


def compute_degree_normalized_M(A, epsilon=1e-10):
    """
    Compute degree-normalized cross-operator M from attention matrix A (materialized).

    SHADE paper (Section 3.2.2, Equation 1): M = D_Q^{-1/2} @ A @ D_K^{-1/2}.
    M reflects information routing independent of degree heterogeneity,
    making spectral properties comparable across heads and layers.

    Args:
        A: Attention matrix of shape (n_q, n_k)
        epsilon: Threshold below which degrees are treated as zero (default: 1e-10)

    Returns:
        M: Degree-normalized cross-operator of shape (n_q, n_k)
        d_q_inv_sqrt: Inverse sqrt of query degree vector of shape (n_q,)
        d_k_inv_sqrt: Inverse sqrt of key degree vector of shape (n_k,)
    """
    # Compute row sums (query degrees): d_Q_i = sum_j A_ij
    # shape: (n_q,); if A is softmax over rows, then D_Q = I
    d_q = A.sum(dim=1)

    # Compute column sums (key degrees): d_K_j = sum_i A_ij
    # shape: (n_k,)
    d_k = A.sum(dim=0)

    # Moore-Penrose pseudoinverse: zero out near-zero degrees
    d_q_inv_sqrt = torch.where(d_q > epsilon, 1.0 / torch.sqrt(d_q), torch.zeros_like(d_q))
    d_k_inv_sqrt = torch.where(d_k > epsilon, 1.0 / torch.sqrt(d_k), torch.zeros_like(d_k))

    M = (d_q_inv_sqrt[:, None] * A) * d_k_inv_sqrt[None, :]

    return M, d_q_inv_sqrt, d_k_inv_sqrt


def matvec_B_blocked(Q, K, x, d_k_inv_sqrt, scale, block_size=256, causal=False):
    """B @ x = M^T @ (M @ x). x: (L,) or (L, cols)."""
    y = matvec_M_blocked(Q, K, x, d_k_inv_sqrt, scale, block_size, causal=causal)
    return matvec_MT_blocked(Q, K, y, d_k_inv_sqrt, scale, block_size, causal=causal)


def matvec_Masym_blocked(Q, K, x, d_k_inv_sqrt, scale, block_size=256, causal=False):
    """(M - M^T)/2 @ x. Fused: one softmax pass instead of two."""
    squeeze = x.ndim == 1
    if squeeze:
        x = x.unsqueeze(1)
    dk = d_k_inv_sqrt.unsqueeze(1)
    Ax, ATx = _apply_A_and_AT_blocked(Q, K, dk * x, x, scale, block_size, causal)
    result = (Ax - dk * ATx) / 2.0
    return result.squeeze(1) if squeeze else result


def matvec_Msym_blocked(Q, K, x, d_k_inv_sqrt, scale, block_size=256, causal=False):
    """(M + M^T)/2 @ x. Fused: one softmax pass instead of two."""
    squeeze = x.ndim == 1
    if squeeze:
        x = x.unsqueeze(1)
    dk = d_k_inv_sqrt.unsqueeze(1)
    Ax, ATx = _apply_A_and_AT_blocked(Q, K, dk * x, x, scale, block_size, causal)
    result = (Ax + dk * ATx) / 2.0
    return result.squeeze(1) if squeeze else result


def matvec_commutator_blocked(Q, K, x, d_k_inv_sqrt, scale, block_size=256, causal=False):
    """[M_sym, M_asym] @ x = M_sym(M_asym(x)) - M_asym(M_sym(x)).

    3 fused softmax passes instead of 8 individual passes.
    """
    squeeze = x.ndim == 1
    if squeeze:
        x = x.unsqueeze(1)
    dk = d_k_inv_sqrt.unsqueeze(1)

    # Pass 1: M@x and M^T@x → derive asym(x) and sym(x)
    Ax, ATx = _apply_A_and_AT_blocked(Q, K, dk * x, x, scale, block_size, causal)
    MTx = dk * ATx
    asym_x = (Ax - MTx) / 2.0
    sym_x = (Ax + MTx) / 2.0

    # Pass 2: M_sym(asym_x) = (M @ asym_x + M^T @ asym_x) / 2
    A_asym, AT_asym = _apply_A_and_AT_blocked(Q, K, dk * asym_x, asym_x, scale, block_size, causal)
    sym_of_asym = (A_asym + dk * AT_asym) / 2.0

    # Pass 3: M_asym(sym_x) = (M @ sym_x - M^T @ sym_x) / 2
    A_sym, AT_sym = _apply_A_and_AT_blocked(Q, K, dk * sym_x, sym_x, scale, block_size, causal)
    asym_of_sym = (A_sym - dk * AT_sym) / 2.0

    result = sym_of_asym - asym_of_sym
    return result.squeeze(1) if squeeze else result


def randomized_svd(matvec, matvec_t, dim, k, p=5, q=2, device="cuda", dtype=None):
    """
    Matrix-free Randomized SVD via Halko, Martinsson, Tropp 2011.

    All operator calls are batched: matvec/matvec_t receive (dim, k+p) matrices
    and return (dim, k+p) results, enabling BLAS-3 throughput.
    """
    p = min(p, max(dim - k, 0))
    n = k + p

    Omega = torch.randn(dim, n, device=device, dtype=dtype)
    Y = matvec(Omega)
    native_dtype = Y.dtype

    # Power iterations with QR re-orthogonalization (Halko et al. §4.4)
    for _ in range(q):
        Z, _ = torch.linalg.qr(matvec_t(Y).float(), mode="reduced")
        Z = Z.to(native_dtype)
        Y, _ = torch.linalg.qr(matvec(Z).float(), mode="reduced")
        Y = Y.to(native_dtype)

    Q, _ = torch.linalg.qr(Y.float(), mode="reduced")
    Q_native = Q.to(native_dtype)

    B = matvec_t(Q_native).T.float()  # (n, dim)
    U_hat, S, Vt = torch.linalg.svd(B, full_matrices=False)

    U = (Q @ U_hat[:, :k]).to(native_dtype)
    V = Vt[:k, :].T.to(native_dtype)
    return U, S[:k], V


def lanczos(operator, dim, k, iters, device, dtype=None):
    """Lanczos iteration with pre-allocated basis and vectorized reorthogonalization.

    operator: v -> operator(v), expects 1D input of shape [dim].
    dim: dimension L.
    k: (unused, kept for interface compat) number of Lanczos vectors.
    iters: total Lanczos steps.
    """
    native_dtype = dtype or torch.float32

    V = torch.empty(dim, iters + 1, device=device, dtype=native_dtype)
    alphas = torch.empty(iters, device=device, dtype=torch.float32)
    betas_arr = torch.empty(iters, device=device, dtype=torch.float32)

    q = torch.randn(dim, device=device, dtype=native_dtype)
    V[:, 0] = q / torch.linalg.norm(q)

    beta = 0.0
    m = 0

    for j in range(iters):
        z = operator(V[:, j])
        alpha = V[:, j].dot(z)
        alphas[j] = alpha

        z = z - alpha * V[:, j]
        if j > 0:
            z = z - beta * V[:, j - 1]

        # Modified Gram-Schmidt reorthogonalization (sequential for fp16 stability)
        for i in range(j + 1):
            z = z - V[:, i].dot(z) * V[:, i]

        beta = torch.linalg.norm(z).item()
        if beta < 1e-8:
            m = j + 1
            break

        betas_arr[j] = beta
        V[:, j + 1] = z / beta
        m = j + 1

    # Build tridiagonal T via vectorized diagonal copy
    T = torch.zeros(m, m, device=device, dtype=torch.float32)
    T.diagonal().copy_(alphas[:m])
    if m > 1:
        T.diagonal(1).copy_(betas_arr[: m - 1])
        T.diagonal(-1).copy_(betas_arr[: m - 1])

    evals, evecs = torch.linalg.eigh(T)
    ritz_vectors = V[:, :m] @ evecs.to(V.dtype)

    return evals, ritz_vectors


def _principal_angles(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Principal angles between column spaces of A and B.

    Both inputs are (dim, k), assumed approximately orthonormal.
    Returns angles in radians, length k, sorted ascending.
    """
    # QR/svdvals require float32 (LAPACK does not support fp16/bf16).
    QA, _ = torch.linalg.qr(A.float(), mode="reduced")
    QB, _ = torch.linalg.qr(B.float(), mode="reduced")
    # Singular values of QA^T QB are cosines of principal angles.
    s = torch.linalg.svdvals(QA.T @ QB).clamp(-1.0, 1.0)
    return torch.acos(s)


def svd_via_lanczos(matvec, matvec_t, dim: int, k: int, iters: int, device: str, dtype=None):
    """Top-k singular triplets via Lanczos on M^T M.

    U recovery is batched: a single matvec(V) call replaces k individual calls.
    """
    evals, ritz = lanczos(
        operator=lambda v: matvec_t(matvec(v)),
        dim=dim,
        k=max(2 * k, k + 2),
        iters=iters,
        device=device,
        dtype=dtype,
    )
    # torch.linalg.eigh returns ascending; take largest-k.
    idx = torch.argsort(evals, descending=True)[:k]
    lam = evals[idx].clamp(min=0.0)
    S = torch.sqrt(lam)
    V = ritz[:, idx]

    # Batched U recovery: single matvec call for all k columns
    MV = matvec(V)
    inv_S = torch.where(S > 1e-12, 1.0 / S, torch.zeros_like(S))
    U = MV * inv_S.unsqueeze(0).to(MV.dtype)

    return U, S, V


def compare_svd_results(matvec, matvec_t, U1, S1, V1, U2, S2, V2, trials: int = 8):
    """Compare two (U,S,V) factorizations via batched operator calls."""
    device = S1.device
    k = min(S1.numel(), S2.numel())
    S1 = S1[:k]
    S2 = S2[:k]
    U1 = U1[:, :k]
    U2 = U2[:, :k]
    V1 = V1[:, :k]
    V2 = V2[:, :k]

    # singular values (order them descending before comparing)
    S1s, _ = torch.sort(S1, descending=True)
    S2s, _ = torch.sort(S2, descending=True)
    sv_abs = (S1s - S2s).abs()
    sv_rel = sv_abs / torch.clamp(torch.max(S1s.abs(), S2s.abs()), min=1e-12)

    # subspace alignment
    ang_U = _principal_angles(U1, U2)
    ang_V = _principal_angles(V1, V2)

    # Cast S to native dtype so mixed arithmetic with U/V doesn't error.
    native = V1.dtype
    S1n = S1s.to(native)
    S2n = S2s.to(native)

    # Batched residuals: 2 matvec calls instead of 2k
    MV2 = matvec(V2)
    MTU2 = matvec_t(U2)
    s_denom = torch.max(S1s, S2s).clamp(min=1e-12)
    mv_res = torch.linalg.norm(MV2.float() - S2s.unsqueeze(0) * U2.float(), dim=0) / s_denom
    mtu_res = torch.linalg.norm(MTU2.float() - S2s.unsqueeze(0) * V2.float(), dim=0) / s_denom

    # Batched reconstruction: 1 matvec call instead of trials
    X = torch.randn(V1.shape[0], trials, device=device, dtype=V1.dtype)
    Y = matvec(X)
    Y1 = U1 @ (S1n.unsqueeze(1) * (V1.T @ X))
    Y2 = U2 @ (S2n.unsqueeze(1) * (V2.T @ X))
    denom = torch.linalg.norm(Y.float(), dim=0).clamp(min=1e-12)
    recon = torch.stack(
        [
            torch.linalg.norm((Y - Y1).float(), dim=0) / denom,
            torch.linalg.norm((Y - Y2).float(), dim=0) / denom,
            torch.linalg.norm((Y1 - Y2).float(), dim=0) / denom,
        ],
        dim=1,
    )  # (trials, 3)

    return {
        "k": k,
        "sv_abs_max": sv_abs.max().item(),
        "sv_rel_max": sv_rel.max().item(),
        "ang_U_max_deg": (ang_U.max() * 180.0 / torch.pi).item(),
        "ang_V_max_deg": (ang_V.max() * 180.0 / torch.pi).item(),
        "mv_res_max": mv_res.max().item(),
        "mtu_res_max": mtu_res.max().item(),
        "recon_M_minus_USVt_mean": recon[:, 0].mean().item(),
        "recon_M_minus_USVt_max": recon[:, 0].max().item(),
        "recon_method_diff_mean": recon[:, 2].mean().item(),
        "recon_method_diff_max": recon[:, 2].max().item(),
    }


def compute_scores_matrix_features(
    Q: torch.Tensor,
    K: torch.Tensor,
    rank: int,
    method: str = "randomized",
) -> SpectralFeatures:
    """Compute spectral features of the scores matrix S = QK^T.

    Returns a SpectralFeatures with singular_values and derived
    spectral features (sv1, sv_ratio, sv_entropy) populated.
    """
    L = Q.shape[0]
    device = Q.device
    k = min(rank, L - 1)

    def mv(v):
        return matvec_S(Q, K, v)

    def mv_t(u):
        return matvec_ST(Q, K, u)

    if method == "lanczos":
        _, S, _ = svd_via_lanczos(mv, mv_t, L, k, max(2 * k + 2, 20), str(device), dtype=Q.dtype)
    else:
        _, S, _ = randomized_svd(mv, mv_t, L, k, device=str(device), dtype=Q.dtype)

    return SpectralFeatures(singular_values=S.cpu().tolist())
