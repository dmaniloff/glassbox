"""
Matrix-free SVD algorithms and spectral feature computation.

A = softmax(QK^T / sqrt(d))
M = Dq_inv_sqrt * A * Dk_inv_sqrt
M is NOT symmetric in general. We compute SVD, not eigen-decomposition,
using matrix-vector products with M and M^T.
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
    """Calculate Sv = Q K^T v in two O(Ld) passes, avoid computing S: [L, L]."""
    # v: [L], Q,K: [L, d]
    z = K.T @ v  # [d]
    return Q @ z  # [L]


def matvec_ST(Q, K, u):
    """Calculate S^T u = K Q^T u in two O(Ld) passes, avoid computing S^T: [L, L]."""
    # u: [L], Q,K: [L, d]
    z = Q.T @ u  # [d]
    return K @ z  # [L]


def apply_A_blocked(Q, K, v, scale, block_size=256, causal=False):
    """A @ v via blocked row-streaming. Peak memory: O(block_size * L_k).

    ``v`` may be a vector ``[L_k]`` or a matrix ``[L_k, m]`` of m right-hand
    sides; in the matrix case all m columns share one softmax pass per block
    (GEMM instead of m GEMVs), the basis for batched Hutchinson probes.
    """
    L_q = Q.shape[0]
    out_shape = (L_q,) if v.dim() == 1 else (L_q, v.shape[1])
    result = torch.zeros(out_shape, device=Q.device, dtype=Q.dtype)
    for i0 in range(0, L_q, block_size):
        i1 = min(i0 + block_size, L_q)
        scores = Q[i0:i1] @ K.T * scale  # [bs, L_k]
        if causal:
            scores = _mask_causal(scores, i0)
        attn = torch.softmax(scores, dim=-1)  # [bs, L_k]
        result[i0:i1] = attn @ v  # [bs] or [bs, m]
    return result


def apply_AT_blocked(Q, K, u, scale, block_size=256, causal=False):
    """A^T @ u via blocked row-streaming. ``u`` may be ``[L_q]`` or ``[L_q, m]``."""
    L_k = K.shape[0]
    out_shape = (L_k,) if u.dim() == 1 else (L_k, u.shape[1])
    result = torch.zeros(out_shape, device=K.device, dtype=K.dtype)
    for i0 in range(0, Q.shape[0], block_size):
        i1 = min(i0 + block_size, Q.shape[0])
        scores = Q[i0:i1] @ K.T * scale
        if causal:
            scores = _mask_causal(scores, i0)
        attn = torch.softmax(scores, dim=-1)  # [bs, L_k]
        result = result + attn.T @ u[i0:i1]  # [L_k] or [L_k, m]
    return result


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
        scores = Q[i0:i1] @ K.T * scale  # [bs, L_k]
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


def _scale_rows(d, x):
    """Broadcast a per-row vector ``d`` [L] against ``x`` of shape [L] or [L, m]."""
    return d.unsqueeze(-1) * x if x.dim() == 2 else d * x


def matvec_M_blocked(Q, K, x, d_k_inv_sqrt, scale, block_size=256, causal=False):
    """M @ x = A @ (D_K^{-1/2} * x). D_Q^{-1/2} = I for row-stochastic A.

    ``x`` may be ``[L]`` or ``[L, m]`` (multi-RHS).
    """
    return apply_A_blocked(Q, K, _scale_rows(d_k_inv_sqrt, x), scale, block_size, causal=causal)


def matvec_MT_blocked(Q, K, x, d_k_inv_sqrt, scale, block_size=256, causal=False):
    """M^T @ x = D_K^{-1/2} * (A^T @ x). ``x`` may be ``[L]`` or ``[L, m]``."""
    return _scale_rows(d_k_inv_sqrt, apply_AT_blocked(Q, K, x, scale, block_size, causal=causal))


def compute_M_fro_norm_blocked(Q, K, d_k_inv_sqrt, scale, block_size=256, causal=False):
    """Compute ||M||_F without materializing M.

    The sum-of-squares is accumulated in a float32 working dtype so fp16/bf16
    inputs do not lose the running total across blocks.
    """
    L_q = Q.shape[0]
    wdt = _working_dtype(Q.dtype)
    norm_sq = torch.tensor(0.0, device=Q.device, dtype=wdt)
    for i0 in range(0, L_q, block_size):
        i1 = min(i0 + block_size, L_q)
        scores = Q[i0:i1] @ K.T * scale
        if causal:
            scores = _mask_causal(scores, i0)
        attn = torch.softmax(scores, dim=-1)  # [bs, L_k]
        M_block = attn * d_k_inv_sqrt.unsqueeze(0)  # broadcast [bs, L_k]
        norm_sq = norm_sq + (M_block.to(wdt) ** 2).sum()
    return torch.sqrt(norm_sq).to(Q.dtype)


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
    """B @ x = M^T @ (M @ x)."""
    y = matvec_M_blocked(Q, K, x, d_k_inv_sqrt, scale, block_size, causal=causal)
    return matvec_MT_blocked(Q, K, y, d_k_inv_sqrt, scale, block_size, causal=causal)


def matvec_Masym_blocked(Q, K, x, d_k_inv_sqrt, scale, block_size=256, causal=False):
    """(M - M^T)/2 @ x = (M@x - M^T@x) / 2."""
    Mx = matvec_M_blocked(Q, K, x, d_k_inv_sqrt, scale, block_size, causal=causal)
    MTx = matvec_MT_blocked(Q, K, x, d_k_inv_sqrt, scale, block_size, causal=causal)
    return (Mx - MTx) / 2.0


def matvec_Msym_blocked(Q, K, x, d_k_inv_sqrt, scale, block_size=256, causal=False):
    """(M + M^T)/2 @ x = (M@x + M^T@x) / 2."""
    Mx = matvec_M_blocked(Q, K, x, d_k_inv_sqrt, scale, block_size, causal=causal)
    MTx = matvec_MT_blocked(Q, K, x, d_k_inv_sqrt, scale, block_size, causal=causal)
    return (Mx + MTx) / 2.0


def matvec_commutator_blocked(Q, K, x, d_k_inv_sqrt, scale, block_size=256, causal=False):
    """[M_sym, M_asym] @ x = M_sym(M_asym(x)) - M_asym(M_sym(x)).

    Cost: 8 matvecs per application (2 per sym/asym call, 4 calls total).
    """
    asym_x = matvec_Masym_blocked(Q, K, x, d_k_inv_sqrt, scale, block_size, causal=causal)
    term1 = matvec_Msym_blocked(Q, K, asym_x, d_k_inv_sqrt, scale, block_size, causal=causal)
    sym_x = matvec_Msym_blocked(Q, K, x, d_k_inv_sqrt, scale, block_size, causal=causal)
    term2 = matvec_Masym_blocked(Q, K, sym_x, d_k_inv_sqrt, scale, block_size, causal=causal)
    return term1 - term2


def randomized_svd(matvec, matvec_t, dim, k, p=5, q=2, device="cuda", seed=42):
    """
    Matrix-free Randomized SVD for a (dim x dim) linear operator given matvecs.

    Computes a rank-k approximation M ≈ U diag(S) V^T without ever forming
    the full L×L matrix. The key insight is that for S = QK^T, we can compute
    Sv = Q(K^T v) and S^T u = K(Q^T u) in O(Ld) each, avoiding the O(L^2)
    cost of materializing S. The caller wraps this into the matvec / matvec_t
    callables, making this routine agnostic to the operator's internal structure.

    Algorithm (Halko, Martinsson, Tropp 2011):
      1. Draw a random Gaussian test matrix Ω of shape (dim, k+p).
      2. Form Y = M Ω via k+p matvec calls.
      3. (Optional) Run q power iterations for better spectral separation.
      4. Compute an orthonormal basis Q for range(Y).
      5. Project: B = Q^T M  (computed via matvec_t on columns of Q).
      6. SVD of the small (k+p)×dim matrix B, then lift U back.

    Args:
        matvec:   v -> M v,   callable on vectors of length `dim`.
        matvec_t: u -> M^T u, callable on vectors of length `dim`.
        dim: Ambient dimension (L, the sequence length).
        k:   Number of singular triplets to return.
        p:   Oversampling parameter (default 5).
        q:   Number of power iterations (default 2).
        device: Torch device.
        seed: RNG seed for the test matrix Ω (default 42 => reproducible
              singular values).  Pass None to draw from the global RNG.

    Returns:
        U: (dim, k) left singular vectors.
        S: (k,)    singular values (descending).
        V: (dim, k) right singular vectors.

    The matvec / matvec_t callables are applied to the whole (dim, k+p) test
    matrix at once (multi-RHS), so range-finding and projection cost one
    blocked pass each instead of k+p separate matvecs.
    """
    # Clamp oversampling so k + p doesn't exceed dim
    p = min(p, max(dim - k, 0))

    # Step 1: random test matrix Ω (seeded for reproducibility)
    if seed is None:
        Omega = torch.randn(dim, k + p, device=device)
    else:
        gen = torch.Generator(device=device).manual_seed(seed)
        Omega = torch.randn(dim, k + p, device=device, generator=gen)

    # Step 2: sample Y = M Ω  (one multi-RHS pass)
    Y = matvec(Omega)  # (dim, k+p)

    # Optional: power iterations to improve spectral separation.
    for _ in range(q):
        Y = matvec(matvec_t(Y))  # M (M^T Y)

    # Step 3: orthonormal basis Q for range(Y)
    Q, _ = torch.linalg.qr(Y, mode="reduced")  # (dim, k+p)

    # Step 4: form small matrix B = Q^T M  (shape (k+p, dim)) via B^T = M^T Q
    Bt = matvec_t(Q)  # (dim, k+p)
    B = Bt.T  # (k+p, dim)

    # Step 5: SVD of small B
    U_hat, S, Vt = torch.linalg.svd(B, full_matrices=False)

    # Step 6: lift left singular vectors back to original space
    U = Q @ U_hat[:, :k]  # (dim, k)
    V = Vt[:k, :].T  # (dim, k)
    return U, S[:k], V


def lanczos(operator, dim, k, iters, device, seed=42):
    """
    operator: function v -> operator(v), expects v shape [dim]
    dim: dimension L
    k: number of Lanczos vectors kept (>= desired eigenvectors)
    iters: total Lanczos steps
    seed: RNG seed for the start vector (default 42 => reproducible Ritz values);
          pass None to draw from the global RNG.
    """
    Q = []
    alphas = []
    betas = []

    # start with random normalized vector (seeded for reproducibility)
    if seed is None:
        q = torch.randn(dim, device=device)
    else:
        gen = torch.Generator(device=device).manual_seed(seed)
        q = torch.randn(dim, device=device, generator=gen)
    q = q / torch.linalg.norm(q)
    Q.append(q)

    beta = torch.tensor(0.0, device=device)

    for _ in range(iters):
        z = operator(Q[-1])  # apply B = Mᵀ M
        alpha = torch.dot(Q[-1], z)
        z = z - alpha * Q[-1] - beta * (Q[-2] if len(Q) > 1 else 0)

        # reorthogonalize for numerical stability
        for q_prev in Q:
            z -= torch.dot(z, q_prev) * q_prev

        beta = torch.linalg.norm(z)
        if beta < 1e-8:
            break

        Q.append(z / beta)
        alphas.append(alpha)
        betas.append(beta)

    # Build small tridiagonal matrix T
    T = torch.zeros(len(Q), len(Q), device=device)
    for i in range(len(Q)):
        if i < len(alphas):
            T[i, i] = alphas[i]
        if i < len(betas):
            T[i, i + 1] = betas[i]
            T[i + 1, i] = betas[i]

    # eigen-decomposition of T
    evals, evecs = torch.linalg.eigh(T)

    # reconstruct Ritz vectors
    V = torch.stack(Q, dim=1)  # [dim, m]
    ritz_vectors = V @ evecs  # [dim, m]

    return evals, ritz_vectors


def _working_dtype(native_dtype):
    if native_dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return native_dtype


def _real_dtype(working_dtype):
    if working_dtype in (torch.complex128, torch.float64):
        return torch.float64
    return torch.float32


def hermitian_lanczos(
    matvec, dim, k, iters, device, which="largest", dtype=None, initial_vectors=None
):
    """Top/bottom-k eigenpairs of a Hermitian operator via Lanczos.

    Supports real and complex dtypes.  Uses conjugated inner products
    (torch.vdot) so it works correctly for complex-Hermitian H.

    Args:
        matvec: v -> H @ v, callable on 1-D tensors of length `dim`.
        dim:    Dimension of the operator.
        k:      Number of eigenvalues/eigenvectors to return.
        iters:  Number of Lanczos iterations.
        device: Torch device string.
        which:  ``"largest"`` (default) or ``"smallest"``.
        dtype:  Native dtype of the operator (None → float32).
        initial_vectors: Optional (dim, p) tensor to seed the Krylov
            subspace.  The first column (after QR) becomes the starting
            vector, biasing convergence toward that direction.

    Returns:
        eigenvalues:  (k,) real tensor (float32 or float64).
        eigenvectors: (dim, k) tensor in *dtype*.
    """
    if which not in ("smallest", "largest"):
        raise ValueError(f"which must be 'smallest' or 'largest', got '{which}'")

    native_dtype = dtype or torch.float32
    work_dt = _working_dtype(native_dtype)
    real_dt = _real_dtype(work_dt)

    if k <= 0 or dim <= 0:
        return torch.empty(0, device=device, dtype=real_dt), torch.empty(
            dim, 0, device=device, dtype=native_dtype
        )

    k = min(k, dim)
    iters = min(iters, dim)

    V = torch.empty(dim, iters + 1, device=device, dtype=work_dt)
    alphas = torch.empty(iters, device=device, dtype=real_dt)
    betas_arr = torch.empty(iters, device=device, dtype=real_dt)

    if initial_vectors is not None:
        init = initial_vectors.to(work_dt)
        if init.ndim == 1:
            init = init.unsqueeze(1)
        if init.shape[0] != dim:
            raise ValueError(f"initial_vectors dim {init.shape[0]} != operator dim {dim}")
        Q_init, _ = torch.linalg.qr(init, mode="reduced")
        V[:, 0] = Q_init[:, 0]
    else:
        q = torch.randn(dim, device=device, dtype=work_dt)
        V[:, 0] = q / torch.linalg.norm(q)

    beta = 0.0
    m = 0

    for j in range(iters):
        z = matvec(V[:, j].to(native_dtype)).to(work_dt)

        alpha = torch.vdot(V[:, j], z).real
        alphas[j] = alpha

        z = z - alpha * V[:, j]
        if j > 0:
            z = z - beta * V[:, j - 1]

        # CGS reorthogonalization
        coeffs = V[:, : j + 1].conj().T @ z
        z = z - V[:, : j + 1] @ coeffs

        beta = torch.linalg.norm(z).item()
        if beta < 1e-8:
            m = j + 1
            break

        betas_arr[j] = beta
        V[:, j + 1] = z / beta
        m = j + 1

    if m == 0:
        return torch.empty(0, device=device, dtype=real_dt), torch.empty(
            dim, 0, device=device, dtype=native_dtype
        )

    T = torch.zeros(m, m, device=device, dtype=real_dt)
    T.diagonal().copy_(alphas[:m])
    if m > 1:
        T.diagonal(1).copy_(betas_arr[: m - 1])
        T.diagonal(-1).copy_(betas_arr[: m - 1])

    evals, evecs = torch.linalg.eigh(T)

    k = min(k, m)
    if which == "smallest":
        sel_evals = evals[:k]
        sel_evecs = evecs[:, :k]
    else:
        sel_evals = evals[-k:].flip(0)
        sel_evecs = evecs[:, -k:].flip(1)

    ritz_vectors = (V[:, :m] @ sel_evecs.to(V.dtype)).to(native_dtype)
    return sel_evals, ritz_vectors


def bordered_rayleigh_ritz(
    matvec,
    dim_new,
    basis_prev,
    k,
    n_explore=2,
    device="cpu",
    which="largest",
    dtype=None,
):
    """Update eigenpairs via bordered Rayleigh-Ritz projection.

    Projects a Hermitian operator onto the subspace spanned by the
    (padded/truncated) prior Ritz basis plus random exploration vectors,
    then solves the small projected eigenproblem.

    Args:
        matvec: v -> H_new @ v, Hermitian operator on vectors of length dim_new.
        dim_new: Dimension of the new operator.
        basis_prev: Previous Ritz basis, shape (dim_prev, r).
        k: Number of eigenpairs to return.
        n_explore: Number of random exploration vectors to augment the basis.
        device: Torch device string.
        which: ``"largest"`` or ``"smallest"``.
        dtype: Native dtype (None -> float32).

    Returns:
        eigenvalues: (k,) real tensor.
        eigenvectors: (dim_new, k) tensor.
        all_projected_evals: all eigenvalues of the projected problem (for gap monitoring).
    """
    if which not in ("smallest", "largest"):
        raise ValueError(f"which must be 'smallest' or 'largest', got '{which}'")

    native_dtype = dtype or torch.float32
    work_dt = _working_dtype(native_dtype)
    real_dt = _real_dtype(work_dt)

    if basis_prev.ndim == 1:
        basis_prev = basis_prev.unsqueeze(1)

    dim_prev, r = basis_prev.shape
    basis_prev = basis_prev.to(device=device, dtype=work_dt)

    # Adapt basis to new dimension
    if dim_new > dim_prev:
        V_adapted = torch.zeros(dim_new, r, device=device, dtype=work_dt)
        V_adapted[:dim_prev] = basis_prev
    elif dim_new < dim_prev:
        V_adapted = basis_prev[:dim_new]
    else:
        V_adapted = basis_prev.clone()

    # Add exploration vectors
    if n_explore > 0:
        V_explore = torch.randn(dim_new, n_explore, device=device, dtype=work_dt)
        V_aug = torch.cat([V_adapted, V_explore], dim=1)
    else:
        V_aug = V_adapted

    # QR orthonormalize — drop rank-deficient columns
    V_orth, R = torch.linalg.qr(V_aug, mode="reduced")
    diag_abs = R.diagonal().abs()
    keep = diag_abs > 1e-10
    if not keep.all():
        V_orth = V_orth[:, keep]

    m = V_orth.shape[1]
    if m == 0:
        return (
            torch.empty(0, device=device, dtype=real_dt),
            torch.empty(dim_new, 0, device=device, dtype=native_dtype),
            torch.empty(0, device=device, dtype=real_dt),
        )

    k = min(k, m)

    # Project the operator: T = V_orth^H @ H @ V_orth
    W = torch.empty(dim_new, m, device=device, dtype=work_dt)
    for j in range(m):
        W[:, j] = matvec(V_orth[:, j].to(native_dtype)).to(work_dt)
    T = V_orth.conj().T @ W

    # Symmetrize (numerical cleanup)
    T = (T + T.conj().T) / 2
    T = T.to(real_dt) if not native_dtype.is_complex else T

    evals_proj, evecs_proj = torch.linalg.eigh(T)

    if which == "smallest":
        sel_evals = evals_proj[:k]
        sel_evecs = evecs_proj[:, :k]
    else:
        sel_evals = evals_proj[-k:].flip(0)
        sel_evecs = evecs_proj[:, -k:].flip(1)

    eigenvectors = (V_orth @ sel_evecs.to(V_orth.dtype)).to(native_dtype)
    return sel_evals, eigenvectors, evals_proj


def _principal_angles(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Principal angles between column spaces of A and B.

    Both inputs are (dim, k), assumed approximately orthonormal.
    Returns angles in radians, length k, sorted ascending.
    """
    # Orthonormalize for stability.
    QA, _ = torch.linalg.qr(A, mode="reduced")
    QB, _ = torch.linalg.qr(B, mode="reduced")
    # Singular values of QA^T QB are cosines of principal angles.
    s = torch.linalg.svdvals(QA.T @ QB).clamp(-1.0, 1.0)
    return torch.acos(s)


def svd_via_lanczos(matvec, matvec_t, dim: int, k: int, iters: int, device: str, seed=42):
    """
    Compute top-k singular triplets using Lanczos on B = M^T M.

    Like randomized_svd, this never forms the L×L matrix. For S = QK^T the
    crucial observation is that Sv = Q(K^T v) costs only O(Ld) — two
    thin matmuls through the L×d factors — so each Lanczos step is O(Ld)
    rather than O(L^2).

    Lanczos builds a Krylov subspace {v, Bv, B^2 v, ...} for the symmetric
    operator B = M^T M using the supplied matvec / matvec_t pair. After
    `iters` steps it eigen-decomposes the resulting small tridiagonal matrix
    to obtain Ritz values λ_i ≈ σ_i^2 and Ritz vectors (right singular
    vectors). Left singular vectors are recovered via u_i = M v_i / σ_i.

    Args:
        matvec:   v -> M v,   callable on vectors of length `dim`.
        matvec_t: u -> M^T u, callable on vectors of length `dim`.
        dim:   Ambient dimension (L, the sequence length).
        k:     Number of singular triplets to return.
        iters: Number of Lanczos iterations.
        device: Torch device.

    Returns:
        U: (dim, k) left singular vectors.
        S: (k,)    singular values (descending).
        V: (dim, k) right singular vectors.
    """
    evals, ritz = lanczos(
        operator=lambda v: matvec_t(matvec(v)),
        dim=dim,
        k=max(2 * k, k + 2),
        iters=iters,
        device=device,
        seed=seed,
    )
    # torch.linalg.eigh returns ascending; take largest-k.
    idx = torch.argsort(evals, descending=True)[:k]
    lam = evals[idx].clamp(min=0.0)
    S = torch.sqrt(lam)
    V = ritz[:, idx]
    # U_i = (1/sigma_i) M v_i
    U_cols = []
    for i in range(k):
        if S[i] < 1e-12:
            U_cols.append(torch.zeros(dim, device=device, dtype=V.dtype))
        else:
            U_cols.append(matvec(V[:, i]) / S[i])
    U = torch.stack(U_cols, dim=1)
    return U, S, V


def compare_svd_results(matvec, matvec_t, U1, S1, V1, U2, S2, V2, trials: int = 8):
    """
    Compare two (U,S,V) factorizations for the same operator M using:
    - singular value agreement
    - principal angles between left/right subspaces
    - residual norms ||M v - s u|| and ||M^T u - s v||
    - randomized reconstruction check on random vectors
    """
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

    # residuals
    mv_res_list: list[torch.Tensor] = []
    mtu_res_list: list[torch.Tensor] = []
    for i in range(k):
        s = torch.max(S1s[i], S2s[i]).clamp(min=1e-12)
        mv_res_list.append(torch.linalg.norm(matvec(V2[:, i]) - S2s[i] * U2[:, i]) / s)
        mtu_res_list.append(torch.linalg.norm(matvec_t(U2[:, i]) - S2s[i] * V2[:, i]) / s)
    mv_res: torch.Tensor = torch.stack(mv_res_list)
    mtu_res: torch.Tensor = torch.stack(mtu_res_list)

    # reconstruction check on random vectors: compare Mx to U diag(S) V^T x
    recon_list: list[torch.Tensor] = []
    for _ in range(trials):
        x = torch.randn(V1.shape[0], device=device)
        y = matvec(x)
        y1 = U1 @ (S1 * (V1.T @ x))
        y2 = U2 @ (S2 * (V2.T @ x))
        denom = torch.linalg.norm(y).clamp(min=1e-12)
        recon_list.append(
            torch.stack(
                [
                    torch.linalg.norm(y - y1) / denom,
                    torch.linalg.norm(y - y2) / denom,
                    torch.linalg.norm(y1 - y2) / denom,
                ]
            )
        )
    recon: torch.Tensor = torch.stack(recon_list, dim=0)  # (trials, 3)

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
        _, S, _ = svd_via_lanczos(mv, mv_t, L, k, max(2 * k + 2, 20), str(device))
    else:
        _, S, _ = randomized_svd(mv, mv_t, L, k, device=str(device))

    return SpectralFeatures(singular_values=S.cpu().tolist())
