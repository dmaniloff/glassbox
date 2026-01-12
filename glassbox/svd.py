"""
Explorations of matrix-free SVD algorithms.

A = softmax(QK^T / sqrt(d))
M = Dq_inv_sqrt * A * Dk_inv_sqrt
M is NOT symmetric in general. We compute SVD, not eigen-decomposition, using matrix–vector products with M and M^T
"""

import torch
from rich import print as rprint


def matvec_S(Q, K, v):
    """Calculate Sv = Q K^T v in two passes."""
    # v: [L], Q,K: [L, d]
    # compute K^T v
    z = K.T @ v  # [d]
    # compute Q z
    return Q @ z  # [L]


def matvec_ST(Q, K, u):
    """Calculate S^T u = K Q^T u in two passes."""
    z = Q.T @ u  # [d]
    return K @ z  # [L]


def matvec_A(x):
    """This is attention @ x."""
    pass


def matvec_AT(x):
    """This is attention^T @ x."""
    pass


def compute_degree_normalized_M(A, epsilon=1e-8):
    """
    Compute the degree-normalized cross-operator M from attention matrix A.

    Following SHADE paper Section 3.2.2, Equation 1:
    M = D_Q^{-1/2} @ A @ D_K^{-1/2}

    Args:
        A: Attention matrix of shape (n_q, n_k)
        epsilon: Small value for numerical stability (default: 1e-8)

    Returns:
        M: Degree-normalized cross-operator of shape (n_q, n_k)
        d_q_inv_sqrt: Inverse sqrt of query degree vector d_q^{-1/2} of shape (n_q,)
        d_k_inv_sqrt: Inverse sqrt of key degree vector d_k^{-1/2} of shape (n_k,)
    """
    # Compute row sums (query degrees): d_Q_i = sum_j A_ij
    d_q = A.sum(dim=1)  # shape: (n_q,); if A is softmax over rows, then D_Q = I

    # Compute column sums (key degrees): d_K_j = sum_i A_ij
    d_k = A.sum(
        dim=0
    )  # shape: (n_k,); in a real implementation, get the column degrees with a call to matvec_AT on an all‑ones vector.

    # Compute inverse sqrt degree *vectors* (we only need elementwise scaling for matvecs).
    d_q_inv_sqrt = 1.0 / torch.sqrt(d_q + epsilon)  # (n_q,)
    d_k_inv_sqrt = 1.0 / torch.sqrt(d_k + epsilon)  # (n_k,)

    # Explicit M (optional / for debugging)
    M = (d_q_inv_sqrt[:, None] * A) * d_k_inv_sqrt[None, :]

    return M, d_q_inv_sqrt, d_k_inv_sqrt


def matvec_M(x, d_q_inv_sqrt, d_k_inv_sqrt):
    # diag-scale input (d*_inv_sqrt are vectors)
    x1 = d_k_inv_sqrt * x  # [L]
    y1 = matvec_A(x1)  # attention @ x1   (FlashAttn-like run with dv=1)
    y = d_q_inv_sqrt * y1  # [L]
    return y


def matvec_MT(x, d_q_inv_sqrt, d_k_inv_sqrt):
    x1 = d_q_inv_sqrt * x
    y1 = matvec_AT(x1)
    y = d_k_inv_sqrt * y1
    return y


def matvec_B(x, d_q_inv_sqrt, d_k_inv_sqrt):
    y = matvec_M(x, d_q_inv_sqrt, d_k_inv_sqrt)
    return matvec_MT(y, d_q_inv_sqrt, d_k_inv_sqrt)


def randomized_svd(matvec, matvec_t, dim, k, p=5, q=2, device="cuda"):
    """
    Standard randomized SVD for a (dim x dim) linear operator M given matvecs.

    Computes a rank-k approximation M ≈ U diag(S) V^T.
    """
    # Step 1: random test matrix Ω
    Omega = torch.randn(dim, k + p, device=device)

    # Step 2: sample Y = M Ω
    Y = torch.stack([matvec(Omega[:, i]) for i in range(k + p)], dim=1)  # (dim, k+p)

    # Optional: power iterations to improve spectral separation.
    for _ in range(q):
        Z = torch.stack([matvec_t(Y[:, i]) for i in range(Y.shape[1])], dim=1)  # M^T Y
        Y = torch.stack(
            [matvec(Z[:, i]) for i in range(Z.shape[1])], dim=1
        )  # M (M^T Y)

    # Step 3: orthonormal basis Q for range(Y)
    Q, _ = torch.linalg.qr(Y, mode="reduced")  # (dim, k+p)

    # Step 4: form small matrix B = Q^T M  (shape (k+p, dim))
    # We can compute B via B^T = M^T Q, using matvec_t.
    Bt = torch.stack(
        [matvec_t(Q[:, i]) for i in range(Q.shape[1])], dim=1
    )  # (dim, k+p)
    B = Bt.T  # (k+p, dim)

    # Step 5: SVD of small B
    U_hat, S, Vt = torch.linalg.svd(B, full_matrices=False)

    # Step 6: lift left singular vectors back to original space
    U = Q @ U_hat[:, :k]  # (dim, k)
    V = Vt[:k, :].T  # (dim, k)
    return U, S[:k], V


def lanczos(operator, dim, k, iters, device):
    """
    operator: function v -> operator(v), expects v shape [dim]
    dim: dimension L
    k: number of Lanczos vectors kept (>= desired eigenvectors)
    iters: total Lanczos steps
    """
    Q = []
    alphas = []
    betas = []

    # start with random normalized vector
    q = torch.randn(dim, device=device)
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


def _principal_angles(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Principal angles between column spaces of A and B (both (dim, k), assumed orthonormal-ish).
    Returns angles in radians, length k, sorted ascending.
    """
    # Orthonormalize for stability.
    QA, _ = torch.linalg.qr(A, mode="reduced")
    QB, _ = torch.linalg.qr(B, mode="reduced")
    # Singular values of QA^T QB are cosines of principal angles.
    s = torch.linalg.svdvals(QA.T @ QB).clamp(-1.0, 1.0)
    return torch.acos(s)


def svd_via_lanczos(matvec, matvec_t, dim: int, k: int, iters: int, device: str):
    """
    Compute top-k singular triplets using Lanczos on B = M^T M.
    """
    evals, ritz = lanczos(
        operator=lambda v: matvec_t(matvec(v)),
        dim=dim,
        k=max(2 * k, k + 2),
        iters=iters,
        device=device,
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
        mtu_res_list.append(
            torch.linalg.norm(matvec_t(U2[:, i]) - S2s[i] * V2[:, i]) / s
        )
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


if __name__ == "__main__":
    # TODO: turn this into a pytest
    device = "cuda" if torch.cuda.is_available() else "cpu"
    L = 8
    D = 2

    Q = torch.randn(L, D, device=device)
    K = torch.randn(L, D, device=device)

    # Matrix-free SVD of the scores matrix S = QK^T
    matvec = lambda x: matvec_S(Q, K, x)
    matvec_t = lambda x: matvec_ST(Q, K, x)

    # Lanczos-based top-k
    U_l, S_l, V_l = svd_via_lanczos(
        matvec=matvec, matvec_t=matvec_t, dim=L, k=2, iters=20, device=device
    )

    # Randomized SVD top-k
    U_r, S_r, V_r = randomized_svd(
        matvec=matvec, matvec_t=matvec_t, dim=L, k=2, p=6, q=2, device=device
    )

    # Torch SVD
    U_t, S_t, V_t = torch.linalg.svd(Q @ K.T, full_matrices=False)

    metrics = compare_svd_results(
        matvec=matvec, matvec_t=matvec_t, U1=U_l, S1=S_l, V1=V_l, U2=U_r, S2=S_r, V2=V_r
    )
    rprint("comparison:", metrics)

    # A = torch.softmax(Q @ K.T / torch.sqrt(D), dim=1)  # [L, L] (attention matrix)

    # _, d_q_inv_sqrt, d_k_inv_sqrt = compute_degree_normalized_M(A)

    # def matvec(v):
    #     return matvec_M(v, d_q_inv_sqrt, d_k_inv_sqrt)

    # def matvec_t(v):
    #     return matvec_MT(v, d_q_inv_sqrt, d_k_inv_sqrt)

    # # Lanczos-based top-k
    # U_l, S_l, V_l = svd_via_lanczos(
    #     matvec=matvec, matvec_t=matvec_t, dim=L, k=2, iters=20, device=device
    # )

    # # Randomized SVD top-k
    # U_r, S_r, V_r = randomized_svd(
    #     matvec=matvec, matvec_t=matvec_t, dim=L, k=2, p=6, q=2, device=device
    # )

    # metrics = compare_svd_results(
    #     matvec=matvec, matvec_t=matvec_t, U1=U_l, S1=S_l, V1=V_l, U2=U_r, S2=S_r, V2=V_r
    # )
    # print("comparison:", metrics)
