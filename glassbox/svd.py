"""
Explorations of matrix-free SVD algorithms.

A = softmax(QK^T / sqrt(d))
M = Dq_inv_sqrt * A * Dk_inv_sqrt
M is NOT symmetric in general. We compute SVD, not eigen-decomposition, using matrix–vector products with M and M^T
"""

import torch

L = 8
ATTENTION_MATRIX = torch.randn(L, L)


def apply_A(x):
    return ATTENTION_MATRIX @ x


def apply_AT(x):
    return ATTENTION_MATRIX.T @ x


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
        D_Q_sqrt_inv: Query degree matrix D_Q^{-1/2}
        D_K_sqrt_inv: Key degree matrix D_K^{-1/2}
    """
    # Compute row sums (query degrees): d_Q_i = sum_j A_ij
    d_q = A.sum(dim=1)  # shape: (n_q,)

    # Compute column sums (key degrees): d_K_j = sum_i A_ij
    d_k = A.sum(dim=0)  # shape: (n_k,)

    # Compute D_Q^{-1/2} with numerical stability (Equation 2 in paper)
    # D^{-1/2}_ii = (d_ii + epsilon)^{-1/2}
    Dq_inv_sqrt = torch.diag(1.0 / torch.sqrt(d_q + epsilon))

    # Compute D_K^{-1/2} with numerical stability
    Dk_inv_sqrt = torch.diag(1.0 / torch.sqrt(d_k + epsilon))

    # Compute M = D_Q^{-1/2} @ A @ D_K^{-1/2}
    M = Dq_inv_sqrt @ A @ Dk_inv_sqrt

    return M, Dq_inv_sqrt, Dk_inv_sqrt


def apply_M(x, Dq_inv_sqrt, Dk_inv_sqrt):
    # diag-scale input
    x1 = Dk_inv_sqrt * x  # [L]
    y1 = apply_A(x1)  # attention @ x1   (FlashAttn-like run with dv=1)
    y = Dq_inv_sqrt * y1  # [L]
    return y


def apply_MT(x, Dq_inv_sqrt, Dk_inv_sqrt):
    x1 = Dq_inv_sqrt * x
    y1 = apply_AT(x1)  # use A^T @ x1 operator
    y = Dk_inv_sqrt * y1
    return y


def matvec_M(v):
    return apply_M(v)


def matvec_MT(v):
    return apply_MT(v)


def matvec_B(v):
    return matvec_MT(matvec_M(v))


def randomized_svd(matvec, matvec_t, dim, k, p=5, q=2, device="cuda"):
    # Step 1: random test matrix
    Omega = torch.randn(dim, k + p, device=device)

    # Step 2: sample Y = M Omega
    Y = torch.stack([matvec(Omega[:, i]) for i in range(k + p)], dim=1)

    # optional: power iterations
    for _ in range(q):
        Z = torch.stack([matvec_t(Y[:, i]) for i in range(Y.shape[1])], dim=1)
        Y = torch.stack([matvec(Z[:, i]) for i in range(Z.shape[1])], dim=1)

    # Step 3: orthonormal QR basis
    Q, _ = torch.linalg.qr(Y)  # [L, k+p]

    # Step 4: compute B = Qᵀ M
    B = torch.stack([matvec(Q[:, i]) for i in range(Q.shape[1])], dim=1)  # [L, k+p]
    B = Q.T @ B  # small (k+p) x (k+p)

    # Step 5: SVD of small B
    U_tilde, S, Vt = torch.linalg.svd(B)

    # Step 6: left singular vectors in original space
    U = Q @ U_tilde[:, :k]
    V = Vt[:k, :].T

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

        if len(Q) > k:
            break

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


if __name__ == "__main__":
    evals, ritz = lanczos(
        operator=lambda v: apply_MT(apply_M(v)),
        dim=L,
        k=4,  # want top 2 singular values => use k>=4
        iters=20,
        device="cuda",
    )
    U, S, V = randomized_svd(matvec=apply_M, matvec_t=apply_MT, dim=L, k=2)
