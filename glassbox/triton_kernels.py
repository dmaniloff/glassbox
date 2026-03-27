"""Fused Triton kernels for glassbox matrix-free SVD.

fused_attn_multi_matvec: computes softmax(Q @ K^T * scale) @ Omega
using online softmax (never materialises full attention rows) and a single
kernel launch.  Forward direction only — the transpose direction
(A^T @ U) stays PyTorch-batched because online softmax is incompatible
with transpose accumulation.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_attn_multi_matvec_kernel(
    output_ptr,
    Q_ptr,
    K_ptr,
    Omega_ptr,
    scale,
    L_q,
    L_k,
    stride_q0,
    stride_k0,
    stride_o0,
    stride_out0,
    D: tl.constexpr,
    N_VECS: tl.constexpr,
    D_PAD: tl.constexpr,
    N_VECS_PAD: tl.constexpr,
    TILE_Q: tl.constexpr,
    TILE_K: tl.constexpr,
):
    """Online-softmax fused attention × multi-vector multiply.

    Each program instance handles one TILE_Q-row strip of Q, iterates over
    all K tiles, and writes TILE_Q rows × N_VECS columns of the output.
    """
    pid = tl.program_id(0)
    q_start = pid * TILE_Q

    offs_q = q_start + tl.arange(0, TILE_Q)
    offs_d = tl.arange(0, D_PAD)
    offs_nv = tl.arange(0, N_VECS_PAD)

    q_mask = offs_q < L_q
    d_mask = offs_d < D
    nv_mask = offs_nv < N_VECS

    # Load Q tile: [TILE_Q, D_PAD]
    Q_tile = tl.load(
        Q_ptr + offs_q[:, None] * stride_q0 + offs_d[None, :],
        mask=q_mask[:, None] & d_mask[None, :],
        other=0.0,
    )

    # Online softmax running state
    m_i = tl.full([TILE_Q], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([TILE_Q], dtype=tl.float32)
    acc = tl.zeros([TILE_Q, N_VECS_PAD], dtype=tl.float32)

    num_k_tiles = tl.cdiv(L_k, TILE_K)
    for j in range(num_k_tiles):
        k_start = j * TILE_K
        offs_k = k_start + tl.arange(0, TILE_K)
        k_mask = offs_k < L_k

        # K tile: [TILE_K, D_PAD]
        K_tile = tl.load(
            K_ptr + offs_k[:, None] * stride_k0 + offs_d[None, :],
            mask=k_mask[:, None] & d_mask[None, :],
            other=0.0,
        )

        # S = Q_tile @ K_tile^T * scale : [TILE_Q, TILE_K]
        S = tl.dot(Q_tile, tl.trans(K_tile)) * scale
        S = tl.where(k_mask[None, :], S, float("-inf"))

        # Online softmax update
        m_j = tl.maximum(m_i, tl.max(S, axis=1))
        m_j = tl.where(m_j > float("-inf"), m_j, 0.0)
        P = tl.exp(S - m_j[:, None])
        l_j = tl.sum(P, axis=1)
        alpha = tl.exp(m_i - m_j)

        acc = acc * alpha[:, None]
        m_i = m_j
        l_i = l_i * alpha + l_j

        # Omega tile: [TILE_K, N_VECS_PAD]
        Omega_tile = tl.load(
            Omega_ptr + offs_k[:, None] * stride_o0 + offs_nv[None, :],
            mask=k_mask[:, None] & nv_mask[None, :],
            other=0.0,
        )

        # Accumulate P @ Omega_tile
        acc += tl.dot(P.to(Omega_tile.dtype), Omega_tile)

    # Final normalisation
    acc = acc / l_i[:, None]

    # Store [TILE_Q, N_VECS] (only unpadded columns)
    tl.store(
        output_ptr + offs_q[:, None] * stride_out0 + offs_nv[None, :],
        acc,
        mask=q_mask[:, None] & nv_mask[None, :],
    )


def fused_attn_multi_matvec(
    Q: torch.Tensor,
    K: torch.Tensor,
    Omega: torch.Tensor,
    scale: float,
    tile_q: int = 64,
    tile_k: int = 64,
) -> torch.Tensor:
    """Compute softmax(Q @ K^T * scale) @ Omega via fused Triton kernel.

    Uses online softmax so peak SRAM usage is O(TILE_Q × TILE_K), never
    O(L_q × L_k).  Single kernel launch vs (L/block_size) × 3 for the
    PyTorch blocked version.

    Only implements the forward direction (A @ Omega).  The transpose
    (A^T @ U) stays PyTorch-batched because online softmax is over K
    columns but the transpose output is K-indexed.

    Args:
        Q: [L_q, d] query matrix.
        K: [L_k, d] key matrix.
        Omega: [L_k, n_vecs] test-vector matrix.
        scale: attention scale factor (typically 1/sqrt(d)).
        tile_q: Q-dimension tile size (default 64).
        tile_k: K-dimension tile size (default 64).

    Returns:
        [L_q, n_vecs] result of A @ Omega.
    """
    L_q, d = Q.shape
    L_k = K.shape[0]
    n_vecs = Omega.shape[1]

    # tl.dot requires constexpr power-of-2 dimensions
    D_PAD = max(triton.next_power_of_2(d), 16)
    N_VECS_PAD = max(triton.next_power_of_2(n_vecs), 16)

    # Ensure Q, K, Omega are contiguous (strides assume row-major)
    Q = Q.contiguous()
    K = K.contiguous()
    Omega = Omega.contiguous()

    output = torch.empty(L_q, n_vecs, device=Q.device, dtype=Q.dtype)

    grid = (triton.cdiv(L_q, tile_q),)

    _fused_attn_multi_matvec_kernel[grid](
        output,
        Q,
        K,
        Omega,
        scale,
        L_q,
        L_k,
        Q.stride(0),
        K.stride(0),
        Omega.stride(0),
        output.stride(0),
        D=d,
        N_VECS=n_vecs,
        D_PAD=D_PAD,
        N_VECS_PAD=N_VECS_PAD,
        TILE_Q=tile_q,
        TILE_K=tile_k,
    )

    return output
