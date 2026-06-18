"""Shared test helpers."""

import math

import torch

from glassbox.svd import compute_degree_normalized_M


def make_M(L, D, seed=42):
    """Generate Q, K, scale, A, M and related quantities."""
    torch.manual_seed(seed)
    Q = torch.randn(L, D)
    K = torch.randn(L, D)
    scale = 1.0 / math.sqrt(D)
    A = torch.softmax(Q @ K.T * scale, dim=-1)
    M, _, d_k_inv_sqrt = compute_degree_normalized_M(A)
    return Q, K, scale, A, M, d_k_inv_sqrt
