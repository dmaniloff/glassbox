"""Cheeger sweep-cut diagnostic implementing the Diagnostic protocol."""

from __future__ import annotations

import math
from typing import Any

import torch

from glassbox.cheeger import (
    compute_cheeger_features_materialized,
    compute_cheeger_features_matrix_free,
    compute_cheeger_witness_materialized,
    compute_cheeger_witness_matrix_free,
)
from glassbox.svd import compute_degree_normalized_M, compute_dk_blocked


class CheegerDiagnostic:
    signal_name = "cheeger"

    def __init__(
        self,
        rank: int = 2,
        method: str = "randomized",
        threshold: int = 512,
        block_size: int = 256,
        causal: bool = True,
    ):
        self.rank = rank
        self.method = method
        self.threshold = threshold
        self.block_size = block_size
        self.causal = causal

    def reduce(self, Qh: torch.Tensor, Kh: torch.Tensor, L: int, **ctx: Any) -> dict:
        scale = 1.0 / math.sqrt(Qh.shape[1])
        k = min(self.rank, L - 1)

        if L <= self.threshold:
            scores = Qh @ Kh.T * scale
            if self.causal:
                scores = scores.masked_fill(
                    ~torch.tril(torch.ones(L, L, dtype=torch.bool, device=scores.device)),
                    float("-inf"),
                )
            A = torch.softmax(scores, dim=-1)
            M, _, _ = compute_degree_normalized_M(A)
            tier = "materialized"
            features = compute_cheeger_features_materialized(M, rank=k)
        else:
            _, d_k_inv_sqrt = compute_dk_blocked(
                Qh, Kh, scale, self.block_size, causal=self.causal
            )
            tier = "matrix_free"
            features = compute_cheeger_features_matrix_free(
                Qh,
                Kh,
                d_k_inv_sqrt,
                scale,
                rank=k,
                svd_method=self.method,
                block_size=self.block_size,
                causal=self.causal,
            )

        return {"features": features, "tier": tier}

    def witness(self, Qh: torch.Tensor, Kh: torch.Tensor, L: int, **ctx: Any) -> torch.Tensor:
        scale = 1.0 / math.sqrt(Qh.shape[1])
        k = min(self.rank, L - 1)

        if L <= self.threshold:
            scores = Qh @ Kh.T * scale
            if self.causal:
                scores = scores.masked_fill(
                    ~torch.tril(torch.ones(L, L, dtype=torch.bool, device=scores.device)),
                    float("-inf"),
                )
            A = torch.softmax(scores, dim=-1)
            M, _, _ = compute_degree_normalized_M(A)
            return compute_cheeger_witness_materialized(M, rank=k)
        else:
            _, d_k_inv_sqrt = compute_dk_blocked(
                Qh, Kh, scale, self.block_size, causal=self.causal
            )
            return compute_cheeger_witness_matrix_free(
                Qh,
                Kh,
                d_k_inv_sqrt,
                scale,
                rank=k,
                svd_method=self.method,
                block_size=self.block_size,
                causal=self.causal,
            )

    def accumulate(self, local: dict, state: dict | None) -> dict:
        return local
