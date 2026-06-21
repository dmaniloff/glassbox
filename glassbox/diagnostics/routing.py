"""Routing diagnostic: SVD + Hodge decomposition of degree-normalized M."""

from __future__ import annotations

import math
from typing import Any

import torch

from glassbox.hodge import (
    compute_routing_features_materialized,
    compute_routing_features_matrix_free,
)
from glassbox.svd import compute_degree_normalized_M, compute_dk_blocked


class RoutingDiagnostic:
    signal_name = "routing"

    def __init__(
        self,
        rank: int = 4,
        method: str = "randomized",
        threshold: int = 512,
        block_size: int = 256,
        causal: bool = True,
        hodge_seed: int = 42,
    ):
        self.rank = rank
        self.method = method
        self.threshold = threshold
        self.block_size = block_size
        self.causal = causal
        self.hodge_seed = hodge_seed

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
            features = compute_routing_features_materialized(M, rank=k)
        else:
            _, d_k_inv_sqrt = compute_dk_blocked(
                Qh,
                Kh,
                scale,
                self.block_size,
                causal=self.causal,
            )
            tier = "matrix_free"
            features = compute_routing_features_matrix_free(
                Qh,
                Kh,
                d_k_inv_sqrt,
                scale,
                rank=k,
                svd_method=self.method,
                block_size=self.block_size,
                seed=self.hodge_seed,
                causal=self.causal,
            )

        return {
            "features": features,
            "singular_values": features.singular_values,
            "tier": tier,
        }

    def witness(self, Qh: torch.Tensor, Kh: torch.Tensor, L: int, **ctx: Any) -> torch.Tensor:
        raise NotImplementedError("RoutingDiagnostic witness not yet implemented")

    def accumulate(self, local: dict, state: dict | None) -> dict:
        return local
