"""Routing diagnostic: SVD + Hodge decomposition of degree-normalized M."""

from __future__ import annotations

import math
from typing import Any

import torch

from glassbox.hodge import (
    compute_routing_features_materialized,
    compute_routing_features_matrix_free,
)
from glassbox.svd import compute_degree_normalized_M, compute_dk_blocked, compute_logsumexp_blocked


class RoutingDiagnostic:
    signal_name = "routing"

    def __init__(
        self,
        rank: int = 4,
        method: str = "randomized",
        threshold: int = 512,
        block_size: int = 256,
        causal: bool = True,
        hodge_target_cv: float = 0.05,
        hodge_confidence: float = 0.95,
        hodge_pilot_size: int = 100,
        hodge_min_samples: int = 200,
        hodge_curl_seed: int = 42,
    ):
        self.rank = rank
        self.method = method
        self.threshold = threshold
        self.block_size = block_size
        self.causal = causal
        self.hodge_target_cv = hodge_target_cv
        self.hodge_confidence = hodge_confidence
        self.hodge_pilot_size = hodge_pilot_size
        self.hodge_min_samples = hodge_min_samples
        self.hodge_curl_seed = hodge_curl_seed

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
            features = compute_routing_features_materialized(
                M,
                rank=k,
                target_cv=self.hodge_target_cv,
                seed=self.hodge_curl_seed,
            )
        else:
            _, d_k_inv_sqrt = compute_dk_blocked(
                Qh,
                Kh,
                scale,
                self.block_size,
                causal=self.causal,
            )
            lse = compute_logsumexp_blocked(
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
                lse,
                rank=k,
                svd_method=self.method,
                block_size=self.block_size,
                target_cv=self.hodge_target_cv,
                confidence=self.hodge_confidence,
                pilot_size=self.hodge_pilot_size,
                min_samples=self.hodge_min_samples,
                seed=self.hodge_curl_seed,
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
