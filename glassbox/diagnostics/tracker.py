"""Tracker diagnostic: features from raw post-softmax attention A."""

from __future__ import annotations

import math
from typing import Any

import torch

from glassbox.attention_tracker import (
    compute_attention_tracker_features_materialized,
    compute_attention_tracker_features_matrix_free,
)
from glassbox.config import TrackerConfig
from glassbox.results import TrackerFeatures


class TrackerDiagnostic:
    signal_name = "tracker"
    features_model = TrackerFeatures

    def __init__(self, config: TrackerConfig):
        self.rank = config.rank
        self.method = config.method
        self.threshold = config.threshold
        self.block_size = config.block_size
        self.causal = config.causal

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
            tier = "materialized"
            features = compute_attention_tracker_features_materialized(A, rank=k)
        else:
            tier = "matrix_free"
            features = compute_attention_tracker_features_matrix_free(
                Qh,
                Kh,
                scale,
                rank=k,
                method=self.method,
                block_size=self.block_size,
                causal=self.causal,
                matvec_strategy=ctx.get("matvec_strategy", "batched"),
            )

        return {
            "features": features,
            "singular_values": features.singular_values,
            "tier": tier,
        }

    def witness(self, Qh: torch.Tensor, Kh: torch.Tensor, L: int, **ctx: Any) -> torch.Tensor:
        raise NotImplementedError("TrackerDiagnostic witness not yet implemented")

    def accumulate(self, local: dict, state: dict | None) -> dict:
        return local
