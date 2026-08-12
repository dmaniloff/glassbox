"""Self-attention diagonal diagnostic (LLM-Check, NeurIPS 2024)."""

from __future__ import annotations

import math
from typing import Any

import torch

from glassbox.attention_diagonal import (
    compute_attention_diagonal_features_materialized,
    compute_attention_diagonal_features_matrix_free,
)
from glassbox.config import SelfAttnConfig
from glassbox.results import SelfAttnFeatures


class SelfAttnDiagnostic:
    signal_name = "selfattn"
    features_model = SelfAttnFeatures

    def __init__(self, config: SelfAttnConfig):
        self.top_k = config.top_k
        self.threshold = config.threshold
        self.block_size = config.block_size
        self.causal = config.causal

    def reduce(self, Qh: torch.Tensor, Kh: torch.Tensor, L: int, **ctx: Any) -> dict:
        scale = 1.0 / math.sqrt(Qh.shape[1])

        if L <= self.threshold:
            scores = Qh @ Kh.T * scale
            if self.causal:
                scores = scores.masked_fill(
                    ~torch.tril(torch.ones(L, L, dtype=torch.bool, device=scores.device)),
                    float("-inf"),
                )
            A = torch.softmax(scores, dim=-1)
            tier = "materialized"
            features = compute_attention_diagonal_features_materialized(A, top_k=self.top_k)
        else:
            tier = "matrix_free"
            features = compute_attention_diagonal_features_matrix_free(
                Qh,
                Kh,
                scale,
                top_k=self.top_k,
                block_size=self.block_size,
                causal=self.causal,
            )

        return {"features": features, "tier": tier}

    def witness(self, Qh: torch.Tensor, Kh: torch.Tensor, L: int, **ctx: Any) -> torch.Tensor:
        raise NotImplementedError("SelfAttnDiagnostic witness not yet implemented")

    def accumulate(self, local: dict, state: dict | None) -> dict:
        return local
