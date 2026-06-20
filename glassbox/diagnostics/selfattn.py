"""Self-attention diagonal diagnostic (LLM-Check, NeurIPS 2024)."""

from __future__ import annotations

import math
from typing import Any

import torch

from glassbox.attention_diagonal import (
    compute_attention_diagonal_features_materialized,
    compute_attention_diagonal_features_matrix_free,
)


class SelfAttnDiagnostic:
    signal_name = "selfattn"

    def __init__(
        self,
        top_k: int = 10,
        threshold: int = 512,
        block_size: int = 256,
        causal: bool = True,
    ):
        self.top_k = top_k
        self.threshold = threshold
        self.block_size = block_size
        self.causal = causal

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
