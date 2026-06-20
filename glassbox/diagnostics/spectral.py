"""Spectral diagnostic: SVD of pre-softmax scores matrix S = QK^T."""

from __future__ import annotations

from typing import Any

import torch

from glassbox.svd import compute_scores_matrix_features


class SpectralDiagnostic:
    signal_name = "spectral"

    def __init__(self, rank: int = 4, method: str = "randomized"):
        self.rank = rank
        self.method = method

    def reduce(self, Qh: torch.Tensor, Kh: torch.Tensor, L: int, **ctx: Any) -> dict:
        features = compute_scores_matrix_features(
            Qh,
            Kh,
            rank=self.rank,
            method=self.method,
        )
        return {
            "features": features,
            "singular_values": features.singular_values,
        }

    def witness(self, Qh: torch.Tensor, Kh: torch.Tensor, L: int, **ctx: Any) -> torch.Tensor:
        raise NotImplementedError("SpectralDiagnostic witness not yet implemented")

    def accumulate(self, local: dict, state: dict | None) -> dict:
        return local
