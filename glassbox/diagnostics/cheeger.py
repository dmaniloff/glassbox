"""Cheeger sweep-cut diagnostic implementing the Diagnostic protocol.

Supports two modes:
  - Batch (streaming=False): full SVD + sweep every window (default).
  - Streaming (streaming=True): bordered Rayleigh-Ritz between windows,
    full recompute only when triggers fire (gap, degree-shift, geometric stride).
"""

from __future__ import annotations

import math
from typing import Any

import torch

from glassbox.cheeger import (
    compute_cheeger_features_materialized,
    compute_cheeger_features_matrix_free,
    compute_cheeger_witness_materialized,
    compute_cheeger_witness_matrix_free,
    compute_improved_cheeger_upper,
)
from glassbox.results import CheegerFeatures
from glassbox.svd import (
    bordered_rayleigh_ritz,
    compute_degree_normalized_M,
    compute_dk_blocked,
    hermitian_lanczos,
    matvec_Msym_blocked,
)


class CheegerDiagnostic:
    signal_name = "cheeger"

    def __init__(
        self,
        rank: int = 2,
        method: str = "randomized",
        threshold: int = 512,
        block_size: int = 256,
        causal: bool = True,
        streaming: bool = False,
        ritz_rank: int = 3,
        n_explore: int = 2,
        gap_threshold: float = 0.01,
        degree_shift_threshold: float = 0.1,
        geometric_base: float = 1.5,
        lanczos_iters: int = 30,
        improved_cheeger_k: int = 4,
    ):
        self.rank = rank
        self.method = method
        self.threshold = threshold
        self.block_size = block_size
        self.causal = causal
        self.streaming = streaming
        self.ritz_rank = ritz_rank
        self.n_explore = n_explore
        self.gap_threshold = gap_threshold
        self.degree_shift_threshold = degree_shift_threshold
        self.geometric_base = geometric_base
        self.lanczos_iters = lanczos_iters
        self.improved_cheeger_k = improved_cheeger_k

    def reduce(self, Qh: torch.Tensor, Kh: torch.Tensor, L: int, **ctx: Any) -> dict:
        prior_state = ctx.get("prior_state")

        if not self.streaming or prior_state is None:
            return self._reduce_full(Qh, Kh, L)

        ritz_basis = prior_state.get("ritz_basis")
        if ritz_basis is None:
            return self._reduce_full(Qh, Kh, L)

        if self._check_triggers(prior_state, Qh, Kh, L):
            return self._reduce_full(Qh, Kh, L, warm_start=ritz_basis)

        return self._reduce_cheap(Qh, Kh, L, prior_state)

    def _reduce_full(
        self,
        Qh: torch.Tensor,
        Kh: torch.Tensor,
        L: int,
        warm_start: torch.Tensor | None = None,
    ) -> dict:
        """Full SVD + sweep. Used on first window and when triggers fire."""
        scale = 1.0 / math.sqrt(Qh.shape[1])
        k = min(self.rank, L - 1)

        # Batch path: materialized or matrix-free SVD + sweep
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
                Qh, Kh, d_k_inv_sqrt, scale,
                rank=k, svd_method=self.method,
                block_size=self.block_size, causal=self.causal,
            )

        result = {"features": features, "tier": tier}

        if not self.streaming:
            return result

        # Streaming: also run Lanczos on M_sym for Ritz basis + gap monitoring
        d_k, d_k_inv_sqrt = compute_dk_blocked(
            Qh, Kh, scale, self.block_size, causal=self.causal
        )

        def msym_matvec(v):
            return matvec_Msym_blocked(
                Qh, Kh, v, d_k_inv_sqrt, scale, self.block_size, causal=self.causal
            )

        init_vecs = None
        if warm_start is not None and warm_start.shape[0] != L:
            adapted = torch.zeros(L, warm_start.shape[1], device=Qh.device, dtype=Qh.dtype)
            common = min(warm_start.shape[0], L)
            adapted[-common:] = warm_start[-common:]
            init_vecs = adapted
        elif warm_start is not None:
            init_vecs = warm_start

        r = max(self.ritz_rank, self.rank + 1, self.improved_cheeger_k + 1)
        evals, evecs = hermitian_lanczos(
            msym_matvec, L, r, self.lanczos_iters, str(Qh.device),
            which="largest", dtype=Qh.dtype,
            initial_vectors=init_vecs,
        )

        gap = float(evals[1] - evals[2]) if len(evals) > 2 else float("inf")
        sigma2_msym = float(evals[1]) if len(evals) > 1 else 0.0
        mu2 = max(1.0 - sigma2_msym, 0.0)

        improved_upper = compute_improved_cheeger_upper(evals, self.improved_cheeger_k)

        bracket_lower = mu2 / 2.0
        bracket_upper = math.sqrt(max(2.0 * mu2, 0.0))

        updated_features = CheegerFeatures(
            phi_star=features.phi_star,
            sigma2=features.sigma2,
            cheeger_lower=bracket_lower,
            cheeger_upper=bracket_upper,
            phi_hat=features.phi_star,
            improved_upper=improved_upper,
            bracket_width=bracket_upper - bracket_lower,
            spectral_gap=gap,
            recomputed=True,
        )

        result["features"] = updated_features
        result["ritz_basis"] = evecs[:, :r].detach()
        result["eigenvalues"] = evals[:r].detach()
        result["sigma2"] = sigma2_msym
        result["phi_star_val"] = features.phi_star
        result["gap"] = gap
        result["d_k_prev"] = d_k.detach()
        result["recomputed"] = True

        return result

    def _reduce_cheap(
        self,
        Qh: torch.Tensor,
        Kh: torch.Tensor,
        L: int,
        prior_state: dict,
    ) -> dict:
        """Bordered Rayleigh-Ritz update between full recomputes."""
        scale = 1.0 / math.sqrt(Qh.shape[1])
        d_k, d_k_inv_sqrt = compute_dk_blocked(
            Qh, Kh, scale, self.block_size, causal=self.causal
        )

        def msym_matvec(v):
            return matvec_Msym_blocked(
                Qh, Kh, v, d_k_inv_sqrt, scale, self.block_size, causal=self.causal
            )

        ritz_basis_prev = prior_state["ritz_basis"]

        r = max(self.ritz_rank, self.rank + 1, self.improved_cheeger_k + 1)
        evals, evecs, all_evals = bordered_rayleigh_ritz(
            msym_matvec, L, ritz_basis_prev,
            k=r,
            n_explore=self.n_explore,
            device=str(Qh.device),
            which="largest",
            dtype=Qh.dtype,
        )

        sigma2_msym = float(evals[1]) if len(evals) > 1 else 0.0
        gap = float(evals[1] - evals[2]) if len(evals) > 2 else float("inf")
        mu2 = max(1.0 - sigma2_msym, 0.0)

        bracket_lower = mu2 / 2.0
        bracket_upper = math.sqrt(max(2.0 * mu2, 0.0))

        phi_star_prev = prior_state.get("phi_star_val", 0.0)

        improved_upper = None
        if gap < self.gap_threshold:
            improved_upper = compute_improved_cheeger_upper(evals, self.improved_cheeger_k)

        features = CheegerFeatures(
            phi_star=phi_star_prev,
            sigma2=sigma2_msym,
            cheeger_lower=bracket_lower,
            cheeger_upper=bracket_upper,
            phi_hat=phi_star_prev if gap >= self.gap_threshold else None,
            improved_upper=improved_upper,
            bracket_width=bracket_upper - bracket_lower,
            spectral_gap=gap,
            recomputed=False,
        )

        return {
            "features": features,
            "tier": "bordered_ritz",
            "ritz_basis": evecs.detach(),
            "eigenvalues": evals.detach(),
            "sigma2": sigma2_msym,
            "phi_star_val": phi_star_prev,
            "gap": gap,
            "d_k_prev": d_k.detach(),
            "recomputed": False,
        }

    def _check_triggers(
        self,
        prior_state: dict,
        Qh: torch.Tensor,
        Kh: torch.Tensor,
        L: int,
    ) -> bool:
        step = prior_state.get("step", 0)
        last_full = prior_state.get("last_full_recompute_step", 0)

        # Geometric stride: fire when steps since last recompute exceeds
        # the current stride threshold (tracked in state, grows geometrically)
        stride_next = prior_state.get("geometric_stride_next", 2)
        steps_since = step - last_full
        if steps_since >= stride_next:
            return True

        # Bracket-width / gap monitor
        gap = prior_state.get("gap", float("inf"))
        if gap < self.gap_threshold:
            return True

        # Degree-shift monitor
        d_k_prev = prior_state.get("d_k_prev")
        if d_k_prev is not None:
            scale = 1.0 / math.sqrt(Qh.shape[1])
            d_k_new, _ = compute_dk_blocked(
                Qh, Kh, scale, self.block_size, causal=self.causal
            )
            if d_k_prev.shape[0] != L:
                aligned = torch.zeros(L, device=d_k_prev.device, dtype=d_k_prev.dtype)
                common = min(d_k_prev.shape[0], L)
                aligned[-common:] = d_k_prev[-common:]
                d_k_prev = aligned
            ratio = torch.sqrt(d_k_prev.clamp(min=1e-12) / d_k_new.clamp(min=1e-12))
            r_t = (1.0 - ratio).abs().mean().item()
            if r_t > self.degree_shift_threshold:
                return True

        return False

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
                Qh, Kh, d_k_inv_sqrt, scale,
                rank=k, svd_method=self.method,
                block_size=self.block_size, causal=self.causal,
            )

    def accumulate(self, local: dict, state: dict | None) -> dict:
        if not self.streaming:
            return local

        new_state = dict(local)

        if state is None:
            if new_state.get("recomputed"):
                new_state["last_full_recompute_step"] = new_state.get("step", 0)
            return new_state

        for key in (
            "ritz_basis",
            "eigenvalues",
            "sigma2",
            "phi_star_val",
            "gap",
            "d_k_prev",
            "last_full_recompute_step",
            "geometric_stride_next",
        ):
            if key not in new_state and key in state:
                new_state[key] = state[key]

        if new_state.get("recomputed"):
            new_state["last_full_recompute_step"] = new_state.get("step", 0)
            prev_stride = state.get("geometric_stride_next", 1) if state else 1
            new_state["geometric_stride_next"] = max(
                math.ceil(self.geometric_base * max(prev_stride, 1)), 2
            )

        return new_state
