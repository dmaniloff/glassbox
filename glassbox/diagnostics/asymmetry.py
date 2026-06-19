"""Asymmetry diagnostic: Hodge G = ||M_asym||_F / ||M||_F of degree-normalized M.

G is estimated matrix-free with a direct Hutchinson estimator on ||M_asym z||^2
(Route B, ``glassbox.hodge``), exact below the materialize threshold.

Streaming (issue #39).  G is a *ratio* of Frobenius norms, so it is not additive
across windows.  Its components are: ``||M_asym||_F^2`` and ``||M||_F^2`` are both
sums-of-squares over matrix entries, hence additive over a *disjoint* partition of
windows.  ``reduce()`` therefore reports the global statistic
``G = sqrt(S_asym) / sqrt(S_M)`` from running sufficient statistics folded out of
``prior_state``, and ``accumulate()`` persists them.  This is unbiased only under
disjoint (tumbling) windowing; under sliding (overlapping) windows the overlap is
double-counted.  The accumulated object is the exact G of the block-diagonal
operator over the processed stream — cross-window asymmetry is not captured.  The
per-token asymmetry profile is emitted per-window as the witness (positions differ
across windows, so it is not accumulated).

Performance: a window's scalar partials and per-row witness share ONE multi-RHS
Hutchinson pass, memoized so the backend's reduce()-then-witness() calls do not
recompute it.  Streaming windows offset the probe seed so per-window estimation
errors decorrelate and average out across the stream.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from glassbox.hodge import EPSILON, asymmetry_partials_and_witness_matrix_free
from glassbox.results import AsymmetryFeatures
from glassbox.svd import compute_degree_normalized_M, compute_dk_blocked


class AsymmetryDiagnostic:
    signal_name = "asymmetry"

    def __init__(
        self,
        threshold: int = 512,
        block_size: int = 256,
        causal: bool = True,
        n_hutchinson: int = 32,
        seed: int = 42,
        streaming: bool = False,
    ):
        self.threshold = threshold
        self.block_size = block_size
        self.causal = causal
        self.n_hutchinson = n_hutchinson
        self.seed = seed
        self.streaming = streaming
        self._cache: tuple | None = None  # (key, (S_asym, S_M, row_sq, tier))

    def _materialize_M(
        self, Qh: torch.Tensor, Kh: torch.Tensor, L: int, scale: float
    ) -> torch.Tensor:
        scores = Qh @ Kh.T * scale
        if self.causal:
            scores = scores.masked_fill(
                ~torch.tril(torch.ones(L, L, dtype=torch.bool, device=scores.device)),
                float("-inf"),
            )
        A = torch.softmax(scores, dim=-1)
        M, _, _ = compute_degree_normalized_M(A)
        return M

    def _window_stats(self, Qh: torch.Tensor, Kh: torch.Tensor, L: int, seed: int) -> tuple:
        """(S_asym, S_M, row_sq, tier) for one window, from a single shared pass.

        Memoized on the input identity so reduce() then witness() over the same
        (Qh, Kh) reuse one Hutchinson pass.  Seed is irrelevant below threshold
        (exact), so it is excluded from the key there.
        """
        key = (
            Qh.data_ptr(),
            Kh.data_ptr(),
            tuple(Qh.shape),
            tuple(Kh.shape),
            L,
            self.n_hutchinson,
            None if L <= self.threshold else seed,
        )
        if self._cache is not None and self._cache[0] == key:
            return self._cache[1]

        scale = 1.0 / math.sqrt(Qh.shape[1])
        if L <= self.threshold:
            M = self._materialize_M(Qh, Kh, L, scale)
            M_asym = (M - M.T) / 2.0
            row_sq = M_asym.square().sum(dim=1)  # [L] exact per-row mass
            S_asym = float(row_sq.sum().item())
            S_M = float(torch.linalg.norm(M, "fro").square().item())
            tier = "materialized"
        else:
            _, d_k_inv_sqrt = compute_dk_blocked(Qh, Kh, scale, self.block_size, causal=self.causal)
            S_asym, S_M, row_sq = asymmetry_partials_and_witness_matrix_free(
                Qh,
                Kh,
                d_k_inv_sqrt,
                scale,
                None,
                self.block_size,
                self.n_hutchinson,
                seed,
                causal=self.causal,
            )
            tier = "matrix_free"

        val = (S_asym, S_M, row_sq, tier)
        self._cache = (key, val)
        return val

    def reduce(self, Qh: torch.Tensor, Kh: torch.Tensor, L: int, **ctx: Any) -> dict:
        prior = ctx.get("prior_state") or {}
        prev = prior.get("partials") if self.streaming else None

        # Offset the probe seed per window so streaming estimation errors are
        # independent and average out across the accumulated stream.
        seed = self.seed + (prev.get("n_windows", 0) if prev else 0)
        S_asym, S_M, _, tier = self._window_stats(Qh, Kh, L, seed)

        if prev:
            S_asym_tot = prev.get("S_asym", 0.0) + S_asym
            S_M_tot = prev.get("S_M", 0.0) + S_M
            n_windows = prev.get("n_windows", 0) + 1
        else:
            S_asym_tot, S_M_tot, n_windows = S_asym, S_M, 1

        # G = ||M_asym||_F / ||M||_F = sqrt(S_asym) / sqrt(S_M); matches
        # compute_G_materialized exactly for a single materialized window.
        G = math.sqrt(max(S_asym_tot, 0.0)) / (math.sqrt(max(S_M_tot, 0.0)) + EPSILON)

        return {
            "features": AsymmetryFeatures(G=G),
            "tier": tier,
            "partials": {"S_asym": S_asym_tot, "S_M": S_M_tot, "n_windows": n_windows},
        }

    def witness(self, Qh: torch.Tensor, Kh: torch.Tensor, L: int, **ctx: Any) -> torch.Tensor:
        # Per-window localization (base seed); shares reduce()'s pass via the cache
        # in batch mode.  witness[i] = ||M_asym[i,:]|| / ||M||_F, so in batch mode
        # ||witness||_2 == the window G.
        _, S_M, row_sq, _ = self._window_stats(Qh, Kh, L, self.seed)
        row_norms = torch.sqrt(row_sq.clamp(min=0.0))
        return row_norms / (math.sqrt(max(S_M, 0.0)) + EPSILON)

    def accumulate(self, local: dict, state: dict | None) -> dict:
        # reduce() already folded prior_state into local["partials"] (streaming)
        # or produced window-only partials (batch); persist local either way.
        return local
