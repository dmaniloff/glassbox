"""Asymmetry diagnostic: Hodge asymmetry coefficient G = ||P_asym||_F / ||P||_F.

**Operator: the row-stochastic post-softmax attention `P`** (causal-masked), NOT the
degree-normalized `M = D_Q^{-1/2} P D_K^{-1/2}`.  The Hodge/asymmetry family is computed
on `P` because the degree normalization is an *asymmetric* scaling (`D_Q != D_K`): the
*beyond-hodge* paper shows symmetric scaling `D·A·D` preserves antisymmetry while asymmetric
scaling does not (and can inflate the antisymmetric rank), so P (no normalization) keeps the
antisymmetric structure — and the gradient = degree-imbalance reading — clean.  This matches the
*streaming-asym-operators* paper, which decomposes `A = (P - P^T)/2`.  Conductance/Cheeger keeps
`M` (its normalization is theorem-required).  See `docs/operator-choice.md`.

`P_asym = (P - P^T)/2`.  G = ||P_asym||_F / ||P||_F is estimated matrix-free with a direct
Hutchinson estimator on ||P_asym z||^2 (Route B, ``glassbox.hodge``), exact below the
materialize threshold.

Streaming.  G is a *ratio* of Frobenius norms, so it is not additive across windows.  Its
components are: ``||P_asym||_F^2`` and ``||P||_F^2`` are both sums-of-squares over matrix
entries, hence additive over a *disjoint* partition of windows.  ``reduce()`` reports the
global statistic ``G = sqrt(S_asym) / sqrt(S_den)`` from running sufficient statistics
folded out of ``prior_state``, and ``accumulate()`` persists them.  This is unbiased only
under disjoint (tumbling) windowing; under sliding (overlapping) windows the overlap is
double-counted.  The accumulated object is the exact G of the block-diagonal operator over
the processed stream — cross-window asymmetry is not captured.  The per-token asymmetry
profile is emitted per-window as the witness.

(NOTE: streaming requires the backend to thread ``prior_state`` and call ``accumulate``;
in batch dispatch each window reports its own G.)
"""

from __future__ import annotations

import math
from typing import Any

import torch

from glassbox.hodge import EPSILON, asymmetry_partials_and_witness_matrix_free
from glassbox.results import AsymmetryFeatures


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
        incremental: bool = False,
    ):
        self.threshold = threshold
        self.block_size = block_size
        self.causal = causal
        self.n_hutchinson = n_hutchinson
        self.seed = seed
        self.streaming = streaming
        # incremental: exact full-operator G by folding only the delta tokens since the
        # last fire into running (S_asym, S_den).  Requires the unbounded full-sequence
        # buffer (each fire's Qh/Kh is a superset of the previous) and causal attention
        # (adding a token only *adds* its row).  O(L^2) total vs O(L^3/interval) recompute,
        # O(1) state, exact.  See issue: incremental exact-full Hodge.
        self.incremental = incremental
        self._cache: tuple | None = None  # (key, (S_asym, S_den, row_sq, tier))

    def _attention_matrix(
        self, Qh: torch.Tensor, Kh: torch.Tensor, L: int, scale: float
    ) -> torch.Tensor:
        """Row-stochastic post-softmax attention P (causal-masked); NOT degree-normalized."""
        scores = Qh @ Kh.T * scale
        if self.causal:
            scores = scores.masked_fill(
                ~torch.tril(torch.ones(L, L, dtype=torch.bool, device=scores.device)),
                float("-inf"),
            )
        return torch.softmax(scores, dim=-1)

    def _window_stats(self, Qh: torch.Tensor, Kh: torch.Tensor, L: int, seed: int) -> tuple:
        """(S_asym, S_den, row_sq, tier) for one window on the attention operator P.

        S_asym ~ ||P_asym||_F^2, S_den = ||P||_F^2, row_sq[i] ~ per-row asymmetry mass.
        Memoized on input identity so reduce() then witness() reuse one Hutchinson pass.
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
            P = self._attention_matrix(Qh, Kh, L, scale)
            P_asym = (P - P.T) / 2.0
            row_sq = P_asym.square().sum(dim=1)  # [L] exact per-row mass
            S_asym = float(row_sq.sum().item())
            S_den = float(torch.linalg.norm(P, "fro").square().item())
            tier = "materialized"
        else:
            # Operator = P (row-stochastic attention): pass a unit degree vector so the
            # shared matrix-free machinery computes on P = A·I, NOT the degree-normalized M.
            ones = torch.ones(L, device=Qh.device, dtype=Qh.dtype)
            S_asym, S_den, row_sq = asymmetry_partials_and_witness_matrix_free(
                Qh,
                Kh,
                ones,
                scale,
                None,
                self.block_size,
                self.n_hutchinson,
                seed,
                causal=self.causal,
            )
            tier = "matrix_free"

        val = (S_asym, S_den, row_sq, tier)
        self._cache = (key, val)
        return val

    def _incremental_reduce(self, Qh: torch.Tensor, Kh: torch.Tensor, L: int, prior: dict) -> dict:
        """Exact full-operator G by folding only the delta tokens since the last fire.

        On causal P, token t adds row P[t,:t+1]; prior rows are unchanged, so:
            S_den  += ||P[t,:]||^2          (full new-row mass)
            S_asym += (sum_{i<t} P[t,i]^2)/2  (new antisymmetric edges)
        G = sqrt(S_asym/S_den) equals the batch full-operator G exactly.
        """
        incr = (prior.get("partials") or {}).get("incr") if prior else None
        S_asym = incr["S_asym"] if incr else 0.0
        S_den = incr["S_den"] if incr else 0.0
        n_prev = incr["n"] if incr else 0
        scale = 1.0 / math.sqrt(Qh.shape[1])
        for t in range(n_prev, L):
            scores = Qh[t] @ Kh[: t + 1].T * scale  # causal: token t attends to keys 0..t
            p = torch.softmax(scores, dim=-1)
            S_den += float((p * p).sum().item())
            if t > 0:
                off = p[:t]  # off-diagonal new edges (exclude self-attention p[t])
                S_asym += float((off * off).sum().item()) / 2.0
        G = math.sqrt(max(S_asym, 0.0)) / (math.sqrt(max(S_den, 0.0)) + EPSILON)
        return {
            "features": AsymmetryFeatures(G=G),
            "tier": "incremental",
            "partials": {"incr": {"S_asym": S_asym, "S_den": S_den, "n": L}},
        }

    def reduce(self, Qh: torch.Tensor, Kh: torch.Tensor, L: int, **ctx: Any) -> dict:
        if self.incremental:
            return self._incremental_reduce(Qh, Kh, L, ctx.get("prior_state") or {})

        prior = ctx.get("prior_state") or {}
        prev = prior.get("partials") if self.streaming else None

        # Offset the probe seed per window so streaming estimation errors are
        # independent and average out across the accumulated stream.
        seed = self.seed + (prev.get("n_windows", 0) if prev else 0)
        S_asym, S_den, _, tier = self._window_stats(Qh, Kh, L, seed)

        if prev:
            S_asym_tot = prev.get("S_asym", 0.0) + S_asym
            S_den_tot = prev.get("S_den", 0.0) + S_den
            n_windows = prev.get("n_windows", 0) + 1
        else:
            S_asym_tot, S_den_tot, n_windows = S_asym, S_den, 1

        # G = ||P_asym||_F / ||P||_F = sqrt(S_asym) / sqrt(S_den).
        G = math.sqrt(max(S_asym_tot, 0.0)) / (math.sqrt(max(S_den_tot, 0.0)) + EPSILON)

        return {
            "features": AsymmetryFeatures(G=G),
            "tier": tier,
            "partials": {"S_asym": S_asym_tot, "S_den": S_den_tot, "n_windows": n_windows},
        }

    def witness(self, Qh: torch.Tensor, Kh: torch.Tensor, L: int, **ctx: Any) -> torch.Tensor:
        # Per-window localization (base seed); shares reduce()'s pass via the cache in batch
        # mode.  witness[i] = ||P_asym[i,:]|| / ||P||_F, so in batch mode ||witness||_2 == G.
        _, S_den, row_sq, _ = self._window_stats(Qh, Kh, L, self.seed)
        row_norms = torch.sqrt(row_sq.clamp(min=0.0))
        return row_norms / (math.sqrt(max(S_den, 0.0)) + EPSILON)

    def accumulate(self, local: dict, state: dict | None) -> dict:
        # reduce() already folded prior_state into local["partials"] (streaming)
        # or produced window-only partials (batch); persist local either way.
        return local
