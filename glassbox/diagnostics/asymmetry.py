"""Asymmetry diagnostic: Hodge asymmetry coefficient G = ||P_asym||_F / ||P||_F.

**Operator: the row-stochastic post-softmax attention `P`** (causal-masked), NOT the
degree-normalized `M = D_Q^{-1/2} P D_K^{-1/2}`.  The Hodge/asymmetry family is computed
on `P` because the degree normalization is an *asymmetric* scaling (`D_Q != D_K`): ShadeFormal
`NormalizationInvariance` proves only that *symmetric* scaling `D·A·D` preserves antisymmetry
(the asymmetric-scaling-inflates-the-antisymmetric-rank direction is a paper remark, not a
formalized theorem), so P (no normalization) is the operator whose antisymmetric structure is
certified clean — keeping the gradient = degree-imbalance reading faithful.  This matches the
streaming-asym-operators paper, which decomposes `A = (P - P^T)/2`.  Conductance/Cheeger keeps
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
        """(S_asym, S_den, grad_energy, row_sq, tier) for one window on the operator P.

        S_asym ~ ||P_asym||_F^2, S_den = ||P||_F^2, row_sq[i] ~ per-row asymmetry mass,
        and grad_energy = 2||r||^2/L (exact Hodge gradient energy, r = A_asym @ 1) for the
        Gamma/C split.  Memoized on input identity so reduce() then witness() reuse one pass.
        """
        # Seed is deliberately NOT in the key: in streaming mode reduce() uses seed
        # self.seed + n_windows while witness() uses self.seed, so keying on seed would
        # miss and recompute a second matrix-free pass with different probes (breaking
        # ||witness||_2 == G). With the per-reduce cache clear, witness() reuses the exact
        # pass (and probes) reduce() just computed this fire.
        key = (
            Qh.data_ptr(),
            Kh.data_ptr(),
            tuple(Qh.shape),
            tuple(Kh.shape),
            L,
            self.n_hutchinson,
        )
        if self._cache is not None and self._cache[0] == key:
            return self._cache[1]

        scale = 1.0 / math.sqrt(Qh.shape[1])
        if L <= self.threshold:
            # fp32 throughout: sums-of-squares over L^2 entries lose precision in fp16/bf16.
            P = self._attention_matrix(Qh, Kh, L, scale).to(torch.float32)
            P_asym = (P - P.T) / 2.0
            row_sq = P_asym.square().sum(dim=1)  # [L] exact per-row mass
            S_asym = float(row_sq.sum().item())
            S_den = float(P.square().sum().item())  # ||P||_F^2 (one pass, reuses P)
            r = P_asym.sum(dim=1)  # exact row-sum vector A_asym @ 1
            tier = "materialized"
        else:
            # Operator = P (row-stochastic attention): pass a unit degree vector so the
            # shared matrix-free machinery computes on P = A·I, NOT the degree-normalized M.
            ones = torch.ones(L, device=Qh.device, dtype=Qh.dtype)
            S_asym, S_den, row_sq, r = asymmetry_partials_and_witness_matrix_free(
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

        grad_energy = 2.0 * float((r * r).sum().item()) / L  # ||A_grad||^2 = 2||r||^2/L
        val = (S_asym, S_den, grad_energy, row_sq, tier)
        self._cache = (key, val)
        return val

    def _split(self, S_asym: float, S_den: float, grad_energy: float) -> tuple:
        """(G, Gamma, C) from sufficient statistics; G^2 = Gamma^2 + C^2.

        G = ||A_asym||/||P||, Gamma = ||A_grad||/||P|| (gradient/hierarchical),
        C = ||A_curl||/||P|| (curl/circulatory, the divergence-free residual).
        """
        den = math.sqrt(max(S_den, 0.0)) + EPSILON
        G = math.sqrt(max(S_asym, 0.0)) / den
        Gamma = math.sqrt(max(grad_energy, 0.0)) / den
        C = math.sqrt(max(S_asym - grad_energy, 0.0)) / den
        return G, Gamma, C

    def _incremental_reduce(self, Qh: torch.Tensor, Kh: torch.Tensor, L: int, prior: dict) -> dict:
        """Exact full-operator G/Gamma/C by folding only the delta tokens since the last fire.

        On causal P, the new rows [n_prev, L) only *add* to the operator (prior rows are
        unchanged), so the whole delta block is computed in ONE batched matmul + one
        causal-masked softmax (no per-token Python loop, no per-iteration GPU syncs):
            S_den  += ||P[t,:]||^2               (full new-row mass)
            S_asym += (sum_{i<t} P[t,i]^2)/2     (new antisymmetric edges, diagonal excluded)
        and the row-sum vector r = A_asym @ 1 updates as r_t += (1 - P[t,t])/2 (the new
        entry's below-mass) and r_i += -(sum_{t>i} P[t,i])/2 (the new column edges).
        Gamma uses the exact gradient energy 2||r||^2/L; C is the residual.  G/Gamma/C
        equal the batch full-operator values exactly.

        State cost: O(N) for the persisted r vector per (layer, head), growing with context
        (the exact-full guarantee requires the full sequence; it cannot be bounded like the
        Q-buffer without losing exactness).  The O(L^2)-total work assumes prior_state is
        threaded every fire; a dropped prior_state restarts at n_prev=0 (a one-off O(L^2)
        recompute that fire, still correct).
        """
        incr = (prior.get("partials") or {}).get("incr") if prior else None
        S_asym = incr["S_asym"] if incr else 0.0
        S_den = incr["S_den"] if incr else 0.0
        n_prev = incr["n"] if incr else 0
        # row-sum vector r = A_asym @ 1 (O(N) state, fp32), grows with context for Gamma/C
        if incr is not None and incr.get("r") is not None:
            r = incr["r"]
            if r.shape[0] < L:
                r = torch.cat([r, r.new_zeros(L - r.shape[0])])
        else:
            r = torch.zeros(L, device=Qh.device, dtype=torch.float32)

        if L > n_prev:
            lo = n_prev
            scale = 1.0 / math.sqrt(Qh.shape[1])
            scores = (Qh[lo:L] @ Kh[:L].T) * scale  # [B, L]
            grow = torch.arange(lo, L, device=Qh.device).unsqueeze(1)  # global row index
            gcol = torch.arange(L, device=Qh.device).unsqueeze(0)
            scores = scores.masked_fill(gcol > grow, float("-inf"))  # causal: keep cols 0..t
            P = torch.softmax(scores, dim=-1).to(torch.float32)  # [B, L], each row sums over 0..t
            B = P.shape[0]
            local = torch.arange(B, device=Qh.device)
            diag = P[local, grow.squeeze(1)]  # [B] self-attention P[t,t]
            S_den += float(P.square().sum().item())
            S_asym += float((P.square().sum() - diag.square().sum()).item()) / 2.0
            # new column edges: r[i] += -(sum over new rows t != i of P[t,i]) / 2
            P_nodiag = P.clone()
            P_nodiag[local, grow.squeeze(1)] = 0.0
            r[:L] += -P_nodiag.sum(dim=0) / 2.0
            # new entries: r[t] += (rowmass below t) / 2 = (1 - P[t,t]) / 2  (additive, not set)
            r[lo:L] += (P.sum(dim=1) - diag) / 2.0

        grad_energy = 2.0 * float((r * r).sum().item()) / L
        G, Gamma, C = self._split(S_asym, S_den, grad_energy)
        return {
            "features": AsymmetryFeatures(G=G, Gamma=Gamma, C=C),
            "tier": "incremental",
            "partials": {"incr": {"S_asym": S_asym, "S_den": S_den, "n": L, "r": r}},
        }

    def reduce(self, Qh: torch.Tensor, Kh: torch.Tensor, L: int, **ctx: Any) -> dict:
        # Drop any prior-fire cache: it is keyed on tensor data_ptr(), and the allocator
        # recycles addresses, so a stale entry from a freed tensor of the same shape could
        # otherwise be returned for new data.  Cleared here, repopulated by this fire's
        # _window_stats, and reused by witness() within the same fire.
        self._cache = None
        if self.incremental:
            return self._incremental_reduce(Qh, Kh, L, ctx.get("prior_state") or {})

        prior = ctx.get("prior_state") or {}
        prev = prior.get("partials") if self.streaming else None

        # Offset the probe seed per window so streaming estimation errors are
        # independent and average out across the accumulated stream.
        seed = self.seed + (prev.get("n_windows", 0) if prev else 0)
        S_asym, S_den, grad_energy, _, tier = self._window_stats(Qh, Kh, L, seed)

        if prev:
            # Block-diagonal streaming: all three energies are additive over disjoint windows
            # (the gradient energy of blockdiag(M1..Mk) is the sum of per-block energies).
            S_asym_tot = prev.get("S_asym", 0.0) + S_asym
            S_den_tot = prev.get("S_den", 0.0) + S_den
            S_grad_tot = prev.get("S_grad", 0.0) + grad_energy
            n_windows = prev.get("n_windows", 0) + 1
        else:
            S_asym_tot, S_den_tot, S_grad_tot, n_windows = S_asym, S_den, grad_energy, 1

        G, Gamma, C = self._split(S_asym_tot, S_den_tot, S_grad_tot)

        return {
            "features": AsymmetryFeatures(G=G, Gamma=Gamma, C=C),
            "tier": tier,
            "partials": {
                "S_asym": S_asym_tot,
                "S_den": S_den_tot,
                "S_grad": S_grad_tot,
                "n_windows": n_windows,
            },
        }

    def witness(self, Qh: torch.Tensor, Kh: torch.Tensor, L: int, **ctx: Any) -> torch.Tensor:
        # Per-window localization (base seed); shares reduce()'s pass via the cache in batch
        # mode.  witness[i] = ||P_asym[i,:]|| / ||P||_F, so in batch mode ||witness||_2 == G.
        _, S_den, _, row_sq, _ = self._window_stats(Qh, Kh, L, self.seed)
        row_norms = torch.sqrt(row_sq.clamp(min=0.0))
        return row_norms / (math.sqrt(max(S_den, 0.0)) + EPSILON)

    def accumulate(self, local: dict, state: dict | None) -> dict:
        # reduce() already folded prior_state into local["partials"] (streaming)
        # or produced window-only partials (batch); persist local either way.
        return local
