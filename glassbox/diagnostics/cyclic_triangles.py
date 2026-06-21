"""Cyclic-triangle diagnostic: |T_cyc| on the pre-softmax sign tournament.

**Operator: the UNMASKED pre-softmax score matrix S = QKᵀ.** The `/√d` is irrelevant — the
tournament depends only on the *sign* of `Sᵢⱼ − Sⱼᵢ`, which is scale-invariant — so we work
with `Δ = S − Sᵀ` directly. NOT the post-softmax attention: a causal post-softmax matrix is
lower-triangular, so its sign tournament is transitive ⇒ `|T_cyc| = 0` identically (the
diagnostic is *vacuous* on masked attention). The orientation/preference structure |T_cyc|
measures lives in the raw `qᵢ·kⱼ` vs `qⱼ·kᵢ` comparison, which survives causal masking.
This is the spectral-companion's discrete partner (see the magnetic-Laplacian diagnostic) —
both the orientation family, on pre-softmax S. See `docs/operator-choice.md`.

Sign tournament: `ω(S)ᵢⱼ = +1` iff `Δᵢⱼ > 0`, or (`Δᵢⱼ = 0` and `i < j`), with
`Δᵢⱼ = qᵢ·kⱼ − qⱼ·kᵢ`. Ties are broken by index order, making `ω` a *complete* tournament;
the index tie-break also freezes prior edges as the stream grows (additivity).

A triple `{i,j,k}` is cyclic iff its three edges form a directed 3-cycle. The count has the
Kendall closed form `|T_cyc| = C(n,3) − Σᵢ C(sᵢ, 2)`, `sᵢ` = out-degree (score) of vertex i
— so it needs only the **O(n) out-degree vector**, not the O(n²) signs. Streaming: when token
t arrives, `|T_cyc| += C(t,2) − C(s_t,2) − Σ_{j beats t} sⱼ` (exact, O(t)/token; prior edges
never flip). The per-token **arrival increment is the witness** (`Σᵢ witnessᵢ = |T_cyc|`).

Theory: streaming-cyclic-triangles (Kendall identity, O(ΔE) streaming update, memory bound).
Application + operator: structural-streaming-attention.
"""

from __future__ import annotations

from typing import Any

import torch

from glassbox.results import CyclicTrianglesFeatures


class CyclicTrianglesDiagnostic:
    signal_name = "cyclic"

    def __init__(self, incremental: bool = False):
        # incremental: maintain the out-degree vector + running count across fires and fold
        # only the delta tokens per fire (the O(ΔE) streaming update); requires the unbounded
        # full-sequence buffer. Else each fire counts the current window from scratch (both
        # exact — there is no estimation, only integer counting).
        self.incremental = incremental
        self._cache: tuple | None = None  # (key, (s, D))

    def _materialized(self, Qh: torch.Tensor, Kh: torch.Tensor, L: int) -> tuple:
        """(s, beats): out-degree vector s [L] (int64) and the tournament ω [L, L] (bool).

        Cleared per reduce() (data_ptr key) so reduce() then witness() share one pass within
        a fire without risking a stale entry from a recycled tensor address.
        """
        key = (Qh.data_ptr(), Kh.data_ptr(), tuple(Qh.shape), tuple(Kh.shape), L)
        if self._cache is not None and self._cache[0] == key:
            return self._cache[1]
        S = Qh @ Kh.T
        D = S - S.T  # Δ_ij = q_i·k_j − q_j·k_i (antisymmetric)
        idx = torch.arange(L, device=D.device)
        # ω[i,j] = i beats j (i ≠ j): Δ_ij > 0, or (Δ_ij == 0 and i < j) — ties by index.
        beats = (D > 0) | ((D == 0) & (idx.unsqueeze(1) < idx.unsqueeze(0)))
        beats.fill_diagonal_(False)
        s = beats.sum(dim=1).to(torch.int64)  # out-degree (score) of each vertex
        val = (s, beats)
        self._cache = (key, val)
        return val

    @staticmethod
    def _kendall(s: torch.Tensor, L: int) -> int:
        """|T_cyc| = C(L,3) − Σ_i C(s_i, 2), exact integer count."""
        c_n3 = L * (L - 1) * (L - 2) // 6
        sum_c2 = int((s * (s - 1) // 2).sum().item())
        return c_n3 - sum_c2

    def _witness(self, beats: torch.Tensor, L: int) -> torch.Tensor:
        """witness[t] = #cyclic triangles {i,j,t} closed at token t (i, j < t); Σ = |T_cyc|.

        Fully vectorized via a column-cumsum of the tournament ω — no Python loop, exact
        integer arithmetic. For token t (priors 0..t-1):
            s_arr[t]    = Σ_{j<t} ω[t,j]           (out-degree of t among priors)
            prefix[j,t] = Σ_{m<t} ω[j,m]           (out-degree of j restricted to 0..t-1)
            sigma[t]    = Σ_{j<t, j beats t} prefix[j,t]
            witness[t]  = C(t,2) − C(s_arr[t],2) − sigma[t]   (the per-arrival increment)
        """
        device = beats.device
        omega = beats.to(torch.int64)
        ii = torch.arange(L, device=device)
        lower = (ii.unsqueeze(1) > ii.unsqueeze(0)).to(torch.int64)  # [t,j] = (j < t)
        upper = (ii.unsqueeze(1) < ii.unsqueeze(0)).to(torch.int64)  # [j,t] = (j < t)
        s_arr = (omega * lower).sum(dim=1)  # [L] out-degree of t among priors
        prefix = torch.cumsum(omega, dim=1) - omega  # [L,L] exclusive prefix out-degrees
        sigma = (omega * prefix * upper).sum(dim=0)  # [L]
        c_t2 = ii * (ii - 1) // 2
        c_s2 = s_arr * (s_arr - 1) // 2
        return (c_t2 - c_s2 - sigma).to(torch.float32)

    def _stream(self, Qh, Kh, s: torch.Tensor, count: int, n_prev: int, L: int) -> tuple:
        """Fold tokens [n_prev, L) into the out-degree vector s and running count (exact)."""
        for t in range(max(n_prev, 1), L):
            d_jt = (Qh[:t] @ Kh[t]) - (Qh[t] @ Kh[:t].T)  # Δ_jt = q_j·k_t − q_t·k_j, j < t
            beats_t = (d_jt > 0) | (d_jt == 0)  # j beats t (ties → j, since j < t)
            n_in = int(beats_t.sum().item())
            s_t = t - n_in
            sigma = int(s[:t][beats_t].sum().item())
            count += t * (t - 1) // 2 - s_t * (s_t - 1) // 2 - sigma
            s[:t] = torch.where(beats_t, s[:t] + 1, s[:t])
            s[t] = s_t
        return s, count

    def reduce(self, Qh: torch.Tensor, Kh: torch.Tensor, L: int, **ctx: Any) -> dict:
        self._cache = None  # drop any prior-fire cache (data_ptr addresses get recycled)
        if self.incremental:
            prior = ctx.get("prior_state") or {}
            st = (prior.get("partials") or {}).get("tcyc")
            if st is not None:
                # warm: fold only the delta tokens [n_prev, L) (the O(ΔE) streaming update;
                # one token per fire under interval=1 → the 1-token global-streaming mode).
                s = st["s"]
                if s.shape[0] < L:
                    s = torch.cat([s, s.new_zeros(L - s.shape[0])])
                s, count = self._stream(Qh, Kh, s, st["count"], st["n"], L)
            else:
                # cold start (e.g. a prefill of L tokens at once): initialize the out-degree
                # vector + count with the vectorized batch path, not an O(L) Python loop.
                s, _ = self._materialized(Qh, Kh, L)
                s = s.clone()
                count = self._kendall(s, L)
            return {
                "features": CyclicTrianglesFeatures(T_cyc=count),
                "tier": "incremental",
                "partials": {"tcyc": {"s": s, "count": count, "n": L}},
            }

        s, _ = self._materialized(Qh, Kh, L)
        return {
            "features": CyclicTrianglesFeatures(T_cyc=self._kendall(s, L)),
            "tier": "materialized",
        }

    def witness(self, Qh: torch.Tensor, Kh: torch.Tensor, L: int, **ctx: Any) -> torch.Tensor:
        _, beats = self._materialized(Qh, Kh, L)
        return self._witness(beats, L)

    def accumulate(self, local: dict, state: dict | None) -> dict:
        # reduce() already folded prior_state into local["partials"] (incremental) or produced
        # a window-only count (batch); persist local either way.
        return local
