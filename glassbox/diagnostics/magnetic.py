"""Magnetic-Laplacian frustration diagnostic: λ₁ of L_φ on the pre-softmax tournament.

**Operator: the UNMASKED pre-softmax scores S = QKᵀ.** The orientation / preference structure
lives in the antisymmetric part of S (`qᵢ·kⱼ` vs `qⱼ·kᵢ`) and survives causal masking; a causal
post-softmax matrix is triangular ⇒ its orientation is transitive ⇒ frustration is vacuous
(the same degeneracy that zeroes ``|T_cyc|``). The magnetic frustration is the spectral /
continuous partner of the cyclic-triangle count: the discrete member counts non-transitive
triangles, the magnetic member measures their spectral frustration. See docs/operator-choice.md.

Construction (formally verified in shade-formal ``MagneticFrustration.lean``):

    magnitude   W_ij = (|S_ij| + |S_ji|) / 2            (symmetric, ≥ 0; W_ii = 0)
    phase       θ_ij = arctan((S_ij − S_ji)/(S_ij + S_ji))   (θ_ij = 0 when S_ij + S_ji = 0)
    transport   (A_θ)_ij = W_ij · exp(i·θ_ij)           (Hermitian: θ antisymmetric)
    degree      D_ii = Σ_j W_ij                          (real)
    L_φ = D − A_θ                                         (Hermitian, PSD)

There is no charge parameter (effectively g = 1).  Frustration ``λ₁`` is the SMALLEST eigenvalue
of the Hermitian PSD ``L_φ`` (so λ₁ ≥ 0).  ``λ₁ = 0`` iff the phase is a pure gauge gradient
(``θ_ij = αᵢ − αⱼ``) — a balanced / curl-free orientation; ``λ₁ > 0`` signals non-transitive
(cyclic) preference loops that cannot be gauged away.  λ₁ is gauge-invariant, so degree
normalization (M vs the row-stochastic P) does not change it.  The bottom eigenvector's
per-token magnitudes are the witness (localization).
"""

from __future__ import annotations

import math
from typing import Any

import torch

from glassbox.results import MagneticFeatures
from glassbox.svd import hermitian_lanczos

EPSILON = 1e-10


class MagneticDiagnostic:
    signal_name = "magnetic"

    def __init__(self, threshold: int = 512, block_size: int = 256, incremental: bool = False):
        self.threshold = threshold
        self.block_size = block_size
        # incremental: report the streamable phase-curl frustration energy (Hodge curl of θ via
        # the row-sum identity, eigensolver-free) folded across fires, instead of the dense λ₁.
        self.incremental = incremental
        self._cache: tuple | None = None  # (key, (lambda1, bottom_evec, phase_curl))

    @staticmethod
    def _phase_curl(theta: torch.Tensor, L: int) -> float:
        """Hodge curl energy of the antisymmetric phase field θ: ‖θ‖² − 2‖θ·1‖²/L.

        Equals the total squared triangle holonomy Σ Φ_ijk²; 0 ⟺ θ is a pure gauge gradient
        ⟺ balanced ⟺ λ₁ = 0.  Streamable via the row-sum r_θ = θ·1 (issue #68).
        """
        r = theta.sum(dim=1)
        total = float((theta * theta).sum().item())
        grad_energy = 2.0 * float((r * r).sum().item()) / L
        return max(0.0, total - grad_energy)

    @staticmethod
    def _phase_and_magnitude(S: torch.Tensor, St: torch.Tensor) -> tuple:
        """(W, theta): symmetric magnitude and antisymmetric phase from scores S and S.T."""
        W = (S.abs() + St.abs()) / 2.0
        denom = S + St
        num = S - St
        safe = torch.where(denom != 0, denom, torch.ones_like(denom))
        theta = torch.where(denom != 0, torch.atan(num / safe), torch.zeros_like(denom))
        return W, theta

    def _materialized(self, Qh: torch.Tensor, Kh: torch.Tensor, L: int) -> tuple:
        """(lambda1, bottom_evec, phase_curl) via dense Hermitian eig.  Cached per fire."""
        key = (Qh.data_ptr(), Kh.data_ptr(), tuple(Qh.shape), tuple(Kh.shape), L)
        if self._cache is not None and self._cache[0] == key:
            return self._cache[1]
        scale = 1.0 / math.sqrt(Qh.shape[1])
        S = (Qh @ Kh.T * scale).to(torch.float32)
        W, theta = self._phase_and_magnitude(S, S.T)
        W = W - torch.diag(torch.diagonal(W))  # no self-loops: W_ii = 0
        A_theta = torch.complex(W * torch.cos(theta), W * torch.sin(theta))
        A_theta = A_theta - torch.diag(torch.diagonal(A_theta))  # zero diagonal phase term
        L_phi = torch.diag(W.sum(dim=1)).to(A_theta.dtype) - A_theta
        evals, evecs = torch.linalg.eigh(L_phi)  # ascending real evals, complex evecs
        val = (float(evals[0].item()), evecs[:, 0], self._phase_curl(theta, L))
        self._cache = (key, val)
        return val

    def _matvec_Lphi(self, Qh, Kh, L, scale, d):
        """Closure v ↦ L_φ @ v for complex v, blocked, without materializing L_φ."""

        def mv(v: torch.Tensor) -> torch.Tensor:
            out = d.to(v.dtype) * v  # D @ v
            for i0 in range(0, L, self.block_size):
                i1 = min(i0 + self.block_size, L)
                Sb = (Qh[i0:i1] @ Kh.T * scale).to(torch.float32)  # S[i, :]
                SbT = (Kh[i0:i1] @ Qh.T * scale).to(torch.float32)  # S[:, i].T = S[j, i]
                W, theta = self._phase_and_magnitude(Sb, SbT)
                A = torch.complex(W * torch.cos(theta), W * torch.sin(theta))  # [bs, L]
                local = torch.arange(i1 - i0, device=Qh.device)
                A[local, torch.arange(i0, i1, device=Qh.device)] = 0  # zero diagonal
                out[i0:i1] = out[i0:i1] - (A @ v)
            return out

        return mv

    def _degree(self, Qh, Kh, L, scale) -> torch.Tensor:
        """Real degree vector d_i = Σ_j W_ij, blocked."""
        d = torch.zeros(L, device=Qh.device, dtype=torch.float32)
        for i0 in range(0, L, self.block_size):
            i1 = min(i0 + self.block_size, L)
            Sb = (Qh[i0:i1] @ Kh.T * scale).to(torch.float32)
            SbT = (Kh[i0:i1] @ Qh.T * scale).to(torch.float32)
            W = (Sb.abs() + SbT.abs()) / 2.0
            local = torch.arange(i1 - i0, device=Qh.device)
            W[local, torch.arange(i0, i1, device=Qh.device)] = 0  # W_ii = 0
            d[i0:i1] = W.sum(dim=1)
        return d

    def _phase_curl_stream(self, Qh, Kh, L, n_prev, r, theta_sq):
        """Fold tokens [n_prev, L) into the phase row-sum r_θ and ‖θ‖² (the curl-energy stats).

        New token t adds edges (t, j) for j < t; θ is antisymmetric (θ_jt = −θ_tj). O(L) state,
        O((L−n_prev)·L) work. Mirrors the asymmetry incremental fold, applied to the phase θ.
        """
        if L <= n_prev:
            return r, theta_sq
        scale = 1.0 / math.sqrt(Qh.shape[1])
        lo = n_prev
        Sb = (Qh[lo:L] @ Kh.T * scale).to(torch.float32)  # S[t, j]
        SbT = (Kh[lo:L] @ Qh.T * scale).to(torch.float32)  # S[j, t]
        _, theta = self._phase_and_magnitude(Sb, SbT)  # θ[t, j], [B, L]
        rows = torch.arange(lo, L, device=Qh.device).unsqueeze(1)
        cols = torch.arange(L, device=Qh.device).unsqueeze(0)
        th = torch.where(cols < rows, theta, torch.zeros_like(theta))  # keep new pairs j < t
        theta_sq = theta_sq + 2.0 * float((th * th).sum().item())  # both (t,j) and (j,t)
        r = r - th.sum(dim=0)  # new columns: r[j] += θ_jt = −θ_tj
        r[lo:L] = r[lo:L] + th.sum(dim=1)  # new entries: r[t] += Σ_{j<t} θ_tj
        return r, theta_sq

    def _matrix_free(self, Qh, Kh, L) -> tuple:
        """(lambda1, bottom_evec, phase_curl) via complex-Hermitian Lanczos (which='smallest')."""
        scale = 1.0 / math.sqrt(Qh.shape[1])
        d = self._degree(Qh, Kh, L, scale)
        mv = self._matvec_Lphi(Qh, Kh, L, scale, d)
        iters = min(L, max(20, 4))
        evals, evecs = hermitian_lanczos(
            mv, L, k=1, iters=iters, device=str(Qh.device), which="smallest", dtype=torch.complex64
        )
        r, theta_sq = self._phase_curl_stream(
            Qh, Kh, L, 0, torch.zeros(L, device=Qh.device, dtype=torch.float32), 0.0
        )
        pc = max(0.0, theta_sq - 2.0 * float((r * r).sum().item()) / L)
        return float(evals[0].item()), evecs[:, 0], pc

    def reduce(self, Qh: torch.Tensor, Kh: torch.Tensor, L: int, **ctx: Any) -> dict:
        self._cache = None
        if self.incremental:
            # Streamable phase-curl frustration (eigensolver-free); λ₁ left None. See issue #68.
            prior = ctx.get("prior_state") or {}
            st = (prior.get("partials") or {}).get("mag")
            if st is not None:
                r = st["r"]
                if r.shape[0] < L:
                    r = torch.cat([r, r.new_zeros(L - r.shape[0])])
                theta_sq, n_prev = st["theta_sq"], st["n"]
            else:
                r = torch.zeros(L, device=Qh.device, dtype=torch.float32)
                theta_sq, n_prev = 0.0, 0
            r, theta_sq = self._phase_curl_stream(Qh, Kh, L, n_prev, r, theta_sq)
            pc = max(0.0, theta_sq - 2.0 * float((r * r).sum().item()) / L)
            return {
                "features": MagneticFeatures(frustration=None, phase_curl=pc),
                "tier": "incremental",
                "partials": {"mag": {"r": r, "theta_sq": theta_sq, "n": L}},
            }
        if L <= self.threshold:
            lambda1, _, pc = self._materialized(Qh, Kh, L)
            tier = "materialized"
        else:
            lambda1, _, pc = self._matrix_free(Qh, Kh, L)
            tier = "matrix_free"
        # L_φ is PSD, so λ₁ ≥ 0; clamp away float-eig noise (e.g. ~-1e-7 on a balanced operator).
        return {
            "features": MagneticFeatures(frustration=max(0.0, lambda1), phase_curl=pc),
            "tier": tier,
        }

    def witness(self, Qh: torch.Tensor, Kh: torch.Tensor, L: int, **ctx: Any) -> torch.Tensor:
        if L <= self.threshold:
            _, evec, _ = self._materialized(Qh, Kh, L)
        else:
            _, evec, _ = self._matrix_free(Qh, Kh, L)
        return evec.abs()  # per-token frustration localization (bottom eigenvector magnitude)

    def accumulate(self, local: dict, state: dict | None) -> dict:
        return local
