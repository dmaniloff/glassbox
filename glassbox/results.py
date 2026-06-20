"""Pydantic models for SVD result emission and JSONL parsing.

Provides a single source of truth for derived spectral features
(sv1, sv_ratio, sv_entropy) and structured result types.
"""

from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SPECTRAL_FEATURE_NAMES = ["sv_ratio", "sv1", "sv_entropy"]


def _spectral_from_svs(svs: list[float]) -> dict[str, float | None]:
    """Compute spectral features from a list of singular values."""
    if not svs:
        return {"sv1": None, "sv_ratio": None, "sv_entropy": None}
    sv1 = svs[0]
    sv_ratio = svs[0] / svs[1] if len(svs) >= 2 and svs[1] > 0 else None
    total = sum(svs)
    if total > 0:
        ps = [s / total for s in svs]
        sv_entropy = -sum(p * math.log(p + 1e-12) for p in ps)
    else:
        sv_entropy = None
    return {"sv1": sv1, "sv_ratio": sv_ratio, "sv_entropy": sv_entropy}


class SpectralFeatures(BaseModel):
    """Features from SVD of pre-softmax scores matrix S = QK^T.

    Single source of truth: singular_values are stored directly, and
    spectral features (sv1, sv_ratio, sv_entropy) are derived from them
    automatically on construction.
    """

    model_config = ConfigDict(frozen=True)

    # Raw singular values
    singular_values: list[float] = Field(description="Singular values of S (descending).")

    # Spectral (derived from singular_values)
    sv1: float | None = Field(None, description="Leading singular value of S.")
    sv_ratio: float | None = Field(None, description="sigma1/sigma2 ratio.")
    sv_entropy: float | None = Field(
        None, description="Entropy of normalized singular value distribution."
    )

    @model_validator(mode="before")
    @classmethod
    def _derive_spectral(cls, values: dict) -> dict:
        if isinstance(values, dict) and "singular_values" in values:
            for k, v in _spectral_from_svs(values["singular_values"]).items():
                values.setdefault(k, v)
        return values


class RoutingFeatures(BaseModel):
    """Features from SVD + Hodge decomposition of degree-normalized M.

    Single source of truth: singular_values are stored directly, and
    spectral features (sv1, sv_ratio, sv_entropy) are derived from them
    automatically on construction.
    """

    model_config = ConfigDict(frozen=True)

    # Raw singular values
    singular_values: list[float] = Field(description="Singular values of M (descending).")

    # Spectral (derived from singular_values)
    sv1: float | None = Field(None, description="Leading singular value of M.")
    sv_ratio: float | None = Field(None, description="sigma1/sigma2 ratio.")
    sv_entropy: float | None = Field(
        None, description="Entropy of normalized singular value distribution."
    )

    # Hodge decomposition
    phi_hat: float | None = Field(
        None,
        description="Cheeger conductance via bipartite sweep cut.",
    )
    sigma2: float | None = Field(None, description="Second singular value of M.")
    G: float | None = Field(None, description="Total asymmetry: ||M_asym||_F / ||M||_F.")
    Gamma: float | None = Field(
        None, description="Gradient coefficient: potential-driven portion of asymmetry."
    )
    C: float | None = Field(
        None,
        description="Curl coefficient: circulatory portion of asymmetry (triangle-sampled).",
    )
    curl_ratio: float | None = Field(
        None, description="C / (G + eps). Share of asymmetry that is circulatory."
    )
    sigma2_asym: float | None = Field(None, description="Second singular value of M_asym.")
    commutator_norm: float | None = Field(
        None,
        description=(
            "||[M_sym, M_asym]||_F / ||M||_F. Entanglement of symmetric and antisymmetric parts."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _derive_spectral(cls, values: dict) -> dict:
        if isinstance(values, dict) and "singular_values" in values:
            for k, v in _spectral_from_svs(values["singular_values"]).items():
                values.setdefault(k, v)
        return values

    @field_validator(
        "G",
        "Gamma",
        "C",
        "curl_ratio",
        "sigma2",
        "sigma2_asym",
        "commutator_norm",
        "phi_hat",
        "sv1",
        "sv_ratio",
        "sv_entropy",
    )
    @classmethod
    def _scrub_nonfinite(cls, v: float | None) -> float | None:
        # NaN/inf must never reach the sink: the upstream max(x, 0.0) guards do NOT scrub NaN
        # (max(nan, 0.0) == nan), so a poisoned M would otherwise emit garbage silently.
        return v if v is None or math.isfinite(v) else None


class CyclicTrianglesFeatures(BaseModel):
    """Cyclic-triangle count |T_cyc| of the pre-softmax sign tournament ω(QKᵀ).

    Counts non-transitive (3-cycle) attention triangles — token i prefers j, j prefers k,
    k prefers i — on the unmasked pre-softmax scores (NOT post-softmax: a causal post-softmax
    tournament is transitive ⇒ |T_cyc| = 0). The per-token triangle-participation profile is
    emitted as the snapshot witness. See docs/operator-choice.md.
    """

    model_config = ConfigDict(frozen=True)

    T_cyc: int | None = Field(None, description="Cyclic-triangle count of ω(QKᵀ).")


class MagneticFeatures(BaseModel):
    """Magnetic-Laplacian frustration λ₁ of the pre-softmax tournament ω(QKᵀ).

    λ₁ = smallest eigenvalue of the Hermitian magnetic Laplacian L_φ = D − A⊙e^{iθ} on the
    unmasked pre-softmax scores (NOT post-softmax: a causal tournament is transitive ⇒ λ₁ = 0).
    λ₁ = 0 is a balanced / curl-free orientation; λ₁ > 0 is frustration (cyclic preference
    loops). The bottom-eigenvector per-token magnitudes are emitted as the witness.
    See docs/operator-choice.md.
    """

    model_config = ConfigDict(frozen=True)

    frustration: float | None = Field(None, description="λ₁ of the magnetic Laplacian L_φ.")

    @field_validator("frustration")
    @classmethod
    def _scrub_nonfinite(cls, v: float | None) -> float | None:
        return v if v is None or math.isfinite(v) else None


class AsymmetryFeatures(BaseModel):
    """Asymmetry coefficient G = ||P_asym||_F / ||P||_F of row-stochastic attention P,
    with its Hodge gradient/curl split G^2 = Gamma^2 + C^2.

    Hodge G signal.  Computed on the post-softmax attention P (NOT the degree-normalized
    M — see docs/operator-choice.md).  G is estimated matrix-free via a direct Hutchinson
    estimator on ||P_asym z||^2 (Route B) above the threshold, exactly below it.  The
    gradient energy is exact via the row-sum identity ||A_grad||^2 = 2||r||^2/L,
    r = A_asym @ 1 (the hierarchical / potential part); the curl C is the divergence-free
    residual (circulatory part).  The per-token asymmetry profile is the snapshot witness.
    """

    model_config = ConfigDict(frozen=True)

    G: float | None = Field(None, description="Total asymmetry: ||P_asym||_F / ||P||_F.")
    Gamma: float | None = Field(
        None, description="Gradient (hierarchical) part: ||A_grad||_F / ||P||_F."
    )
    C: float | None = Field(
        None, description="Curl (circulatory) part: ||A_curl||_F / ||P||_F; G^2 = Gamma^2 + C^2."
    )


class TrackerFeatures(BaseModel):
    """Features from raw post-softmax attention matrix A.

    Based on AttentionTracker (arXiv:2411.00348), span-independent features.
    Singular values come from A directly (not degree-normalized).
    Spectral features (sv1, sv_ratio, sv_entropy) are derived from
    singular_values automatically on construction.
    """

    model_config = ConfigDict(frozen=True)

    # Raw singular values of A
    singular_values: list[float] = Field(description="Singular values of A (descending).")

    # Spectral (derived from singular_values)
    sv1: float | None = Field(None, description="Leading singular value of A.")
    sv_ratio: float | None = Field(None, description="sigma1/sigma2 ratio.")
    sv_entropy: float | None = Field(
        None, description="Entropy of normalized singular value distribution."
    )

    # AttentionTracker features
    sigma2: float | None = Field(None, description="Second singular value of A.")
    sigma2_asym: float | None = Field(
        None, description="Second singular value of A_asym = (A - A^T) / 2."
    )
    commutator_norm: float | None = Field(
        None,
        description=(
            "||[A_sym, A_asym]||_F / ||A||_F. Coupling of symmetric and antisymmetric parts."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _derive_spectral(cls, values: dict) -> dict:
        if isinstance(values, dict) and "singular_values" in values:
            for k, v in _spectral_from_svs(values["singular_values"]).items():
                values.setdefault(k, v)
        return values


class SelfAttnFeatures(BaseModel):
    """Features from the attention matrix diagonal.

    - ``attn_diag_logmean``: mean log self-attention weight mean_i(log(A[i,i]))
      from LLM-Check (NeurIPS 2024).
    - ``eigvals``: top-k diagonal values of A, sorted descending.  For causal
      (lower-triangular) attention these are the eigenvalues of A, used as a
      baseline in LapEigvals (EMNLP 2025, arXiv:2502.17598).

    No SVD involved — direct scalar/vector statistics from diag(A).
    """

    model_config = ConfigDict(frozen=True)

    attn_diag_logmean: float = Field(
        description="Mean of log diagonal of A. Higher = stronger self-attention."
    )
    eigvals: list[float] = Field(
        default=[],
        description=(
            "Top-k diagonal values of A, sorted descending (attention eigenvalues for causal A)."
        ),
    )


class LaplacianFeatures(BaseModel):
    """Laplacian eigenvalue features from attention graphs.

    Treats the attention matrix A as a weighted directed graph and computes
    the in-degree graph Laplacian L = D_in - A, where D_in[i,i] = sum_j A[j,i]
    (column sums of A).  For causal (lower-triangular) attention the diagonal
    entries of L are its eigenvalues.

    Reference:
        Binkowski et al., "Hallucination Detection in LLMs Using Spectral
        Features of Attention Maps", EMNLP 2025 (arXiv:2502.17598).
    """

    model_config = ConfigDict(frozen=True)

    eigvals: list[float] = Field(description="Top-k Laplacian diagonal values, sorted descending.")


class SVDSnapshot(BaseModel):
    """One SVD observation emitted per (request, layer, head, step)."""

    model_config = ConfigDict(frozen=True)

    # "spectral" | "routing" | "tracker" | "selfattn" | "laplacian"
    signal: str
    request_id: int
    layer: str
    layer_idx: int | None
    head: int
    step: int
    L: int
    singular_values: list[float] = []
    tier: str | None = None  # "materialized" | "matrix_free"
    witness: list[float] | None = None
    features: (
        SpectralFeatures
        | RoutingFeatures
        | AsymmetryFeatures
        | CyclicTrianglesFeatures
        | MagneticFeatures
        | TrackerFeatures
        | SelfAttnFeatures
        | LaplacianFeatures
    )

    def __repr__(self) -> str:
        feat_dict = self.features.model_dump(exclude_none=True)
        feat_str = " ".join(f"{k}={v}" for k, v in feat_dict.items() if k != "singular_values")
        parts = [
            f"[{self.signal}]",
            self.layer,
            f"head={self.head}",
            f"step={self.step}",
            f"L={self.L}",
        ]
        if self.tier is not None:
            parts.append(f"tier={self.tier}")
        if feat_str:
            parts.append(feat_str)
        return " ".join(parts)

    @classmethod
    def from_jsonl_row(cls, raw: dict) -> SVDSnapshot:
        """Deserialize a JSONL row, discriminating features by signal."""
        d = dict(raw)
        sig = d["signal"]
        feat_raw = d["features"]
        if isinstance(feat_raw, dict):
            if sig == "routing":
                d["features"] = RoutingFeatures(**feat_raw)
            elif sig == "asymmetry":
                d["features"] = AsymmetryFeatures(**feat_raw)
            elif sig == "cyclic":
                d["features"] = CyclicTrianglesFeatures(**feat_raw)
            elif sig == "magnetic":
                d["features"] = MagneticFeatures(**feat_raw)
            elif sig == "tracker":
                d["features"] = TrackerFeatures(**feat_raw)
            elif sig == "selfattn":
                d["features"] = SelfAttnFeatures(**feat_raw)
            elif sig == "laplacian":
                d["features"] = LaplacianFeatures(**feat_raw)
            else:
                d["features"] = SpectralFeatures(**feat_raw)
        return cls(**d)
