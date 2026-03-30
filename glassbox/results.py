"""Pydantic models for SVD result emission and JSONL parsing.

Provides a single source of truth for derived spectral features
(sv1, sv_ratio, sv_entropy) and structured result types.
"""

from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict, Field

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
    spectral features (sv1, sv_ratio, sv_entropy) are derived from them.
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

    @classmethod
    def from_singular_values(cls, singular_values: list[float]) -> SpectralFeatures:
        """Build from singular values, deriving spectral features."""
        kwargs = _spectral_from_svs(singular_values)
        kwargs["singular_values"] = singular_values
        return cls(**kwargs)


class RoutingFeatures(BaseModel):
    """Features from SVD + Hodge decomposition of degree-normalized M.

    Single source of truth: singular_values are stored directly, and
    spectral features (sv1, sv_ratio, sv_entropy) are derived from them.
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
        description="Conductance (1 - sigma2). High = bottlenecked through one mode.",
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

    @classmethod
    def from_hodge(
        cls,
        singular_values: list[float],
        **hodge_features,
    ) -> RoutingFeatures:
        """Build from singular values and Hodge decomposition features.

        Spectral features (sv1, sv_ratio, sv_entropy) are derived from
        singular_values automatically.
        """
        kwargs = _spectral_from_svs(singular_values)
        kwargs["singular_values"] = singular_values
        kwargs.update(hodge_features)
        return cls(**kwargs)


class TrackerFeatures(BaseModel):
    """Features from raw post-softmax attention matrix A.

    Based on AttentionTracker (arXiv:2411.00348), span-independent features.
    Singular values come from A directly (not degree-normalized).
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

    @classmethod
    def from_attention_tracker(
        cls,
        singular_values: list[float],
        **tracker_features,
    ) -> TrackerFeatures:
        """Build from singular values and tracker features."""
        kwargs = _spectral_from_svs(singular_values)
        kwargs["singular_values"] = singular_values
        kwargs.update(tracker_features)
        return cls(**kwargs)


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
    features: (
        SpectralFeatures
        | RoutingFeatures
        | TrackerFeatures
        | SelfAttnFeatures
        | LaplacianFeatures
    )

    @classmethod
    def from_jsonl_row(cls, raw: dict) -> SVDSnapshot:
        """Deserialize a JSONL row, discriminating features by signal."""
        d = dict(raw)
        sig = d["signal"]
        feat_raw = d["features"]
        if isinstance(feat_raw, dict):
            if sig == "routing":
                d["features"] = RoutingFeatures(**feat_raw)
            elif sig == "tracker":
                d["features"] = TrackerFeatures(**feat_raw)
            elif sig == "selfattn":
                d["features"] = SelfAttnFeatures(**feat_raw)
            elif sig == "laplacian":
                d["features"] = LaplacianFeatures(**feat_raw)
            else:
                d["features"] = SpectralFeatures(**feat_raw)
        return cls(**d)
