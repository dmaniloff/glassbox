"""Concrete Diagnostic implementations for each signal."""

from glassbox.diagnostics.asymmetry import AsymmetryDiagnostic
from glassbox.diagnostics.laplacian import LaplacianDiagnostic
from glassbox.diagnostics.routing import RoutingDiagnostic
from glassbox.diagnostics.selfattn import SelfAttnDiagnostic
from glassbox.diagnostics.spectral import SpectralDiagnostic
from glassbox.diagnostics.tracker import TrackerDiagnostic

DIAGNOSTIC_REGISTRY: dict[str, type] = {
    "spectral": SpectralDiagnostic,
    "routing": RoutingDiagnostic,
    "asymmetry": AsymmetryDiagnostic,
    "tracker": TrackerDiagnostic,
    "selfattn": SelfAttnDiagnostic,
    "laplacian": LaplacianDiagnostic,
}

__all__ = [
    "DIAGNOSTIC_REGISTRY",
    "SpectralDiagnostic",
    "RoutingDiagnostic",
    "AsymmetryDiagnostic",
    "TrackerDiagnostic",
    "SelfAttnDiagnostic",
    "LaplacianDiagnostic",
]
