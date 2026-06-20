"""Concrete Diagnostic implementations for each signal."""

from glassbox.diagnostics.laplacian import LaplacianDiagnostic
from glassbox.diagnostics.routing import RoutingDiagnostic
from glassbox.diagnostics.selfattn import SelfAttnDiagnostic
from glassbox.diagnostics.spectral import SpectralDiagnostic
from glassbox.diagnostics.tracker import TrackerDiagnostic

DIAGNOSTIC_REGISTRY: dict[str, type] = {
    "spectral": SpectralDiagnostic,
    "routing": RoutingDiagnostic,
    "tracker": TrackerDiagnostic,
    "selfattn": SelfAttnDiagnostic,
    "laplacian": LaplacianDiagnostic,
}

__all__ = [
    "DIAGNOSTIC_REGISTRY",
    "SpectralDiagnostic",
    "RoutingDiagnostic",
    "TrackerDiagnostic",
    "SelfAttnDiagnostic",
    "LaplacianDiagnostic",
]
