"""Concrete Diagnostic implementations for each signal."""

from glassbox.diagnostics.asymmetry import AsymmetryDiagnostic
from glassbox.diagnostics.cyclic_triangles import CyclicTrianglesDiagnostic
from glassbox.diagnostics.laplacian import LaplacianDiagnostic
from glassbox.diagnostics.magnetic import MagneticDiagnostic
from glassbox.diagnostics.routing import RoutingDiagnostic
from glassbox.diagnostics.selfattn import SelfAttnDiagnostic
from glassbox.diagnostics.spectral import SpectralDiagnostic
from glassbox.diagnostics.tracker import TrackerDiagnostic

DIAGNOSTIC_REGISTRY: dict[str, type] = {
    "spectral": SpectralDiagnostic,
    "routing": RoutingDiagnostic,
    "asymmetry": AsymmetryDiagnostic,
    "cyclic": CyclicTrianglesDiagnostic,
    "magnetic": MagneticDiagnostic,
    "tracker": TrackerDiagnostic,
    "selfattn": SelfAttnDiagnostic,
    "laplacian": LaplacianDiagnostic,
}

__all__ = [
    "DIAGNOSTIC_REGISTRY",
    "SpectralDiagnostic",
    "RoutingDiagnostic",
    "AsymmetryDiagnostic",
    "CyclicTrianglesDiagnostic",
    "MagneticDiagnostic",
    "TrackerDiagnostic",
    "SelfAttnDiagnostic",
    "LaplacianDiagnostic",
]
