"""Concrete Diagnostic implementations for each signal."""

from glassbox.diagnostics.asymmetry import AsymmetryDiagnostic
from glassbox.diagnostics.cyclic_triangles import CyclicTrianglesDiagnostic
from glassbox.diagnostics.laplacian import LaplacianDiagnostic
from glassbox.diagnostics.magnetic import MagneticDiagnostic
from glassbox.diagnostics.routing import RoutingDiagnostic
from glassbox.diagnostics.selfattn import SelfAttnDiagnostic
from glassbox.diagnostics.spectral import SpectralDiagnostic
from glassbox.diagnostics.tracker import TrackerDiagnostic

# This package is where a signal's three pieces meet: its config (glassbox.config), the
# Features model it emits (glassbox.results), and the code that computes it.  The registries
# below are keyed off each class's own ``signal_name`` rather than a hand-typed string, so
# the keys cannot disagree with the classes they point at.  The tuple is the import list
# above -- which has to exist regardless -- not a second list to keep in sync.
#
# The dependency runs config <- diagnostics, never the reverse: glassbox.config stays free
# of torch (~0.2s to import, vs ~1.5s for this package), which matters because the CLI and
# the vLLM plugin both import GlassboxConfig.
_DIAGNOSTICS: tuple[type, ...] = (
    SpectralDiagnostic,
    RoutingDiagnostic,
    AsymmetryDiagnostic,
    CyclicTrianglesDiagnostic,
    MagneticDiagnostic,
    TrackerDiagnostic,
    SelfAttnDiagnostic,
    LaplacianDiagnostic,
)

DIAGNOSTIC_REGISTRY: dict[str, type] = {d.signal_name: d for d in _DIAGNOSTICS}

#: Signal name -> the Features model it emits.  Lets a caller ask what columns a signal
#: produces without running it; previously this pairing was written out by hand in
#: cli/extract.py, and the three signals added after it was written were never added to it.
FEATURES_MODELS: dict[str, type] = {d.signal_name: d.features_model for d in _DIAGNOSTICS}

__all__ = [
    "DIAGNOSTIC_REGISTRY",
    "FEATURES_MODELS",
    "SpectralDiagnostic",
    "RoutingDiagnostic",
    "AsymmetryDiagnostic",
    "CyclicTrianglesDiagnostic",
    "MagneticDiagnostic",
    "TrackerDiagnostic",
    "SelfAttnDiagnostic",
    "LaplacianDiagnostic",
]
