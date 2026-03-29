"""
Glassbox - Observability for transformer attention internals on vLLM.

Extracts spectral, topological, and statistical signals from attention
matrices during inference via a custom vLLM attention backend plugin.
No vLLM source modifications required.

The plugin is registered automatically via the ``vllm.general_plugins``
entry point when the package is installed.  Launch vLLM with
``--attention-backend CUSTOM --enforce-eager`` to activate it.
"""

from glassbox.config import GlassboxConfig
from glassbox.results import (
    AttentionDiagonalFeatures,
    AttentionTrackerFeatures,
    DegreeNormalizedFeatures,
    LaplacianEigvalsFeatures,
    ScoresMatrixFeatures,
    SVDSnapshot,
)

__all__ = [
    "GlassboxConfig",
    "SVDSnapshot",
    "ScoresMatrixFeatures",
    "DegreeNormalizedFeatures",
    "AttentionTrackerFeatures",
    "AttentionDiagonalFeatures",
    "LaplacianEigvalsFeatures",
]
