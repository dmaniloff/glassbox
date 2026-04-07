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
from glassbox.handlers import (
    JsonlHandler,
    LoggingHandler,
    OtelHandler,
    SnapshotHandler,
)
from glassbox.results import (
    LaplacianFeatures,
    RoutingFeatures,
    SelfAttnFeatures,
    SpectralFeatures,
    SVDSnapshot,
    TrackerFeatures,
)

__all__ = [
    "GlassboxConfig",
    "SVDSnapshot",
    "SpectralFeatures",
    "RoutingFeatures",
    "TrackerFeatures",
    "SelfAttnFeatures",
    "LaplacianFeatures",
    "SnapshotHandler",
    "JsonlHandler",
    "LoggingHandler",
    "OtelHandler",
]
