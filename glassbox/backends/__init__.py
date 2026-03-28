"""
Custom vLLM attention backends.

The primary backend is ``SVDTritonAttentionBackend``, which wraps
vLLM's Triton attention to extract signals from attention matrices
at configurable intervals during inference.
"""

from glassbox.backends.svd_backend import SVDTritonAttentionBackend

__all__ = ["SVDTritonAttentionBackend"]
