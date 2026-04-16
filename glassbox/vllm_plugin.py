"""vLLM general plugin that registers the glassbox SVD attention backend.

This is loaded automatically by vLLM in all processes (API server, engine core,
workers) via the ``vllm.general_plugins`` entry point defined in pyproject.toml.

When using ``vllm serve --attention-backend CUSTOM``, put a ``glassbox.yaml``
in the working directory to configure signals, output, and OTel emission.
The plugin calls ``set_config()`` so that handlers are initialised even
without the ``glassbox-run`` CLI wrapper.
"""


def register_svd_backend():
    from vllm.v1.attention.backends.registry import (
        AttentionBackendEnum,
        register_backend,
    )

    register_backend(
        AttentionBackendEnum.CUSTOM,
        "glassbox.backends.svd_backend.SVDTritonAttentionBackend",
    )

    # Initialise config from glassbox.yaml (if present) and set up handlers.
    # This makes `vllm serve --attention-backend CUSTOM` work out of the box
    # without requiring the glassbox-run CLI wrapper.
    #
    # Skip if config was already set programmatically (e.g. by run_extraction
    # or glassbox-run).  After fork, the subprocess inherits the explicit
    # config; re-running set_config here would overwrite it with defaults.
    from glassbox.backends.svd_backend import SVDTritonAttentionImpl

    if not SVDTritonAttentionImpl._config_set_explicitly:
        from glassbox.config import GlassboxConfig

        SVDTritonAttentionImpl.set_config(GlassboxConfig())
