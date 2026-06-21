"""Unit tests for SVDTritonAttentionImpl.set_config class-state effects.

Imports the backend (and thus vLLM), but exercises only set_config's
class-attribute bookkeeping — no engine, no GPU. This is the only coverage of
set_config in the `not e2e` suite (the e2e tests that touch it don't run in CI).
"""

import pytest

from glassbox.backends.svd_backend import PerLayerState, SVDTritonAttentionImpl
from glassbox.config import GlassboxConfig
from glassbox.qbuffer import QBuffer


@pytest.fixture
def backend():
    """Yield the impl class with its config-derived state saved and restored,
    so these tests don't leak class-level state into each other."""
    cls = SVDTritonAttentionImpl
    saved = (
        cls.config,
        cls.state_dict,
        cls._diagnostics,
        cls._handlers,
        cls._config_set_explicitly,
    )
    try:
        yield cls
    finally:
        (
            cls.config,
            cls.state_dict,
            cls._diagnostics,
            cls._handlers,
            cls._config_set_explicitly,
        ) = saved


def test_resets_state_dict(backend):
    """set_config drops per-layer state so QBuffers rebuild with the new policy."""
    backend.state_dict["model.layers.0.self_attn"] = PerLayerState(
        qbuf=QBuffer(max_tokens=8, mode="sliding")
    )
    backend.set_config(GlassboxConfig(q_buffer_max_tokens=256, q_buffer_mode="tumbling"))
    assert backend.state_dict == {}


def test_rebuilds_diagnostics_from_new_config(backend):
    """set_config rebuilds the diagnostics cache, reflecting the new config."""
    backend.set_config(GlassboxConfig(spectral={"enabled": True, "rank": 7}))
    assert set(backend._diagnostics) == {
        "spectral",
        "routing",
        "tracker",
        "selfattn",
        "laplacian",
    }
    # The cached diagnostic captured the new algorithm param, not the default (4).
    assert backend._diagnostics["spectral"].rank == 7


def test_sets_explicit_flag(backend):
    """set_config marks config as explicitly set (the anti-clobber guard)."""
    backend._config_set_explicitly = False
    backend.set_config(GlassboxConfig())
    assert backend._config_set_explicitly is True


def test_rebuilds_handlers(backend):
    """set_config rebuilds the handler list from the new config."""
    backend.set_config(GlassboxConfig())
    assert isinstance(backend._handlers, list)
