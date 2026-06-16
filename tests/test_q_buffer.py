"""Tests for Q-buffer windowing and position alignment."""

import torch

from glassbox.config import GlassboxConfig


def _trim(q_buffer: list[torch.Tensor], max_tokens: int) -> list[torch.Tensor]:
    """Replicate PerLayerSVDState.trim() logic for testing without vLLM import."""
    if max_tokens <= 0:
        return q_buffer
    total = sum(t.shape[0] for t in q_buffer)
    while total > max_tokens and q_buffer:
        removed = q_buffer.pop(0)
        total -= removed.shape[0]
    return q_buffer


class TestTrim:
    def test_noop_when_zero(self):
        buf = [torch.randn(10, 4, 64)]
        _trim(buf, 0)
        assert len(buf) == 1

    def test_noop_when_under_limit(self):
        buf = [torch.randn(5, 4, 64)]
        _trim(buf, 10)
        assert len(buf) == 1
        assert buf[0].shape[0] == 5

    def test_drops_oldest(self):
        t1 = torch.randn(10, 4, 64)
        t2 = torch.randn(5, 4, 64)
        t3 = torch.randn(3, 4, 64)
        buf = [t1, t2, t3]
        _trim(buf, 8)
        assert len(buf) == 2
        assert buf[0] is t2
        assert buf[1] is t3
        total = sum(t.shape[0] for t in buf)
        assert total == 8

    def test_bounded_after_many_appends(self):
        buf: list[torch.Tensor] = []
        for _ in range(100):
            buf.append(torch.randn(1, 4, 64))
            _trim(buf, 16)
        total = sum(t.shape[0] for t in buf)
        assert total <= 16

    def test_empty_buffer(self):
        buf: list[torch.Tensor] = []
        _trim(buf, 10)
        assert buf == []

    def test_exact_limit(self):
        buf = [torch.randn(5, 4, 64), torch.randn(5, 4, 64)]
        _trim(buf, 10)
        assert len(buf) == 2
        assert sum(t.shape[0] for t in buf) == 10


class TestConfig:
    def test_default_zero(self):
        cfg = GlassboxConfig()
        assert cfg.q_buffer_max_tokens == 0

    def test_default_mode_sliding(self):
        cfg = GlassboxConfig()
        assert cfg.q_buffer_mode == "sliding"

    def test_from_cli_args(self):
        cfg = GlassboxConfig.from_cli_args(q_buffer_max_tokens=512)
        assert cfg.q_buffer_max_tokens == 512

    def test_from_cli_args_tumbling(self):
        cfg = GlassboxConfig.from_cli_args(q_buffer_max_tokens=512, q_buffer_mode="tumbling")
        assert cfg.q_buffer_max_tokens == 512
        assert cfg.q_buffer_mode == "tumbling"

    def test_from_cli_args_none_uses_default(self):
        cfg = GlassboxConfig.from_cli_args()
        assert cfg.q_buffer_max_tokens == 0
        assert cfg.q_buffer_mode == "sliding"

    def test_programmatic(self):
        cfg = GlassboxConfig(q_buffer_max_tokens=1024)
        assert cfg.q_buffer_max_tokens == 1024

    def test_programmatic_tumbling(self):
        cfg = GlassboxConfig(q_buffer_max_tokens=256, q_buffer_mode="tumbling")
        assert cfg.q_buffer_mode == "tumbling"


class TestKAlignment:
    """Verify that Q and K slicing takes the LAST L rows (most recent positions)."""

    def test_last_l_rows_when_q_shorter(self):
        L_q, L_k = 8, 20
        num_heads, d = 4, 64
        Q_all = torch.arange(L_q).unsqueeze(1).unsqueeze(2).expand(L_q, num_heads, d).float()
        K_all = torch.arange(L_k).unsqueeze(1).unsqueeze(2).expand(L_k, num_heads, d).float()

        L = min(L_q, L_k)
        Q_win = Q_all[-L:]
        K_win = K_all[-L:]

        assert Q_win.shape[0] == L
        assert K_win.shape[0] == L
        assert Q_win[0, 0, 0].item() == 0
        assert K_win[0, 0, 0].item() == L_k - L

    def test_positions_match_when_windowed(self):
        """When Q buffer is windowed to W < L_k, the last W positions of K
        must correspond to the same sequence positions as Q."""
        W = 4
        L_k = 20
        Q_positions = list(range(L_k - W, L_k))
        K_all_positions = list(range(L_k))
        K_windowed_positions = K_all_positions[-W:]
        assert Q_positions == K_windowed_positions

    def test_noop_when_equal(self):
        L = 16
        Q = torch.randn(L, 4, 64)
        K = torch.randn(L, 4, 64)
        assert Q[-L:].data_ptr() == Q.data_ptr()
        assert K[-L:].data_ptr() == K.data_ptr()


class TestTumblingMode:
    """Simulate tumbling window semantics without importing vLLM."""

    @staticmethod
    def _simulate_tumbling(n_steps: int, W: int) -> list[int]:
        """Return the steps at which diagnostics fire in tumbling mode.

        Each step appends 1 token. When buffer reaches W, fire and flush.
        """
        buf_tokens = 0
        fire_steps: list[int] = []
        for step in range(1, n_steps + 1):
            buf_tokens += 1
            if buf_tokens >= W:
                fire_steps.append(step)
                buf_tokens = 0
        return fire_steps

    def test_fires_at_window_boundaries(self):
        fires = self._simulate_tumbling(20, W=5)
        assert fires == [5, 10, 15, 20]

    def test_non_overlapping(self):
        fires = self._simulate_tumbling(30, W=10)
        for i in range(1, len(fires)):
            assert fires[i] - fires[i - 1] == 10

    def test_no_partial_window_at_end(self):
        fires = self._simulate_tumbling(13, W=5)
        assert fires == [5, 10]

    def test_window_1_fires_every_step(self):
        fires = self._simulate_tumbling(5, W=1)
        assert fires == [1, 2, 3, 4, 5]

    def test_buffer_empty_after_flush(self):
        """After firing, buffer should be empty (0 tokens)."""
        buf: list[torch.Tensor] = []
        W = 4
        flush_counts = []
        for _ in range(12):
            buf.append(torch.randn(1, 4, 64))
            total = sum(t.shape[0] for t in buf)
            if total >= W:
                flush_counts.append(total)
                buf = []
        assert flush_counts == [4, 4, 4]
        assert buf == []
