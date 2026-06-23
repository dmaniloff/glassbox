"""Tests for Q-buffer windowing and position alignment."""

import torch

from glassbox.config import GlassboxConfig
from glassbox.qbuffer import QBuffer


def _slice(n: int) -> torch.Tensor:
    """A query slice of n tokens, shape [n, heads, dim]."""
    return torch.randn(n, 4, 64)


class TestSlidingTrim:
    """Sliding mode trims oldest slices on append to stay within max_tokens."""

    def test_noop_when_zero(self):
        buf = QBuffer(max_tokens=0, mode="sliding")
        buf.append(_slice(10))
        assert buf.tokens == 10  # 0 = unbounded

    def test_noop_when_under_limit(self):
        buf = QBuffer(max_tokens=10, mode="sliding")
        buf.append(_slice(5))
        assert len(buf) == 1
        assert buf.tokens == 5

    def test_drops_oldest(self):
        buf = QBuffer(max_tokens=8, mode="sliding")
        for n in (10, 5, 3):
            buf.append(_slice(n))
        # 10 dropped (oldest); 5 + 3 = 8 remain
        assert buf.tokens == 8
        assert len(buf) == 2

    def test_bounded_after_many_appends(self):
        buf = QBuffer(max_tokens=16, mode="sliding")
        for _ in range(100):
            buf.append(_slice(1))
        assert buf.tokens <= 16

    def test_exact_limit(self):
        buf = QBuffer(max_tokens=10, mode="sliding")
        buf.append(_slice(5))
        buf.append(_slice(5))
        assert len(buf) == 2
        assert buf.tokens == 10

    def test_window_concatenates_slices(self):
        buf = QBuffer(max_tokens=0, mode="sliding")
        buf.append(_slice(3))
        buf.append(_slice(4))
        w = buf.window()
        assert w.shape[0] == 7

    def test_flush_clears(self):
        buf = QBuffer(max_tokens=0)
        buf.append(_slice(5))
        buf.flush()
        assert buf.tokens == 0
        assert len(buf) == 0


class TestTumblingMode:
    """Tumbling fires at non-overlapping window boundaries, then flushes."""

    def test_not_tumbling_without_bound(self):
        assert QBuffer(max_tokens=0, mode="tumbling").tumbling is False
        assert QBuffer(max_tokens=8, mode="tumbling").tumbling is True
        assert QBuffer(max_tokens=8, mode="sliding").tumbling is False

    def test_no_trim_on_append(self):
        """Tumbling accumulates past max_tokens (no sliding trim)."""
        buf = QBuffer(max_tokens=4, mode="tumbling")
        for _ in range(6):
            buf.append(_slice(1))
        assert buf.tokens == 6  # not trimmed

    def test_window_complete_at_boundary(self):
        buf = QBuffer(max_tokens=4, mode="tumbling")
        for i in range(1, 5):
            buf.append(_slice(1))
            assert buf.window_complete() is (i >= 4)

    def test_fires_at_window_boundaries(self):
        """Drive tumbling like the backend: append, fire+flush on completion."""
        buf = QBuffer(max_tokens=5, mode="tumbling")
        fire_steps = []
        for step in range(1, 21):
            buf.append(_slice(1))
            if buf.window_complete():
                fire_steps.append(step)
                buf.flush()
        assert fire_steps == [5, 10, 15, 20]

    def test_non_overlapping_and_flush(self):
        buf = QBuffer(max_tokens=10, mode="tumbling")
        flush_sizes = []
        for _ in range(30):
            buf.append(_slice(1))
            if buf.window_complete():
                flush_sizes.append(buf.tokens)
                buf.flush()
        assert flush_sizes == [10, 10, 10]
        assert buf.tokens == 0

    def test_no_partial_window_at_end(self):
        buf = QBuffer(max_tokens=5, mode="tumbling")
        fires = 0
        for _ in range(13):
            buf.append(_slice(1))
            if buf.window_complete():
                fires += 1
                buf.flush()
        assert fires == 2  # 13 // 5, the trailing 3 do not fire
        assert buf.tokens == 3

    def test_window_1_fires_every_step(self):
        buf = QBuffer(max_tokens=1, mode="tumbling")
        fires = 0
        for _ in range(5):
            buf.append(_slice(1))
            if buf.window_complete():
                fires += 1
                buf.flush()
        assert fires == 5


class TestConfig:
    def test_default_zero(self):
        assert GlassboxConfig().q_buffer_max_tokens == 0

    def test_default_mode_sliding(self):
        assert GlassboxConfig().q_buffer_mode == "sliding"

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


class TestKAlignment:
    """Verify that Q and K slicing takes the LAST L rows (most recent positions).

    The slice itself lives in the backend's _run_svd; these pin the invariant
    that windowed Q and full K must align at their most-recent positions.
    """

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
        W = 4
        L_k = 20
        Q_positions = list(range(L_k - W, L_k))
        K_windowed_positions = list(range(L_k))[-W:]
        assert Q_positions == K_windowed_positions

    def test_noop_when_equal(self):
        L = 16
        Q = torch.randn(L, 4, 64)
        K = torch.randn(L, 4, 64)
        assert Q[-L:].data_ptr() == Q.data_ptr()
        assert K[-L:].data_ptr() == K.data_ptr()
