"""Tests for the VerdictStore shared state."""

from __future__ import annotations

import threading

import pytest

from glassbox.verdict import Verdict, VerdictStore


@pytest.fixture(autouse=True)
def _clean_store():
    """Reset VerdictStore before and after each test."""
    VerdictStore.reset()
    yield
    VerdictStore.reset()


class TestVerdictStore:
    def test_report_and_consume(self):
        VerdictStore.map_request_id("req-abc", 0)
        VerdictStore.report(0, 0.85, "abort")

        verdict = VerdictStore.consume("req-abc")
        assert verdict is not None
        assert verdict.request_id == 0
        assert verdict.probability == pytest.approx(0.85)
        assert verdict.action == "abort"

    def test_consume_removes_verdict(self):
        VerdictStore.map_request_id("req-1", 0)
        VerdictStore.report(0, 0.9, "abort")

        first = VerdictStore.consume("req-1")
        assert first is not None

        second = VerdictStore.consume("req-1")
        assert second is None

    def test_unmapped_id_returns_none(self):
        VerdictStore.report(0, 0.9, "abort")
        assert VerdictStore.consume("unknown-id") is None

    def test_no_verdict_returns_none(self):
        VerdictStore.map_request_id("req-1", 0)
        # No report() call
        assert VerdictStore.consume("req-1") is None

    def test_clear_by_vllm_id(self):
        VerdictStore.map_request_id("req-1", 0)
        VerdictStore.report(0, 0.7, "abort")

        VerdictStore.clear_by_vllm_id("req-1")

        assert VerdictStore.consume("req-1") is None

    def test_clear_unknown_id_is_noop(self):
        VerdictStore.clear_by_vllm_id("nonexistent")  # should not raise

    def test_multiple_requests(self):
        VerdictStore.map_request_id("req-a", 0)
        VerdictStore.map_request_id("req-b", 1)
        VerdictStore.report(0, 0.3, "continue")
        VerdictStore.report(1, 0.9, "abort")

        va = VerdictStore.consume("req-a")
        vb = VerdictStore.consume("req-b")
        assert va.action == "continue"
        assert vb.action == "abort"

    def test_report_overwrites_previous(self):
        VerdictStore.map_request_id("req-1", 0)
        VerdictStore.report(0, 0.3, "continue")
        VerdictStore.report(0, 0.9, "abort")

        verdict = VerdictStore.consume("req-1")
        assert verdict.action == "abort"
        assert verdict.probability == pytest.approx(0.9)

    def test_thread_safety(self):
        """Concurrent report/consume from multiple threads."""
        n = 100
        results = [None] * n

        for i in range(n):
            VerdictStore.map_request_id(f"req-{i}", i)

        def writer():
            for i in range(n):
                VerdictStore.report(i, 0.5, "abort")

        def reader():
            for i in range(n):
                results[i] = VerdictStore.consume(f"req-{i}")

        t1 = threading.Thread(target=writer)
        t2 = threading.Thread(target=reader)
        t1.start()
        t1.join()  # write all, then read all
        t2.start()
        t2.join()

        consumed = [r for r in results if r is not None]
        assert len(consumed) == n

    def test_reset_clears_all(self):
        VerdictStore.map_request_id("req-1", 0)
        VerdictStore.report(0, 0.9, "abort")

        VerdictStore.reset()

        assert VerdictStore.consume("req-1") is None


class TestVerdict:
    def test_timestamp_auto_set(self):
        v = Verdict(request_id=0, probability=0.5, action="continue")
        assert v.timestamp > 0
