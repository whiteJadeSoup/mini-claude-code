"""Unit tests for the compact/400-path diagnostic logging.

These prove the JSONL trace captures the data a human actually needs
when debugging a "why did we still hit 400" session: tracker snapshot
at the moment of failure, per-attempt drop counts, gap tokens, and
final outcome.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from mini_cc import config
from mini_cc.consumers import persistence
from mini_cc.engine import query_engine
from mini_cc.engine._diagnostics import diag_path, log_event, tracker_snapshot
from mini_cc.engine.messages import (
    AssistantMessage,
    SystemPromptMessage,
    TextBlock,
    UserMessage,
)
from mini_cc.state import usage


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_home(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(persistence, "SESSION_ID", "test-diag-session")
    monkeypatch.setattr(config, "CWD", "/test/cwd")
    return tmp_path


@pytest.fixture
def fresh_tracker(monkeypatch):
    fresh = usage.UsageTracker()
    monkeypatch.setattr(usage, "_tracker", fresh)
    return fresh


class _ProgrammableLLM:
    """Reuse of the same pattern from test_compact_overflow — scripted
    responses; raised exceptions surface as compact attempts that fail."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[list] = []

    async def ainvoke(self, messages, *a, **kw):
        self.calls.append(list(messages))
        if not self._responses:
            raise AssertionError("LLM stub ran out of scripted responses")
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        if callable(item):
            return item(messages)
        return item

    def bind_tools(self, tools):
        return self

    async def astream(self, *a, **kw):
        yield AIMessage(content="")


def _summary_response(body="STUB"):
    return AIMessage(
        content=f"<analysis>x</analysis>\n<summary>{body}</summary>",
        usage_metadata={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
        response_metadata={},
    )


def _make_engine(llm) -> query_engine.QueryEngine:
    eng = query_engine.QueryEngine(
        llm_base=llm,
        main_tools=[],
        sub_tools=[],
        model_name="stub",
        system_prompt_builder=lambda: "sys",
    )
    query_engine.set_engine(eng)
    return eng


def _seed(engine, rounds: int = 4, chars: int = 500):
    engine.store.append(SystemPromptMessage(content="sys", source="boot"))
    for i in range(rounds):
        engine.store.append(UserMessage(content=f"u{i} " + "x" * chars))
        engine.store.append(
            AssistantMessage(
                turn_id=f"t{i}", model="m",
                content=TextBlock(text=f"a{i} " + "x" * chars),
            )
        )


def _read_diag(isolated_home) -> list[dict]:
    path = diag_path()
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


# ---------------------------------------------------------------------------
# log_event — pure behavior
# ---------------------------------------------------------------------------


class TestLogEvent:
    def test_writes_jsonl_line(self, isolated_home):
        log_event("test", foo=1, bar="baz")
        records = _read_diag(isolated_home)
        assert len(records) == 1
        assert records[0]["event"] == "test"
        assert records[0]["foo"] == 1
        assert records[0]["bar"] == "baz"
        assert "timestamp" in records[0]

    def test_disk_failure_swallowed(self, isolated_home, monkeypatch):
        # Make Path.open raise; log_event MUST NOT propagate the error.
        def _boom(self, *a, **kw):
            raise OSError("no space left")

        monkeypatch.setattr(Path, "open", _boom)

        # Must not raise.
        log_event("test", foo=1)


# ---------------------------------------------------------------------------
# tracker_snapshot
# ---------------------------------------------------------------------------


class TestTrackerSnapshot:
    def test_no_records_returns_zeros(self, isolated_home, fresh_tracker):
        snap = tracker_snapshot()
        assert snap["last_input"] == 0
        assert snap["last_output"] == 0
        assert snap["last_cache_read"] == 0
        assert snap["last_reasoning"] == 0
        assert snap["total_in"] == 0
        assert snap["total_out"] == 0
        assert snap["context_limit"] > 0  # has a default

    def test_populated_after_record(self, isolated_home, fresh_tracker):
        fresh_tracker.record(
            "agent",
            {
                "input_tokens": 12345,
                "output_tokens": 678,
                "total_tokens": 13023,
                "input_token_details": {"cache_read": 500},
                "output_token_details": {"reasoning": 200},
            },
            {"model_name": "deepseek-reasoner"},
        )
        snap = tracker_snapshot()
        assert snap["last_input"] == 12345
        assert snap["last_output"] == 678
        assert snap["last_cache_read"] == 500
        assert snap["last_reasoning"] == 200
        assert snap["total_in"] == 12345
        assert snap["total_out"] == 678
        assert snap["model"] == "deepseek-reasoner"


# ---------------------------------------------------------------------------
# compact() emits diagnostic records
# ---------------------------------------------------------------------------


class TestCompactDiagnostics:
    @pytest.mark.asyncio
    async def test_success_writes_compact_record(
        self, isolated_home, fresh_tracker, fresh_tasks
    ):
        llm = _ProgrammableLLM([_summary_response()])
        engine = _make_engine(llm)
        _seed(engine, rounds=3)

        await engine.compact(trigger="manual")

        records = _read_diag(isolated_home)
        compact_records = [r for r in records if r["event"] == "compact"]
        assert len(compact_records) == 1
        r = compact_records[0]
        assert r["trigger"] == "manual"
        assert r["final_outcome"] == "success"
        assert r["final_error"] is None
        assert len(r["attempts"]) == 1
        assert r["attempts"][0]["outcome"] == "success"
        # Baseline tracker snapshot is captured.
        assert "tracker_before" in r
        assert "context_limit" in r["tracker_before"]

    @pytest.mark.asyncio
    async def test_retry_success_records_gap_and_next_extra_chars(
        self, isolated_home, fresh_tracker, fresh_tasks
    ):
        # First ainvoke raises context_exceeded with a DeepSeek-format
        # message; second attempt succeeds.
        err = RuntimeError(
            "maximum context length is 131072 tokens. However you requested 138172 tokens"
        )
        llm = _ProgrammableLLM([err, _summary_response()])
        engine = _make_engine(llm)
        _seed(engine, rounds=8, chars=800)

        await engine.compact(trigger="pre_call")

        records = _read_diag(isolated_home)
        compact_records = [r for r in records if r["event"] == "compact"]
        assert len(compact_records) == 1
        r = compact_records[0]
        assert r["trigger"] == "pre_call"
        assert r["final_outcome"] == "success"
        assert len(r["attempts"]) == 2

        a0 = r["attempts"][0]
        assert a0["outcome"] == "context_exceeded"
        assert a0["gap_tokens"] == 7100  # 138172 - 131072
        assert a0["next_extra_chars"] == (7100 + 5_000) * 2

        a1 = r["attempts"][1]
        assert a1["outcome"] == "success"
        # Second attempt dropped more rounds than the first (extra_chars > 0).
        assert a1["extra_chars"] > 0

    @pytest.mark.asyncio
    async def test_exhausted_attempts_records_context_exhausted(
        self, isolated_home, fresh_tracker, fresh_tasks
    ):
        # Use a SMALL gap (200 tokens over) so each retry's extra_chars is
        # small enough not to collapse kept to 1 — we want the exhaustion
        # path (hit MAX_ATTEMPTS while still having rounds to drop), not
        # the single-round-overflow fast-fail path.
        err = RuntimeError(
            "maximum context length is 131072 tokens. However you requested 131272 tokens"
        )
        llm = _ProgrammableLLM([err, err, err])
        engine = _make_engine(llm)
        _seed(engine, rounds=10, chars=1000)

        with pytest.raises(RuntimeError):
            await engine.compact(trigger="query_retry")

        records = _read_diag(isolated_home)
        compact_records = [r for r in records if r["event"] == "compact"]
        assert len(compact_records) == 1
        r = compact_records[0]
        assert r["trigger"] == "query_retry"
        assert r["final_outcome"] == "context_exhausted"
        assert r["final_error"] and "maximum context length" in r["final_error"]
        assert len(r["attempts"]) == 3
        # Every attempt recorded context_exceeded.
        assert all(a["outcome"] == "context_exceeded" for a in r["attempts"])

    @pytest.mark.asyncio
    async def test_single_round_overflow_fast_fail(
        self, isolated_home, fresh_tracker, fresh_tasks
    ):
        err = RuntimeError(
            "maximum context length is 131072 tokens. However you requested 250000 tokens"
        )
        llm = _ProgrammableLLM([err])
        engine = _make_engine(llm)
        # Seed with only ONE user turn — after grouping, len(groups) == 1,
        # so no room to drop further.
        engine.store.append(SystemPromptMessage(content="sys", source="boot"))
        engine.store.append(UserMessage(content="x" * 100_000))

        with pytest.raises(RuntimeError):
            await engine.compact(trigger="manual")

        records = _read_diag(isolated_home)
        compact_records = [r for r in records if r["event"] == "compact"]
        assert len(compact_records) == 1
        r = compact_records[0]
        assert r["final_outcome"] == "single_round_overflow"
        assert len(r["attempts"]) == 1
        assert len(llm.calls) == 1  # no retry on single-round overflow

    @pytest.mark.asyncio
    async def test_non_context_error_records_non_context_error(
        self, isolated_home, fresh_tracker, fresh_tasks
    ):
        llm = _ProgrammableLLM([RuntimeError("connection reset by peer")])
        engine = _make_engine(llm)
        _seed(engine, rounds=3)

        with pytest.raises(RuntimeError, match="connection reset"):
            await engine.compact(trigger="manual")

        records = _read_diag(isolated_home)
        compact_records = [r for r in records if r["event"] == "compact"]
        assert len(compact_records) == 1
        r = compact_records[0]
        assert r["final_outcome"] == "non_context_error"
        assert "connection reset" in r["final_error"]
        assert len(r["attempts"]) == 1
