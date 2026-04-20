"""Unit tests for the compact-overflow rescue path.

Covers the new API-round trimming + MARKER + retry machinery added to
query_engine.compact(). The existing test_compact.py covers the happy
path (post-state, auto vs manual body, persistence). This file focuses
on:
  - _group_by_api_round grouping
  - _plan_kept_groups invariants (H1: drop ≥1, H2: keep ≥1)
  - _parse_context_gap_tokens / _chars_to_shed_on_retry
  - compact retry state machine and MARKER injection
  - CompactBoundaryMessage audit metadata
  - UsageTracker.context_tokens_used (API-billed baseline + running delta)
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from mini_cc import config
from mini_cc.consumers import persistence
from mini_cc.engine import query_engine
from mini_cc.engine.messages import (
    AssistantMessage,
    CompactBoundaryMessage,
    SystemPromptMessage,
    TextBlock,
    UserMessage,
)
from mini_cc.engine.query_engine import (
    ApiRound,
    MARKER_TEXT,
    _chars_to_shed_on_retry,
    _group_by_api_round,
    _parse_context_gap_tokens,
    _plan_kept_groups,
)
from mini_cc.state import usage


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_home(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(persistence, "SESSION_ID", "test-compact-overflow")
    monkeypatch.setattr(config, "CWD", "/test/cwd")
    return tmp_path


@pytest.fixture
def fresh_tracker(monkeypatch):
    """Reset usage._tracker between tests so records don't bleed across."""
    fresh = usage.UsageTracker()
    monkeypatch.setattr(usage, "_tracker", fresh)
    return fresh


class _ProgrammableLLM:
    """LLM stub driven by a scripted response queue.

    Each item is either a callable returning the response, an Exception
    to raise, or an AIMessage to return directly. `calls` records every
    ainvoke payload for later inspection.
    """

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


def _make_summary_response(body="STUB SUMMARY"):
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


# ---------------------------------------------------------------------------
# _group_by_api_round
# ---------------------------------------------------------------------------


class TestGroupByApiRound:
    def test_empty_input(self):
        assert _group_by_api_round([]) == []

    def test_preamble_only(self):
        msgs = [HumanMessage(content="hello")]
        groups = _group_by_api_round(msgs)
        assert len(groups) == 1
        assert groups[0].is_preamble is True

    def test_single_round_no_preamble(self):
        # Conversation that somehow starts with an assistant message.
        msgs = [AIMessage(content="hi"), ToolMessage(content="r", tool_call_id="1")]
        groups = _group_by_api_round(msgs)
        assert len(groups) == 1
        assert groups[0].is_preamble is False

    def test_screenshot_scenario(self):
        # Mirrors the user's screenshot example after api_view() merging.
        # Expected layout: preamble, 5 assistant rounds (A-E), with a
        # trailing HumanMessage absorbed into round C.
        msgs = [
            HumanMessage(content="帮我重构 src/query.ts"),                 # group 0
            AIMessage(                                                       # group 1 start
                content="好的，我先读文件",
                tool_calls=[{"id": "1", "name": "Read", "args": {"path": "query.ts"}}],
            ),
            ToolMessage(content="<query.ts content>", tool_call_id="1"),
            AIMessage(                                                       # group 2 start
                content="文件较长，我并行 grep 两处",
                tool_calls=[
                    {"id": "2", "name": "Grep", "args": {"pattern": "QueryEngine"}},
                    {"id": "3", "name": "Grep", "args": {"pattern": "compact"}},
                ],
            ),
            ToolMessage(content="grep1 results", tool_call_id="2"),
            ToolMessage(content="grep2 results", tool_call_id="3"),
            AIMessage(                                                       # group 3 start
                content="准备分三步改",
                tool_calls=[{"id": "4", "name": "Edit", "args": {}}],
            ),
            ToolMessage(content="edit ok", tool_call_id="4"),
            HumanMessage(content="再帮我补下测试"),                          # absorbed into group 3
            AIMessage(                                                       # group 4 start
                content="",
                tool_calls=[{"id": "5", "name": "Glob", "args": {}}],
            ),
            ToolMessage(content="file list", tool_call_id="5"),
            AIMessage(content="写完了"),                                     # group 5 start
        ]
        groups = _group_by_api_round(msgs)
        assert len(groups) == 6
        assert groups[0].is_preamble is True
        assert groups[1].is_preamble is False
        # The trailing HumanMessage "再帮我补下测试" is absorbed into group 3.
        group3_humans = [m for m in groups[3].messages if isinstance(m, HumanMessage)]
        assert len(group3_humans) == 1
        assert "再帮我补下测试" in group3_humans[0].content


# ---------------------------------------------------------------------------
# ApiRound.size_chars — includes tool_calls args
# ---------------------------------------------------------------------------


class TestSizeEstimate:
    def test_size_includes_tool_call_args(self):
        big_args = {"new_string": "x" * 5000}
        ai = AIMessage(
            content="short",
            tool_calls=[{"id": "1", "name": "Edit", "args": big_args}],
        )
        round_ = ApiRound(messages=[ai], is_preamble=False)
        # Estimate must include the 5000-char args payload, not just
        # len("short") + formatting overhead.
        assert round_.size_chars > 5000

    def test_size_plain_message(self):
        human = HumanMessage(content="x" * 1000)
        round_ = ApiRound(messages=[human], is_preamble=True)
        # ~1000 chars of content + ~16 chars of format overhead.
        assert 1000 <= round_.size_chars <= 1100


# ---------------------------------------------------------------------------
# _plan_kept_groups — H1 and H2
# ---------------------------------------------------------------------------


def _round(size: int, is_preamble: bool = False) -> ApiRound:
    """Build a synthetic ApiRound of approximately `size` chars."""
    content = "x" * max(0, size - 16)  # minus format overhead
    msg = (
        HumanMessage(content=content) if is_preamble
        else AIMessage(content=content)
    )
    return ApiRound(messages=[msg], is_preamble=is_preamble)


class TestPlanKeptGroups:
    def test_empty_input(self):
        kept, dropped, marker = _plan_kept_groups([], 100_000, 0)
        assert kept == []
        assert dropped == 0
        assert marker is False

    def test_h1_forces_drop_when_size_fits(self):
        # H1: compact is a reduction op — even if total fits budget, we must
        # drop at least 1 round when we have ≥ 2 rounds.
        groups = [_round(100, True), _round(100), _round(100)]
        kept, dropped, _ = _plan_kept_groups(groups, 1_000_000, 0)
        assert dropped >= 1
        assert len(kept) == len(groups) - dropped

    def test_h2_keeps_at_least_one(self):
        # H2: no matter how aggressive, we never drop down to zero rounds.
        groups = [_round(10_000, True), _round(10_000), _round(10_000)]
        kept, dropped, _ = _plan_kept_groups(groups, 100, extra_chars_to_shed=10_000_000)
        assert len(kept) == 1

    def test_single_group_h1_yields_to_h2(self):
        # Single-round edge case: H2 wins — we keep the only round rather
        # than dropping to empty.
        groups = [_round(100, True)]
        kept, dropped, marker = _plan_kept_groups(groups, 1000, 0)
        assert kept == groups
        assert dropped == 0
        assert marker is False

    def test_marker_when_preamble_dropped(self):
        groups = [_round(100, is_preamble=True), _round(100), _round(100)]
        kept, dropped, marker = _plan_kept_groups(groups, 1_000_000, 0)
        assert dropped >= 1
        # groups[0] was the preamble and must no longer be the first kept round.
        assert kept[0] is not groups[0]
        assert marker is True

    def test_extra_chars_forces_more_drops(self):
        groups = [_round(5_000, True), _round(5_000), _round(5_000), _round(5_000)]
        base_kept, base_dropped, _ = _plan_kept_groups(groups, 100_000, 0)
        more_kept, more_dropped, _ = _plan_kept_groups(
            groups, 100_000, extra_chars_to_shed=10_000
        )
        assert more_dropped >= base_dropped
        assert len(more_kept) <= len(base_kept)


# ---------------------------------------------------------------------------
# _parse_context_gap_tokens / _chars_to_shed_on_retry
# ---------------------------------------------------------------------------


class TestGapParsing:
    def test_deepseek_format_parsed(self):
        err = (
            "Error code: 400 - {'error': {'message': \"This model's maximum context "
            "length is 131072 tokens. However you requested 138172 tokens (138172 in "
            "the messages, 0 in the completion).\"}}"
        )
        assert _parse_context_gap_tokens(err) == 7100

    def test_unknown_format_returns_none(self):
        err = "AnthropicError: prompt is too long for model"
        assert _parse_context_gap_tokens(err) is None

    def test_chars_to_shed_uses_gap_when_available(self):
        err = (
            "maximum context length is 131072 tokens. However you requested "
            "138172 tokens"
        )
        # gap=7100 tokens + safety margin 5000 → 12100 tokens → 24200 chars
        assert _chars_to_shed_on_retry(err, 131_072) == 24_200

    def test_chars_to_shed_falls_back_to_20_percent(self):
        err = "AnthropicError: prompt is too long"
        # 131_072 * 0.20 = 26_214 tokens → 52_428 chars
        assert _chars_to_shed_on_retry(err, 131_072) == 52_428


# ---------------------------------------------------------------------------
# compact() end-to-end behavior
# ---------------------------------------------------------------------------


def _populate_many_rounds(engine, n_rounds: int, chars_per_round: int = 1000) -> None:
    """Seed engine.store with n_rounds of user+assistant pairs.

    Enough variety so compact's grouping produces distinct ApiRounds.
    Tracker sees nothing — context_tokens_used walks the store's api_view
    at read time, so seeding directly into the store is enough.
    """
    engine.store.append(SystemPromptMessage(content="sys", source="boot"))
    for i in range(n_rounds):
        engine.store.append(UserMessage(content=f"user {i} " + "x" * chars_per_round))
        engine.store.append(AssistantMessage(
            turn_id=f"t{i}",
            model="m",
            content=TextBlock(text=f"assistant {i} " + "x" * chars_per_round),
        ))


class TestCompactIntegration:
    @pytest.mark.asyncio
    async def test_success_on_first_attempt_records_metadata(
        self, isolated_home, fresh_tracker, fresh_tasks
    ):
        llm = _ProgrammableLLM([_make_summary_response()])
        engine = _make_engine(llm)
        _populate_many_rounds(engine, 5, chars_per_round=500)

        await engine.compact(auto=True)

        boundary = next(
            m for m in engine.store.all() if isinstance(m, CompactBoundaryMessage)
        )
        assert boundary.attempts == 1
        assert boundary.retained_rounds >= 1
        # With 5 rounds and plenty of budget, H1 still forces ≥1 drop.
        assert boundary.dropped_rounds >= 1
        assert boundary.original_chars > 0
        assert boundary.sent_chars > 0

    @pytest.mark.asyncio
    async def test_marker_prepended_when_preamble_dropped(
        self, isolated_home, fresh_tracker, fresh_tasks
    ):
        llm = _ProgrammableLLM([_make_summary_response()])
        engine = _make_engine(llm)
        _populate_many_rounds(engine, 5, chars_per_round=500)

        await engine.compact(auto=True)

        # When at least the preamble is dropped, MARKER_TEXT must appear in
        # the formatted payload sent to the compact LLM.
        sent_to_llm = llm.calls[0][-1].content
        boundary = next(
            m for m in engine.store.all() if isinstance(m, CompactBoundaryMessage)
        )
        if boundary.marker_used:
            assert MARKER_TEXT in sent_to_llm

    @pytest.mark.asyncio
    async def test_retry_on_context_exceeded_with_gap(
        self, isolated_home, fresh_tracker, fresh_tasks
    ):
        first_error = RuntimeError(
            "maximum context length is 131072 tokens. However you requested 138172 tokens"
        )
        llm = _ProgrammableLLM([first_error, _make_summary_response()])
        engine = _make_engine(llm)
        _populate_many_rounds(engine, 10, chars_per_round=800)

        await engine.compact(auto=True)

        assert len(llm.calls) == 2, "Second attempt should have been issued"
        boundary = next(
            m for m in engine.store.all() if isinstance(m, CompactBoundaryMessage)
        )
        assert boundary.attempts == 2

    @pytest.mark.asyncio
    async def test_non_context_error_propagates_without_retry(
        self, isolated_home, fresh_tracker, fresh_tasks
    ):
        llm = _ProgrammableLLM([RuntimeError("connection reset by peer")])
        engine = _make_engine(llm)
        _populate_many_rounds(engine, 3)

        with pytest.raises(RuntimeError, match="connection reset"):
            await engine.compact(auto=True)
        # Only one attempt: we don't retry non-context errors.
        assert len(llm.calls) == 1

    @pytest.mark.asyncio
    async def test_all_attempts_exhausted_preserves_tracker(
        self, isolated_home, fresh_tracker, fresh_tasks
    ):
        # Pre-seed tracker with a successful record so we can verify it
        # survives a fully-failed compact.
        fresh_tracker.record(
            "pre",
            {"input_tokens": 12345, "output_tokens": 100, "total_tokens": 12445},
            {"model_name": "stub"},
        )
        err = RuntimeError(
            "maximum context length is 131072 tokens. However you requested 200000 tokens"
        )
        llm = _ProgrammableLLM([err, err, err])
        engine = _make_engine(llm)
        _populate_many_rounds(engine, 10, chars_per_round=800)

        with pytest.raises(RuntimeError, match="maximum context length"):
            await engine.compact(auto=True)
        # Tracker must still hold the pre-existing record — failed compact
        # doesn't reset anything.
        assert len(fresh_tracker._records) == 1
        assert fresh_tracker._records[0].input == 12345
        assert fresh_tracker._records[0].output == 100

    @pytest.mark.asyncio
    async def test_single_round_overflow_fast_fails(
        self, isolated_home, fresh_tracker, fresh_tasks
    ):
        err = RuntimeError(
            "maximum context length is 131072 tokens. However you requested 200000 tokens"
        )
        llm = _ProgrammableLLM([err])
        engine = _make_engine(llm)
        # Only 1 user turn in the store — after grouping, len(groups) == 1.
        # H2 prevents further drops, so compact should raise without retry.
        engine.store.append(SystemPromptMessage(content="sys", source="boot"))
        engine.store.append(UserMessage(content="x" * 100_000))

        with pytest.raises(RuntimeError, match="maximum context length"):
            await engine.compact(auto=True)
        assert len(llm.calls) == 1, "Single-round overflow should not trigger retry"


# ---------------------------------------------------------------------------
# UsageTracker.context_tokens_used — powers /context, TUI footer, and auto-compact
# ---------------------------------------------------------------------------


class TestContextTokensUsed:
    """context_tokens_used(messages) returns what the next API call will send.

    Design matches OpenCode / Roo-Code / Gemini CLI: trust the API's
    last-reported input+output as baseline, locally char-estimate only
    the messages dispatched AFTER that response (tool_results, pending
    user msg).
    """

    @pytest.mark.asyncio
    async def test_empty_returns_zero(self, isolated_home, fresh_tracker):
        assert fresh_tracker.context_tokens_used([]) == 0

    @pytest.mark.asyncio
    async def test_pre_first_record_uses_char_estimate(
        self, isolated_home, fresh_tracker
    ):
        # No record() yet (boot / post-compact): full-history char estimate.
        msgs = [
            HumanMessage(content="x" * 5000),
        ]
        est = fresh_tracker.context_tokens_used(msgs)
        # ~5k chars / 2 tokens-per-char ≈ 2500, plus overhead.
        assert est > 2_000

    @pytest.mark.asyncio
    async def test_post_record_baseline_when_ends_in_ai(
        self, isolated_home, fresh_tracker
    ):
        # After record() and with api_view ending in an AI message (no
        # tool_results yet), occupancy = API baseline exactly.
        fresh_tracker.record(
            "agent",
            {"input_tokens": 5000, "output_tokens": 500, "total_tokens": 5500},
            {"model_name": "stub"},
        )
        msgs = [
            HumanMessage(content="hi"),
            AIMessage(content="there"),
        ]
        assert fresh_tracker.context_tokens_used(msgs) == 5500

    @pytest.mark.asyncio
    async def test_pending_tool_result_added_to_baseline(
        self, isolated_home, fresh_tracker
    ):
        # A big tool_result landed after the last API response. The walk
        # stops at the last AIMessage; everything after counts as pending.
        fresh_tracker.record(
            "agent",
            {"input_tokens": 100, "output_tokens": 10, "total_tokens": 110},
            {"model_name": "stub"},
        )
        msgs = [
            HumanMessage(content="hi"),
            AIMessage(content="calling tool", tool_calls=[{"id": "1", "name": "read", "args": {}}]),
            ToolMessage(content="y" * 10_000, tool_call_id="1"),
        ]
        # baseline(110) + ~5k from 10k tool_result chars ≫ 110.
        assert fresh_tracker.context_tokens_used(msgs) > 3_000

    @pytest.mark.asyncio
    async def test_walk_stops_at_last_ai_only(self, isolated_home, fresh_tracker):
        # Older tool_result sits BEFORE the last AI; it's covered by the
        # API baseline and must not be double-counted.
        fresh_tracker.record(
            "agent",
            {"input_tokens": 100, "output_tokens": 10, "total_tokens": 110},
            {"model_name": "stub"},
        )
        msgs = [
            HumanMessage(content="hi"),
            AIMessage(content="", tool_calls=[{"id": "1", "name": "r", "args": {}}]),
            ToolMessage(content="y" * 10_000, tool_call_id="1"),   # old: already billed
            AIMessage(content="done"),                             # last AI boundary
        ]
        # No pending content after the last AI → occupancy == baseline.
        assert fresh_tracker.context_tokens_used(msgs) == 110

    @pytest.mark.asyncio
    async def test_cleared_tool_result_reduces_estimate_naturally(
        self, isolated_home, fresh_tracker
    ):
        # _clear_old_tool_results replaces old content with "[Cleared]".
        # Because context_tokens_used reads the CURRENT messages, a
        # cleared tool_result (if pending — after last AI) naturally
        # reports less. No separate hook needed.
        msgs_full = [
            HumanMessage(content="hi"),
            AIMessage(content=""),
            ToolMessage(content="y" * 10_000, tool_call_id="1"),
        ]
        msgs_cleared = [
            HumanMessage(content="hi"),
            AIMessage(content=""),
            ToolMessage(content="[Cleared]", tool_call_id="1"),
        ]
        fresh_tracker.record(
            "agent",
            {"input_tokens": 100, "output_tokens": 10, "total_tokens": 110},
            {"model_name": "stub"},
        )
        full = fresh_tracker.context_tokens_used(msgs_full)
        cleared = fresh_tracker.context_tokens_used(msgs_cleared)
        assert cleared < full
        # cleared occupancy ≈ baseline (the short "[Cleared]" string is tiny).
        assert cleared - 110 < 50

    @pytest.mark.asyncio
    async def test_sub_agent_tracker_isolated(self, isolated_home, fresh_tracker):
        # _sub_agent_scope swaps usage._tracker for a fresh instance; the
        # outer tracker's state is unchanged on exit.
        from mini_cc.tools.builtins import _sub_agent_scope
        fresh_tracker.record(
            "agent",
            {"input_tokens": 5000, "output_tokens": 500, "total_tokens": 5500},
            {"model_name": "stub"},
        )
        before = fresh_tracker.context_tokens_used([AIMessage(content="x")])
        assert before == 5500
        with _sub_agent_scope("test"):
            # Inside scope, usage._tracker is a different instance.
            assert usage._tracker is not fresh_tracker
            assert usage._tracker.context_tokens_used([]) == 0
        # Back to the outer tracker unchanged.
        assert usage._tracker is fresh_tracker
        assert fresh_tracker.context_tokens_used([AIMessage(content="x")]) == before

    @pytest.mark.asyncio
    async def test_headroom_left_inverts_context_tokens_used(
        self, isolated_home, fresh_tracker
    ):
        fresh_tracker.record(
            "agent",
            {"input_tokens": 50_000, "output_tokens": 500, "total_tokens": 50_500},
            {"model_name": "deepseek-reasoner"},
        )
        msgs = [AIMessage(content="done")]
        assert (
            fresh_tracker.headroom_left(msgs)
            == fresh_tracker.context_limit - fresh_tracker.context_tokens_used(msgs)
        )


# ---------------------------------------------------------------------------
# Persistence — new metadata fields are written to JSONL
# ---------------------------------------------------------------------------


class TestCompactBoundaryPersistence:
    @pytest.mark.asyncio
    async def test_boundary_metadata_in_jsonl(
        self, isolated_home, fresh_tracker, fresh_tasks
    ):
        from mini_cc.consumers.persistence import PersistenceConsumer

        llm = _ProgrammableLLM([_make_summary_response()])
        engine = _make_engine(llm)
        engine.subscribe(PersistenceConsumer(), name="persistence")
        _populate_many_rounds(engine, 4, chars_per_round=300)
        await engine.compact(auto=True)

        path = persistence.transcript_path()
        records = [
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        ]
        boundaries = [r for r in records if r.get("type") == "compact_boundary"]
        assert len(boundaries) == 1
        b = boundaries[0]
        for key in (
            "dropped_rounds",
            "retained_rounds",
            "marker_used",
            "attempts",
            "original_chars",
            "sent_chars",
        ):
            assert key in b, f"missing metadata field: {key}"

    def test_old_jsonl_records_still_deserialize(self):
        # Old persisted records (pre-rewrite) lack the new fields. Pydantic
        # should fill defaults.
        data = {
            "type": "compact_boundary",
            "pre_count": 42,
            "auto": True,
            "id": "x",
            "session_id": "s",
            "cwd": "/cwd",
            "created_at": "2025-01-01T00:00:00+00:00",
            "source": "compact",
        }
        msg = CompactBoundaryMessage(**data)
        assert msg.dropped_rounds == 0
        assert msg.retained_rounds == 0
        assert msg.marker_used is False
        assert msg.attempts == 1
        assert msg.original_chars == 0
        assert msg.sent_chars == 0
