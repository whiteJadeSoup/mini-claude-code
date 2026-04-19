"""Unit tests for QueryEngine.compact()."""
import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_home(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(persistence, "SESSION_ID", "test-compact-session")
    monkeypatch.setattr(config, "CWD", "/test/cwd")
    return tmp_path


class _StubLLM:
    """Minimal LLM stub. Async-invoke only; compact uses ainvoke."""

    def __init__(self, summary="STUB SUMMARY BODY"):
        self._summary = summary
        self.last_messages = None

    async def ainvoke(self, messages, *args, **kwargs):
        self.last_messages = messages
        return AIMessage(
            content=f"<analysis>irrelevant</analysis>\n<summary>{self._summary}</summary>",
            usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            response_metadata={},
        )

    def bind_tools(self, tools):
        return self  # not used by compact

    async def astream(self, *a, **kw):  # not used by compact, but must exist
        yield AIMessage(content="")


@pytest.fixture
def engine(monkeypatch, isolated_home, fresh_tasks):
    stub = _StubLLM()
    eng = query_engine.QueryEngine(
        llm_base=stub,
        main_tools=[],
        sub_tools=[],
        model_name="stub",
        system_prompt_builder=lambda: "sys",
    )
    query_engine.set_engine(eng)
    return eng


def _populate(engine, sys_content="sys"):
    """Seed engine store with minimal main-branch conversation (no dispatch needed)."""
    engine.store.append(SystemPromptMessage(content=sys_content, source="boot"))
    engine.store.append(UserMessage(content="user turn"))
    engine.store.append(
        AssistantMessage(
            turn_id="t1", model="m", content=TextBlock(text="assistant reply")
        )
    )


# ---------------------------------------------------------------------------
# Post-state
# ---------------------------------------------------------------------------


class TestCompactPostState:
    @pytest.mark.asyncio
    async def test_store_has_system_compact_boundary_user(self, engine):
        _populate(engine)
        await engine.compact()
        msgs = engine.store.all()
        layer1 = [m for m in msgs if isinstance(m, (SystemPromptMessage, UserMessage))]
        layer2 = [m for m in msgs if isinstance(m, CompactBoundaryMessage)]
        assert len(layer1) == 2
        assert len(layer2) == 1
        assert isinstance(msgs[0], SystemPromptMessage)

    @pytest.mark.asyncio
    async def test_system_prompt_content_preserved(self, engine):
        engine.store.append(SystemPromptMessage(content="original system prompt", source="boot"))
        engine.store.append(UserMessage(content="hi"))
        await engine.compact()
        first = engine.store.all()[0]
        assert isinstance(first, SystemPromptMessage)
        assert first.content == "original system prompt"

    @pytest.mark.asyncio
    async def test_compact_message_is_synthetic(self, engine):
        _populate(engine)
        await engine.compact()
        synthetic = [
            m for m in engine.store.all()
            if isinstance(m, UserMessage) and m.is_synthetic
        ]
        assert len(synthetic) == 1

    @pytest.mark.asyncio
    async def test_body_contains_summary(self, engine):
        _populate(engine)
        await engine.compact()
        cm = next(
            m for m in engine.store.all()
            if isinstance(m, UserMessage) and m.is_synthetic
        )
        assert "STUB SUMMARY BODY" in cm.content

    @pytest.mark.asyncio
    async def test_body_contains_transcript_path(self, engine):
        _populate(engine)
        await engine.compact()
        cm = next(
            m for m in engine.store.all()
            if isinstance(m, UserMessage) and m.is_synthetic
        )
        assert str(persistence.transcript_path()) in cm.content

    @pytest.mark.asyncio
    async def test_body_contains_task_state_when_present(self, engine, fresh_tasks):
        fresh_tasks.plan([{"id": "a", "description": "do something"}])
        _populate(engine)
        await engine.compact()
        cm = next(
            m for m in engine.store.all()
            if isinstance(m, UserMessage) and m.is_synthetic
        )
        assert "Active tasks at time of compaction" in cm.content
        assert "[a]" in cm.content

    @pytest.mark.asyncio
    async def test_body_omits_task_state_when_empty(self, engine):
        _populate(engine)
        await engine.compact()
        cm = next(
            m for m in engine.store.all()
            if isinstance(m, UserMessage) and m.is_synthetic
        )
        assert "Active tasks at time of compaction" not in cm.content

    @pytest.mark.asyncio
    async def test_compact_boundary_records_pre_count(self, engine):
        _populate(engine)
        pre = len(engine.store.all())
        await engine.compact()
        boundary = next(
            m for m in engine.store.all() if isinstance(m, CompactBoundaryMessage)
        )
        assert boundary.pre_count == pre

    @pytest.mark.asyncio
    async def test_old_compact_boundaries_preserved(self, engine):
        engine.store.append(
            CompactBoundaryMessage(pre_count=10, auto=False, source="compact")
        )
        _populate(engine)
        await engine.compact()
        boundaries = [
            m for m in engine.store.all() if isinstance(m, CompactBoundaryMessage)
        ]
        assert len(boundaries) == 2


# ---------------------------------------------------------------------------
# auto vs manual body
# ---------------------------------------------------------------------------


class TestCompactAutoVsManual:
    @pytest.mark.asyncio
    async def test_auto_contains_behavior_lock(self, engine):
        _populate(engine)
        await engine.compact(auto=True)
        cm = next(
            m for m in engine.store.all()
            if isinstance(m, UserMessage) and m.is_synthetic
        )
        assert "Pick up the last task" in cm.content
        assert "ran out of context" in cm.content

    @pytest.mark.asyncio
    async def test_manual_omits_behavior_lock(self, engine):
        _populate(engine)
        await engine.compact(auto=False)
        cm = next(
            m for m in engine.store.all()
            if isinstance(m, UserMessage) and m.is_synthetic
        )
        assert "Pick up the last task" not in cm.content
        assert "at the user's request" in cm.content

    @pytest.mark.asyncio
    async def test_default_is_manual(self, engine):
        _populate(engine)
        await engine.compact()
        cm = next(
            m for m in engine.store.all()
            if isinstance(m, UserMessage) and m.is_synthetic
        )
        assert "Pick up the last task" not in cm.content

    @pytest.mark.asyncio
    async def test_custom_instructions_flow_through(self, engine):
        _populate(engine)
        await engine.compact(custom_instructions="focus on bugs")
        stub = engine._llm_base
        assert stub.last_messages is not None
        assert "focus on bugs" in stub.last_messages[-1].content


# ---------------------------------------------------------------------------
# Persistence side effect via PersistenceConsumer
# ---------------------------------------------------------------------------


class TestCompactPersists:
    @pytest.mark.asyncio
    async def test_compact_messages_written_to_jsonl(self, engine):
        from mini_cc.consumers.persistence import PersistenceConsumer
        engine.subscribe(PersistenceConsumer(), name="persistence")
        _populate(engine)
        await engine.compact(auto=True)
        path = persistence.transcript_path()
        assert path.exists()
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
        ]
        compact_records = [
            r for r in records
            if r.get("type") == "user" and r.get("is_synthetic")
        ]
        assert len(compact_records) == 1
        assert "STUB SUMMARY BODY" in compact_records[0]["content"]
