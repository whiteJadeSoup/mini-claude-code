"""boot injects the channel B-1 context message and drops the task_state
paired 'Acknowledged' assistant. boot only dispatches to the store (no LLM
call), so a MagicMock llm_base is enough. asyncio_mode=auto awaits these."""
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from mini_cc.engine import query_engine as qe_mod
from mini_cc.engine.query_engine import QueryEngine
from mini_cc.engine.messages import AssistantMessage


def _engine() -> QueryEngine:
    return QueryEngine(
        llm_base=MagicMock(),
        main_tools=[],
        sub_tools=[],
        model_name="test-model",
        system_prompt_builder=lambda: "SYS",
    )


async def test_boot_injects_memory_context(monkeypatch, fresh_tasks):
    monkeypatch.setattr(qe_mod, "build_memory_context",
                        lambda _dir: "Contents of MEMORY.md (...):\n\n- [Role](r.md) — x")
    engine = _engine()
    await engine.boot()
    view = engine.store.api_view()
    text = "\n".join(m.content for m in view if isinstance(m, HumanMessage))
    assert "Contents of MEMORY.md" in text
    assert "# memory" in text
    assert "# currentDate" in text
    assert "Today's date is" in text


async def test_boot_skips_memory_when_none(monkeypatch, fresh_tasks):
    monkeypatch.setattr(qe_mod, "build_memory_context", lambda _dir: None)
    engine = _engine()
    await engine.boot()
    view = engine.store.api_view()
    human = "\n".join(m.content for m in view if isinstance(m, HumanMessage))
    assert "Contents of MEMORY.md" not in human


async def test_boot_task_state_has_no_paired_assistant(monkeypatch, fresh_tasks):
    monkeypatch.setattr(qe_mod, "build_memory_context", lambda _dir: None)
    monkeypatch.setattr(qe_mod.tasks._tasks, "state_summary", lambda: "TASK STATE BLOCK")
    engine = _engine()
    await engine.boot()
    assistants = [m for m in engine.store._messages if isinstance(m, AssistantMessage)]
    assert all("Acknowledged" not in (getattr(m.content, "text", "") or "") for m in assistants)


async def test_boot_then_api_view_merges_into_single_human_preamble(monkeypatch, fresh_tasks):
    monkeypatch.setattr(qe_mod, "build_memory_context",
                        lambda _dir: "Contents of MEMORY.md (...):\n\n- [Role](r.md) — x")
    monkeypatch.setattr(qe_mod.tasks._tasks, "state_summary", lambda: "TASK STATE BLOCK")
    engine = _engine()
    await engine.boot()
    view = engine.store.api_view()
    assert isinstance(view[0], SystemMessage)
    humans = [m for m in view if isinstance(m, HumanMessage)]
    assert len(humans) == 1, f"expected 1 merged human, got {len(humans)}"
    merged = humans[0].content
    assert "Contents of MEMORY.md" in merged
    assert "TASK STATE BLOCK" in merged
    assert not any(isinstance(m, AIMessage) for m in view)
