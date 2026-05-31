"""Prefetch wiring：kick-off / consume / cancel 三处 seam（不打 API）。"""
import asyncio
from unittest.mock import MagicMock

import pytest

from mini_cc.engine import query_engine as qe_mod
from mini_cc.engine.messages import RelevantMemoryMessage, SurfacedMemory, UserMessage
from mini_cc.engine.query_engine import QueryEngine


def _engine() -> QueryEngine:
    # QueryEngine.__init__ 真实签名：llm_base, main_tools, sub_tools, model_name, system_prompt_builder
    # llm_base.bind_tools() 在 __init__ 里被调用，需要 MagicMock 支持
    return QueryEngine(llm_base=MagicMock(), main_tools=[], sub_tools=[],
                       model_name="test", system_prompt_builder=lambda: "SYS")


def _sm(filename="user_role.md"):
    return SurfacedMemory(filename=filename, path=f"/abs/{filename}",
                          content="likes uv", mtime_ms=1, line_count=1,
                          header=f"Memory: {filename}:")


def _patch_memdir(monkeypatch, tmp_path):
    monkeypatch.setattr(qe_mod, "get_auto_mem_path", lambda: tmp_path)


async def test_start_skips_single_word(monkeypatch, tmp_path):
    _patch_memdir(monkeypatch, tmp_path)
    eng = _engine()
    eng._start_memory_prefetch(UserMessage(content="hi", source="user"))
    assert eng._pending is None


async def test_consume_injects_and_records(monkeypatch, tmp_path):
    _patch_memdir(monkeypatch, tmp_path)
    async def _fake_surface(*a, **k):
        return [_sm()]
    monkeypatch.setattr(qe_mod, "surface_relevant", _fake_surface)
    monkeypatch.setattr(qe_mod.file_read_state._state, "get", lambda p: None)
    recorded = []
    monkeypatch.setattr(qe_mod.file_read_state._state, "record",
                        lambda path, *a, **k: recorded.append(path))

    eng = _engine()
    eng._start_memory_prefetch(UserMessage(content="recommend a pkg manager", source="user"))
    await eng._pending.task
    await eng._consume_prefetch_if_ready(parent_id=None)

    injected = [m for m in eng.store._messages if isinstance(m, RelevantMemoryMessage)]
    assert len(injected) == 1 and eng._pending.consumed is True
    assert "/abs/user_role.md" in recorded


async def test_consume_noop_when_not_ready(monkeypatch, tmp_path):
    _patch_memdir(monkeypatch, tmp_path)
    async def _slow(*a, **k):
        await asyncio.sleep(10)
        return []
    monkeypatch.setattr(qe_mod, "surface_relevant", _slow)
    eng = _engine()
    eng._start_memory_prefetch(UserMessage(content="a real query", source="user"))
    await eng._consume_prefetch_if_ready(parent_id=None)
    assert [m for m in eng.store._messages if isinstance(m, RelevantMemoryMessage)] == []
    assert eng._pending.consumed is False
    eng._cancel_prefetch()


async def test_consume_skips_sidechain(monkeypatch, tmp_path):
    _patch_memdir(monkeypatch, tmp_path)
    async def _fake_surface(*a, **k):
        return [_sm()]
    monkeypatch.setattr(qe_mod, "surface_relevant", _fake_surface)
    eng = _engine()
    eng._start_memory_prefetch(UserMessage(content="a real query", source="user"))
    await eng._pending.task
    await eng._consume_prefetch_if_ready(parent_id="sidechain-id")
    assert [m for m in eng.store._messages if isinstance(m, RelevantMemoryMessage)] == []


async def test_consume_skips_already_read_unchanged(monkeypatch, tmp_path):
    # 读过且没变（mtime 一致 → is_consistent）→ 跳过，不重复注入。
    from mini_cc.state.file_read_state import FileReadState
    _patch_memdir(monkeypatch, tmp_path)
    async def _fake_surface(*a, **k):
        return [_sm()]                                   # mtime_ms=1, content="likes uv"
    monkeypatch.setattr(qe_mod, "surface_relevant", _fake_surface)
    seen = FileReadState()
    seen.record("/abs/user_role.md", "likes uv", 1, offset=1, limit=1)  # 同 mtime+content
    monkeypatch.setattr(qe_mod.file_read_state, "_state", seen)
    eng = _engine()
    eng._start_memory_prefetch(UserMessage(content="a real query", source="user"))
    await eng._pending.task
    await eng._consume_prefetch_if_ready(parent_id=None)
    assert [m for m in eng.store._messages if isinstance(m, RelevantMemoryMessage)] == []


async def test_consume_resurfaces_stale(monkeypatch, tmp_path):
    # 读过但文件已变（mtime+content 不同 → is_consistent False）→ 重新召回。
    from mini_cc.state.file_read_state import FileReadState
    _patch_memdir(monkeypatch, tmp_path)
    async def _fake_surface(*a, **k):
        return [_sm()]                                   # 当前：mtime_ms=1, content="likes uv"
    monkeypatch.setattr(qe_mod, "surface_relevant", _fake_surface)
    stale = FileReadState()
    stale.record("/abs/user_role.md", "OLD likes pip", 999, offset=1, limit=1)  # 旧 mtime+content
    monkeypatch.setattr(qe_mod.file_read_state, "_state", stale)
    eng = _engine()
    eng._start_memory_prefetch(UserMessage(content="a real query", source="user"))
    await eng._pending.task
    await eng._consume_prefetch_if_ready(parent_id=None)
    injected = [m for m in eng.store._messages if isinstance(m, RelevantMemoryMessage)]
    assert len(injected) == 1                            # 内容变了 → 重新召回


async def test_consume_empty_surface_marks_consumed(monkeypatch, tmp_path):
    # gate 通过但 surface 返回空 → 不注入，但仍标记 consumed（下轮不再重 poll）。
    _patch_memdir(monkeypatch, tmp_path)
    async def _empty(*a, **k):
        return []
    monkeypatch.setattr(qe_mod, "surface_relevant", _empty)
    eng = _engine()
    eng._start_memory_prefetch(UserMessage(content="a real query", source="user"))
    await eng._pending.task
    await eng._consume_prefetch_if_ready(parent_id=None)
    assert eng._pending.consumed is True
    assert [m for m in eng.store._messages if isinstance(m, RelevantMemoryMessage)] == []


async def test_cancel_cancels_running_task(monkeypatch, tmp_path):
    _patch_memdir(monkeypatch, tmp_path)
    async def _slow(*a, **k):
        await asyncio.sleep(10)
        return []
    monkeypatch.setattr(qe_mod, "surface_relevant", _slow)
    eng = _engine()
    eng._start_memory_prefetch(UserMessage(content="a real query", source="user"))
    task = eng._pending.task
    eng._cancel_prefetch()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert eng._pending is None
