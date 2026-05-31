# Memory Prefetch (Layer 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Proactively surface relevant memories each turn via a non-blocking prefetch: fire `find_relevant_memories` when the user message arrives, splice the chosen memories into the conversation when ready, deduped and bounded.

**Architecture:** Three seams in `QueryEngine` (kick-off in `query()`, consume in the per-iteration `get_messages` hook, cancel in `query()`'s finally) drive an `asyncio.Task` running a surfacing coroutine. Surfaced memories are injected as a new structured `RelevantMemoryMessage` (Layer-1, renders to one `<system-reminder>` block per memory inside a single HumanMessage, each block carrying a frozen age-as-words freshness header). Derivation (`already_surfaced`, `recent_tools`, byte budget) is computed from the message store; per-file truncation + `file_read_state` dedup (with mark-after) round it out. Non-blocking by design (adaptive-retrieval rationale; the B-1 index + on-demand Read covers no-tool turns).

**Tech Stack:** Python 3.11, asyncio, pydantic v2, `pytest` (`asyncio_mode=auto`), `uv`.

**Spec:** `docs/superpowers/specs/2026-05-31-memory-prefetch-design.md` (① locked §1–§7; ②-⑥ §9 + this plan).

**Reuses (do not rewrite):** `find_relevant_memories` (`memdir/selector.py`), `scan_memory_files` (`memdir/scan.py`), `MemoryHeader` (`memdir/types.py`), `file_read_state._state` (`state/file_read_state.py`), `_neutralize_reminder_tags` (`memdir/injection.py`).

**Task order is a hard dependency chain:** Task 1 (message type) → Task 2 (freshness + derivation/surfacing, imports the type) → Task 3 (wiring, imports both).

---

## Task 0: Create feature branch

Repo is on `master`. Do not commit on `master`.

- [ ] **Step 1: Branch + commit the design spec (currently untracked)**

```bash
cd "D:/coding projects/build-mini-cc"
git checkout -b feat/memory-prefetch
git add docs/superpowers/specs/2026-05-31-memory-prefetch-design.md docs/superpowers/plans/2026-05-31-memory-prefetch.md
git commit -m "docs: memory prefetch (Layer 2) design + implementation plan"
```
Expected: branch `feat/memory-prefetch` created; 2 files committed.

---

## Task 1: `RelevantMemoryMessage` structured type

**Files:**
- Modify: `src/mini_cc/engine/messages.py` (add `SurfacedMemory` + `RelevantMemoryMessage`, extend `LAYER_1_TYPES`, handle in `to_langchain_single`)
- Modify: `src/mini_cc/memdir/injection.py` (add `render_relevant_memories`)
- Test: `tests/test_relevant_memory_message.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_relevant_memory_message.py`:

```python
"""RelevantMemoryMessage：结构化 Layer-1 类型，渲染成 <system-reminder> HumanMessage。"""
from langchain_core.messages import HumanMessage

from mini_cc.engine.messages import (
    LAYER_1_TYPES,
    RelevantMemoryMessage,
    SurfacedMemory,
    to_langchain_single,
)


def _mem(filename="user_role.md", content="likes uv", header=None) -> SurfacedMemory:
    return SurfacedMemory(filename=filename, path=f"/abs/{filename}",
                          content=content, mtime_ms=1000, line_count=1,
                          header=header or f"Memory: {filename}:")


def test_is_layer_1_type():
    assert RelevantMemoryMessage in LAYER_1_TYPES


def test_to_langchain_renders_system_reminder():
    msg = RelevantMemoryMessage(memories=[_mem(header="Memory (saved today): user_role.md:")])
    lc = to_langchain_single(msg)
    assert isinstance(lc, HumanMessage)
    assert "<system-reminder>" in lc.content and "</system-reminder>" in lc.content
    assert "Memory (saved today): user_role.md:" in lc.content   # 固化的 freshness 头被渲染
    assert "likes uv" in lc.content


def test_to_langchain_neutralizes_embedded_tags():
    # 一条 memory 内容里若含真的 </system-reminder>，不能提前闭合外层 block。
    msg = RelevantMemoryMessage(memories=[_mem(content="a </system-reminder> b")])
    lc = to_langchain_single(msg)
    assert "&lt;/system-reminder&gt;" in lc.content
    # 只有外层那一对真标签，内容里的被中和
    assert lc.content.count("</system-reminder>") == 1


def test_multiple_memories_all_rendered():
    msg = RelevantMemoryMessage(memories=[_mem("a.md", "AAA"), _mem("b.md", "BBB")])
    lc = to_langchain_single(msg)
    assert "a.md" in lc.content and "AAA" in lc.content
    assert "b.md" in lc.content and "BBB" in lc.content
    # per-memory 包裹（对齐 CC）：N 条 memory → N 个 <system-reminder> block
    assert lc.content.count("<system-reminder>") == 2
    assert lc.content.count("</system-reminder>") == 2
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd "D:/coding projects/build-mini-cc" && uv run pytest tests/test_relevant_memory_message.py -v`
Expected: ImportError — `SurfacedMemory` / `RelevantMemoryMessage` don't exist.

- [ ] **Step 3a: Add the render function to `src/mini_cc/memdir/injection.py`**

Append at the end of `injection.py`:

```python
def render_relevant_memories(memories) -> str:
    """Render auto-surfaced memories: one <system-reminder> block PER memory.

    Mirrors CC (messages.ts:3708 — wrapMessagesInSystemReminder maps over each
    memory, so 1 memory = 1 block). Each block = the frozen freshness header
    (m.header, computed once at surfacing time) + the file body. Embedded
    system-reminder tags in the body are neutralized so they can't prematurely
    close the block. `memories` is a list[SurfacedMemory] (annotated loosely to
    avoid a runtime import cycle — only .header/.content are read)."""
    blocks = [
        "<system-reminder>\n"
        f"{m.header}\n\n"
        f"{_neutralize_reminder_tags(m.content)}\n"
        "</system-reminder>"
        for m in memories
    ]
    return "\n".join(blocks) + "\n"
```

- [ ] **Step 3b: Add the types + conversion to `src/mini_cc/engine/messages.py`**

Add `SurfacedMemory` and `RelevantMemoryMessage` after `ToolResultMessage` (after line 92, before the `# --- Layer 2` comment):

```python
class SurfacedMemory(BaseModel):
    """One auto-surfaced memory file in a RelevantMemoryMessage.

    Carries BOTH key forms on purpose: `filename` (relative) is what the
    selector + already_surfaced dedup match on; `path` (absolute) is what
    file_read_state keys on. See prefetch design §"SurfacedMemory 双 key"."""
    filename: str
    path: str
    content: str       # already truncated by read_and_truncate (⑤)
    mtime_ms: int
    line_count: int    # surfaced line count → file_read_state record's `limit`
    header: str        # frozen freshness header (memory_header), computed once
                       # at surfacing time → cache-stable rendered bytes


class RelevantMemoryMessage(Message):
    type: Literal["relevant_memory"] = "relevant_memory"
    memories: list[SurfacedMemory]
```

Extend `LAYER_1_TYPES` (currently line 119):

```python
LAYER_1_TYPES = (SystemPromptMessage, UserMessage, AssistantMessage, ToolResultMessage, RelevantMemoryMessage)
```

Add a branch in `to_langchain_single` (before the final `return None`, after the `ToolResultMessage` branch at line 137-138):

```python
    if isinstance(msg, RelevantMemoryMessage):
        # Lazy import avoids a load-time cycle (injection imports nothing from
        # messages at runtime; this mirrors ToolResultMessage's output_from_dict).
        from mini_cc.memdir.injection import render_relevant_memories
        return HumanMessage(content=render_relevant_memories(msg.memories))
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd "D:/coding projects/build-mini-cc" && uv run pytest tests/test_relevant_memory_message.py -v`
Expected: `4 passed`.

- [ ] **Step 5: Run the full suite (no regressions — LAYER_1_TYPES change is broad)**

Run: `cd "D:/coding projects/build-mini-cc" && uv run pytest -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/mini_cc/engine/messages.py src/mini_cc/memdir/injection.py tests/test_relevant_memory_message.py
git commit -m "feat(memdir): add RelevantMemoryMessage structured type for surfacing"
```

---

## Task 2: Freshness + derivation + surfacing functions

**Files:**
- Create: `src/mini_cc/memdir/age.py` (freshness: age-as-words + frozen header)
- Create: `src/mini_cc/memdir/prefetch.py`
- Test: `tests/test_memory_age.py`
- Test: `tests/test_memdir_prefetch.py`

Two pure-function modules: `age.py` (freshness header — port of CC `memoryAge.ts`
+ `memoryHeader`) and `prefetch.py` (②③④⑤ derivation + the ① surfacing coro).
The header is computed **once at surfacing time** and frozen into
`SurfacedMemory.header`, so the rendered bytes stay stable across turns
(prompt-cache friendly — CC `messages.ts:3711-3715`). No engine wiring here.

### 2A — `age.py` (freshness)

- [ ] **Step 1: Write the failing test**

Create `tests/test_memory_age.py`:

```python
"""age.py：human-readable age + 固化 freshness header（port CC memoryAge.ts）。"""
import time

from mini_cc.memdir.age import (
    memory_age, memory_age_days, memory_freshness_text, memory_header,
)

_DAY = 86_400_000


def _ago(days: int) -> int:
    return int(time.time() * 1000) - days * _DAY


def test_age_words():
    assert memory_age(_ago(0)) == "today"
    assert memory_age(_ago(1)) == "yesterday"
    assert memory_age(_ago(47)) == "47 days ago"


def test_freshness_text_empty_when_fresh():
    assert memory_freshness_text(_ago(0)) == ""
    assert memory_freshness_text(_ago(1)) == ""


def test_freshness_text_present_when_stale():
    t = memory_freshness_text(_ago(47))
    assert "47 days old" in t and "Verify against current code" in t


def test_header_fresh_has_saved_today():
    assert memory_header("user_role.md", _ago(0)) == "Memory (saved today): user_role.md:"


def test_header_stale_has_caveat_and_name():
    h = memory_header("x.md", _ago(47))
    assert h.startswith("This memory is 47 days old.")
    assert h.endswith("Memory: x.md:")


def test_age_days_never_negative():
    # 未来 mtime（时钟偏移）→ age 不为负
    assert memory_age_days(int(time.time() * 1000) + 10 * _DAY) == 0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd "D:/coding projects/build-mini-cc" && uv run pytest tests/test_memory_age.py -v`
Expected: ModuleNotFoundError — `mini_cc.memdir.age` doesn't exist.

- [ ] **Step 3: Write `src/mini_cc/memdir/age.py`**

```python
"""Memory freshness as human-readable age (port of CC memoryAge.ts + memoryHeader).

Models reason poorly about ISO timestamps but well about "47 days ago", which
triggers staleness reasoning. memory_header() is computed once at surfacing time
and frozen into SurfacedMemory.header so the rendered bytes stay stable across
turns (prompt-cache friendly)."""
from __future__ import annotations

import time

_DAY_MS = 86_400_000


def _now_ms() -> int:
    return int(time.time() * 1000)


def memory_age_days(mtime_ms: int) -> int:
    return max(0, (_now_ms() - mtime_ms) // _DAY_MS)


def memory_age(mtime_ms: int) -> str:
    d = memory_age_days(mtime_ms)
    if d == 0:
        return "today"
    if d == 1:
        return "yesterday"
    return f"{d} days ago"


def memory_freshness_text(mtime_ms: int) -> str:
    """Staleness caveat for memories > 1 day old; '' for fresh ones (≤1 day —
    a warning there is just noise)."""
    d = memory_age_days(mtime_ms)
    if d <= 1:
        return ""
    return (
        f"This memory is {d} days old. Memories are point-in-time observations, "
        "not live state — claims about code behavior or file:line citations may "
        "be outdated. Verify against current code before asserting as fact."
    )


def memory_header(filename: str, mtime_ms: int) -> str:
    """Per-memory header line: stale → caveat + 'Memory: <name>:'; fresh →
    'Memory (saved today): <name>:'. Frozen into SurfacedMemory at creation."""
    staleness = memory_freshness_text(mtime_ms)
    if staleness:
        return f"{staleness}\n\nMemory: {filename}:"
    return f"Memory (saved {memory_age(mtime_ms)}): {filename}:"
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd "D:/coding projects/build-mini-cc" && uv run pytest tests/test_memory_age.py -v`
Expected: `6 passed`.

### 2B — `prefetch.py` (derivation + surfacing)

- [ ] **Step 5: Write the failing test**

Create `tests/test_memdir_prefetch.py`:

```python
"""Layer 2 派生 + surfacing 纯函数单测。"""
from mini_cc.engine.messages import (
    AssistantMessage, RelevantMemoryMessage, SurfacedMemory, TextBlock,
    ToolResultMessage, ToolUseBlock, UserMessage,
)
from mini_cc.memdir import prefetch as pf_mod
from mini_cc.memdir.prefetch import (
    MAX_MEMORY_LINES, collect_recent_successful_tools, collect_surfaced,
    read_and_truncate, should_prefetch, surface_relevant,
)
from mini_cc.memdir.types import MemoryHeader, MemoryType
from mini_cc.tools.base import ToolOutput


def _sm(filename="a.md", content="x"):
    return SurfacedMemory(filename=filename, path=f"/abs/{filename}",
                          content=content, mtime_ms=1, line_count=1,
                          header=f"Memory: {filename}:")


def _asst_tool(call_id, name):
    return AssistantMessage(turn_id="t", model="m",
                            content=ToolUseBlock(call_id=call_id, name=name, args={}))


def _result(call_id, *, error):
    return ToolResultMessage(content="r", tool_call_id=call_id,
                             output=ToolOutput(is_error=error))


def test_should_prefetch_skips_single_word():
    assert should_prefetch("hi", 0) is False
    assert should_prefetch("recommend a package manager", 0) is True


def test_should_prefetch_skips_when_budget_full():
    assert should_prefetch("a real query here", 60 * 1024) is False


def test_collect_surfaced_paths_and_bytes():
    msgs = [RelevantMemoryMessage(memories=[_sm("a.md", "AAA"), _sm("b.md", "BB")])]
    names, total = collect_surfaced(msgs)
    assert names == frozenset({"a.md", "b.md"})
    assert total == len("AAA") + len("BB")


def test_collect_surfaced_empty():
    assert collect_surfaced([UserMessage(content="hi", source="user")]) == (frozenset(), 0)


def test_recent_tools_success_minus_failed():
    msgs = [
        UserMessage(id="u1", content="prev", source="user"),
        _asst_tool("c1", "file_edit"), _result("c1", error=False),
        _asst_tool("c2", "execute_command"), _result("c2", error=True),
        AssistantMessage(turn_id="t", model="m", content=TextBlock(text="done")),
        UserMessage(id="u2", content="cur", source="user"),
    ]
    assert collect_recent_successful_tools(msgs, current_user_id="u2") == ("file_edit",)


def test_recent_tools_excludes_tool_that_ever_errored():
    msgs = [UserMessage(id="u1", content="p", source="user"),
            _asst_tool("c1", "file_edit"), _result("c1", error=False),
            _asst_tool("c3", "file_edit"), _result("c3", error=True),
            UserMessage(id="u2", content="c", source="user")]
    assert collect_recent_successful_tools(msgs, current_user_id="u2") == ()


def test_recent_tools_stops_at_previous_user_turn():
    msgs = [_asst_tool("cX", "grep"), _result("cX", error=False),
            UserMessage(id="u0", content="old", source="user"),
            UserMessage(id="u1", content="p", source="user"),
            _asst_tool("c1", "file_edit"), _result("c1", error=False),
            UserMessage(id="u2", content="c", source="user")]
    assert collect_recent_successful_tools(msgs, current_user_id="u2") == ("file_edit",)


def test_read_and_truncate_under_cap(tmp_path):
    f = tmp_path / "m.md"
    f.write_text("line1\nline2\n", encoding="utf-8")
    h = MemoryHeader(filename="m.md", file_path=f, mtime_ms=1.0,
                     description="d", type=MemoryType.USER)
    sm = read_and_truncate(h)
    assert sm is not None and "line1" in sm.content and "truncated" not in sm.content
    assert sm.path == str(f) and sm.filename == "m.md"
    # header 在创建时固化（mtime=1ms → epoch → stale → 带 caveat，以 "Memory: m.md:" 收尾）
    assert sm.header.endswith("Memory: m.md:") and "days old" in sm.header


def test_read_and_truncate_over_line_cap(tmp_path):
    f = tmp_path / "big.md"
    f.write_text("\n".join(f"l{i}" for i in range(MAX_MEMORY_LINES + 50)), encoding="utf-8")
    h = MemoryHeader(filename="big.md", file_path=f, mtime_ms=1.0, description=None, type=None)
    sm = read_and_truncate(h)
    assert sm is not None and "truncated" in sm.content
    assert sm.line_count == MAX_MEMORY_LINES


def test_read_and_truncate_missing_file_returns_none(tmp_path):
    h = MemoryHeader(filename="gone.md", file_path=tmp_path / "gone.md",
                     mtime_ms=1.0, description=None, type=None)
    assert read_and_truncate(h) is None


async def test_surface_relevant_reads_selected(tmp_path, monkeypatch):
    f = tmp_path / "user_role.md"
    f.write_text("likes uv", encoding="utf-8")
    h = MemoryHeader(filename="user_role.md", file_path=f, mtime_ms=1.0,
                     description="d", type=MemoryType.USER)

    async def _fake_find(query, memdir, *, recent_tools=(), already_surfaced=frozenset()):
        return [h]
    monkeypatch.setattr(pf_mod, "find_relevant_memories", _fake_find)

    out = await surface_relevant("q", tmp_path)
    assert len(out) == 1
    assert out[0].filename == "user_role.md" and "likes uv" in out[0].content
    assert out[0].path == str(f)
    assert out[0].header.endswith("user_role.md:")   # 固化 header 一并产出
```

- [ ] **Step 6: Run the test to verify it fails**

Run: `cd "D:/coding projects/build-mini-cc" && uv run pytest tests/test_memdir_prefetch.py -v`
Expected: ModuleNotFoundError — `mini_cc.memdir.prefetch` doesn't exist.

- [ ] **Step 7: Write the implementation**

Create `src/mini_cc/memdir/prefetch.py`:

```python
"""Layer 2 派生 + surfacing 函数（纯计算 + 一个 surfacing coro）。

CC 对照：collectSurfacedMemories / collectRecentSuccessfulTools /
readMemoriesForSurfacing。异步 prefetch 的生命周期接线在 query_engine.py。
"""
from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from mini_cc.engine.messages import (
    AssistantMessage, Message, RelevantMemoryMessage, SurfacedMemory,
    ToolResultMessage, ToolUseBlock, UserMessage,
)
from mini_cc.memdir.age import memory_header
from mini_cc.memdir.selector import find_relevant_memories
from mini_cc.memdir.types import MemoryHeader

MAX_SESSION_BYTES = 60 * 1024   # ④ CC RELEVANT_MEMORIES_CONFIG.MAX_SESSION_BYTES
MAX_MEMORY_LINES = 200          # ⑤ CC surfacing 上限（≠ index 的 25KB）
MAX_MEMORY_BYTES = 4096


@dataclass
class PrefetchHandle:
    task: asyncio.Task
    consumed: bool = False


def should_prefetch(user_text: str, surfaced_bytes: int) -> bool:
    """Gate：session 字节预算满（④）或单词 query（无语义）→ 跳过。"""
    if surfaced_bytes >= MAX_SESSION_BYTES:
        return False
    return len(user_text.split()) >= 2


def collect_surfaced(messages: Sequence[Message]) -> tuple[frozenset[str], int]:
    """② already_surfaced（相对 filename 集）+ ④ 已注入总字节。

    扫 store 历史 RelevantMemoryMessage —— compact 删掉它们后自然重置。"""
    names: set[str] = set()
    total = 0
    for m in messages:
        if isinstance(m, RelevantMemoryMessage):
            for mem in m.memories:
                names.add(mem.filename)
                total += len(mem.content)
    return frozenset(names), total


def collect_recent_successful_tools(
    messages: Sequence[Message], current_user_id: str
) -> tuple[str, ...]:
    """③ 上一轮"成功且从未报错"的工具名（selector 反噪声信号）。

    name 在 AssistantMessage(ToolUseBlock)，成败在 ToolResultMessage.output.is_error，
    靠 call_id join。倒序扫到上一个真实 user（id ≠ 当前）为止。"""
    use_to_name: dict[str, str] = {}
    errored: dict[str, bool] = {}
    for m in reversed(list(messages)):
        if isinstance(m, UserMessage) and not m.is_synthetic and m.id != current_user_id:
            break
        if isinstance(m, AssistantMessage) and isinstance(m.content, ToolUseBlock):
            use_to_name[m.content.call_id] = m.content.name
        elif isinstance(m, ToolResultMessage):
            errored[m.tool_call_id] = bool(m.output is not None and m.output.is_error)
    succeeded = {n for cid, n in use_to_name.items() if errored.get(cid) is False}
    failed = {n for cid, n in use_to_name.items() if errored.get(cid) is True}
    return tuple(succeeded - failed)


def read_and_truncate(h: MemoryHeader) -> SurfacedMemory | None:
    """⑤ 读 memory 文件，截到 200 行 / 4KB，附截断提示。OSError → None。"""
    try:
        lines = h.file_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    truncated = len(lines) > MAX_MEMORY_LINES
    body = "\n".join(lines[:MAX_MEMORY_LINES])
    if len(body.encode("utf-8")) > MAX_MEMORY_BYTES:
        body = body.encode("utf-8")[:MAX_MEMORY_BYTES].decode("utf-8", "ignore")
        truncated = True
    if truncated:
        body += f"\n\n> [truncated — use file_read for the full file: {h.file_path}]"
    mtime = int(h.mtime_ms)
    return SurfacedMemory(
        filename=h.filename, path=str(h.file_path), content=body,
        mtime_ms=mtime, line_count=min(len(lines), MAX_MEMORY_LINES),
        header=memory_header(h.filename, mtime),   # ⑤+freshness：创建时固化 header
    )


async def surface_relevant(
    query: str, memdir: Path, *,
    recent_tools: Sequence[str] = (),
    already_surfaced: frozenset[str] = frozenset(),
) -> list[SurfacedMemory]:
    """① 的 coro：选 top-N（Layer 1）→ 读+截断（⑤）→ 结构化 SurfacedMemory。"""
    headers = await find_relevant_memories(
        query, memdir, recent_tools=recent_tools, already_surfaced=already_surfaced)
    out: list[SurfacedMemory] = []
    for h in headers:
        sm = read_and_truncate(h)
        if sm is not None:
            out.append(sm)
    return out
```

- [ ] **Step 8: Run the test to verify it passes**

Run: `cd "D:/coding projects/build-mini-cc" && uv run pytest tests/test_memdir_prefetch.py -v`
Expected: `12 passed`.

- [ ] **Step 9: Commit (age.py + prefetch.py together)**

```bash
git add src/mini_cc/memdir/age.py tests/test_memory_age.py \
        src/mini_cc/memdir/prefetch.py tests/test_memdir_prefetch.py
git commit -m "feat(memdir): add freshness (age.py) + prefetch derivation/surfacing functions"
```

---

## Task 3: Wiring (`query_engine.py` — three seams)

**Files:**
- Modify: `src/mini_cc/engine/query_engine.py`
- Test: `tests/test_prefetch_wiring.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_prefetch_wiring.py`:

```python
"""Prefetch wiring：kick-off / consume / cancel 三处 seam（不打 API）。"""
import asyncio
from unittest.mock import MagicMock

import pytest

from mini_cc.engine import query_engine as qe_mod
from mini_cc.engine.messages import RelevantMemoryMessage, SurfacedMemory, UserMessage
from mini_cc.engine.query_engine import QueryEngine


def _engine() -> QueryEngine:
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
    await eng._pending.task                                # 等 prefetch 完成
    await eng._consume_prefetch_if_ready(parent_id=None)

    injected = [m for m in eng.store._messages if isinstance(m, RelevantMemoryMessage)]
    assert len(injected) == 1 and eng._pending.consumed is True
    assert "/abs/user_role.md" in recorded                # ⑥ mark-after


async def test_consume_noop_when_not_ready(monkeypatch, tmp_path):
    _patch_memdir(monkeypatch, tmp_path)
    async def _slow(*a, **k):
        await asyncio.sleep(10)
        return []
    monkeypatch.setattr(qe_mod, "surface_relevant", _slow)
    eng = _engine()
    eng._start_memory_prefetch(UserMessage(content="a real query", source="user"))
    await eng._consume_prefetch_if_ready(parent_id=None)  # task 未 done
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
    await eng._consume_prefetch_if_ready(parent_id="sidechain-id")   # 非 main branch
    assert [m for m in eng.store._messages if isinstance(m, RelevantMemoryMessage)] == []


async def test_consume_filters_already_read(monkeypatch, tmp_path):
    _patch_memdir(monkeypatch, tmp_path)
    async def _fake_surface(*a, **k):
        return [_sm()]
    monkeypatch.setattr(qe_mod, "surface_relevant", _fake_surface)
    monkeypatch.setattr(qe_mod.file_read_state._state, "get", lambda p: object())  # 已读
    eng = _engine()
    eng._start_memory_prefetch(UserMessage(content="a real query", source="user"))
    await eng._pending.task
    await eng._consume_prefetch_if_ready(parent_id=None)
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd "D:/coding projects/build-mini-cc" && uv run pytest tests/test_prefetch_wiring.py -v`
Expected: AttributeError — `_start_memory_prefetch` / `_pending` don't exist.

- [ ] **Step 3a: Add imports to `src/mini_cc/engine/query_engine.py`**

Add `import asyncio` near the top imports (after `import sys`, line 25). Add `RelevantMemoryMessage` to the existing `from mini_cc.engine.messages import (...)` block. Add after the existing memdir imports (line 56-57):

```python
from mini_cc.memdir.prefetch import (
    PrefetchHandle, collect_recent_successful_tools, collect_surfaced,
    should_prefetch, surface_relevant,
)
from mini_cc.state import file_read_state
```

- [ ] **Step 3b: Add `_pending` state in `__init__`**

After `self.store = MessageStore()` (line 103), add:

```python
        self._pending: PrefetchHandle | None = None
```

- [ ] **Step 3c: Kick off + cancel in `query()`**

In `query()` (line 280-333), replace the first dispatch line:

```python
        await self._dispatch(UserMessage(content=user_text, source="user"))
```
with:
```python
        um = UserMessage(content=user_text, source="user")
        await self._dispatch(um)
        self._start_memory_prefetch(um)
```

And in the `finally:` block (line 332-333), add the cancel before the turn_end dispatch:

```python
        finally:
            self._cancel_prefetch()
            await self._dispatch(StatusMessage(event="turn_end", source="agent"))
```

- [ ] **Step 3d: Route the per-iteration hook through consume**

In `run_loop()` (line 264-269), change the `get_messages` arg:

```python
        async for msg in loop.run(
            get_messages=lambda: self._turn_pre_call(parent_id),
            parent_id=parent_id,
            source=source,
            sandbox=sandbox,
        ):
```

- [ ] **Step 3e: Add the four new methods**

Add to the `QueryEngine` class (e.g. after `_prepare_messages`, before `_clear_old_tool_results`):

```python
    # -- memory prefetch (Layer 2) -----------------------------------------

    def _start_memory_prefetch(self, user_msg: UserMessage) -> None:
        """Fire the non-blocking relevant-memory prefetch for this turn.
        Gated; sets self._pending = None when skipped. recent_tools /
        already_surfaced / byte-budget all derived from the store."""
        msgs = self.store.all()
        surfaced_names, surfaced_bytes = collect_surfaced(msgs)
        if not should_prefetch(user_msg.content, surfaced_bytes):
            self._pending = None
            return
        recent = collect_recent_successful_tools(msgs, user_msg.id)
        coro = surface_relevant(
            user_msg.content, get_auto_mem_path(),
            recent_tools=recent, already_surfaced=surfaced_names)
        self._pending = PrefetchHandle(task=asyncio.create_task(coro))

    async def _turn_pre_call(self, parent_id: str | None) -> list[BaseMessage]:
        """get_messages hook: consume the prefetch if ready, then build view."""
        await self._consume_prefetch_if_ready(parent_id)
        return await self._prepare_messages(parent_id=parent_id)

    async def _consume_prefetch_if_ready(self, parent_id: str | None) -> None:
        """Non-blocking poll. If the prefetch settled (and we're on the main
        branch), inject surviving memories as a RelevantMemoryMessage and mark
        them seen in file_read_state (⑥ mark-after). Consume at most once."""
        if parent_id is not None:                       # sidechain 不注入
            return
        h = self._pending
        if h is None or h.consumed or not h.task.done():
            return
        try:
            surfaced = h.task.result()
        except Exception:                               # prefetch 失败/取消 → best-effort
            h.consumed = True
            return
        fresh = [m for m in surfaced if file_read_state._state.get(m.path) is None]  # ⑥ filter
        if fresh:
            await self._dispatch(RelevantMemoryMessage(memories=fresh, source="memory"))
            for m in fresh:                             # ⑥ mark-after（先 filter 后 record）
                file_read_state._state.record(
                    m.path, m.content, m.mtime_ms, offset=1, limit=m.line_count)
        h.consumed = True

    def _cancel_prefetch(self) -> None:
        """Symbol.dispose 的 Python 替代：回合退出时取消未完成的 prefetch。"""
        if self._pending is not None and not self._pending.task.done():
            self._pending.task.cancel()
        self._pending = None
```

- [ ] **Step 4: Run the wiring test to verify it passes**

Run: `cd "D:/coding projects/build-mini-cc" && uv run pytest tests/test_prefetch_wiring.py -v`
Expected: `6 passed`.

- [ ] **Step 5: Run the FULL suite (no regressions — query_engine is core)**

Run: `cd "D:/coding projects/build-mini-cc" && uv run pytest -q`
Expected: all pass (existing + Layer-1 selector + the new Layer-2 tests).

- [ ] **Step 6: Commit**

```bash
git add src/mini_cc/engine/query_engine.py tests/test_prefetch_wiring.py
git commit -m "feat(engine): wire non-blocking memory prefetch into the turn loop"
```

---

## Notes / out of scope

- **Non-blocking by design** (D-NONBLOCK): the first LLM call / no-tool turns may not see surfaced memory — covered by the B-1 index + on-demand Read (adaptive-retrieval rationale, design §2). Strengthening `WHEN_TO_ACCESS` prompt instructions is the lever to reduce the gap, not blocking.
- **Freshness done here** (pulled forward from L4): per-memory age-as-words header via `memdir/age.py` (`memory_age`/`memory_freshness_text`/`memory_header`), **frozen at surfacing time** into `SurfacedMemory.header` so rendered bytes are cache-stable (CC `messages.ts:3711-3715`). Render uses per-memory `<system-reminder>` wrapping (1 memory = 1 block), aligning with CC `messages.ts:3708`.
- **Deferred** (not this plan): `file_read_state` staleness re-surfacing (consume filter currently "seen → skip"); a finer retrieve-or-not gate beyond single-word/byte-budget.
- **No push.** Local commits only; pushing `feat/memory-prefetch` needs explicit user confirmation.

## Self-Review

1. **Spec coverage** (`2026-05-31-memory-prefetch-design.md` §1–§9): ① wiring → Task 3 (`_start`/`_turn_pre_call`/`_consume`/`_cancel`). D-ATTACH structured type → Task 1. ② already_surfaced + ④ byte budget → `collect_surfaced` (Task 2) + `should_prefetch`. ③ recent_tools → `collect_recent_successful_tools` (Task 2). ⑤ truncation → `read_and_truncate` (Task 2). ⑥ filter+mark-after → Task 3 `_consume`. SurfacedMemory dual-key → Task 1. Per-memory `<system-reminder>` wrapping + frozen age-as-words freshness header → Task 1 `render_relevant_memories` + Task 2A `age.py` + Task 2B `read_and_truncate` (header computed once, stored). All covered.
2. **Placeholder scan:** no TBD/TODO; every code step has full code; every run step has expected output. Clean.
3. **Type consistency:** `SurfacedMemory(filename, path, content, mtime_ms, line_count, header)` identical in Task 1 (def + `render_relevant_memories` reads `.header`/`.content`), Task 2 (`read_and_truncate` builds it incl. `header=memory_header(filename, mtime)`, `collect_surfaced` reads `.filename`/`.content`), Task 3 (`_consume` reads `.path`/`.content`/`.mtime_ms`/`.line_count`). `memory_header(filename, mtime_ms) -> str` (Task 2A def) is called only in `read_and_truncate` (Task 2B import); every test helper (`_mem` Task 1, `_sm` Task 2/3) passes `header`. `surface_relevant(query, memdir, *, recent_tools, already_surfaced)` matches Task 2 def + Task 3 call. `collect_surfaced -> (frozenset, int)`, `collect_recent_successful_tools(msgs, current_user_id) -> tuple`, `should_prefetch(text, bytes) -> bool`, `PrefetchHandle(task, consumed)` — all consistent across Task 2 def and Task 3 use. `find_relevant_memories(query, memdir, *, recent_tools, already_surfaced)` matches the shipped Layer-1 signature.

