# Dual-Channel Memory Injection (方案 Y) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让模型每轮都看到 MEMORY.md 索引——通过 boot 时把索引注入 messages[0] 的 `<system-reminder>` context message（通道 B-1），并用 api_view 边界合并连续 HumanMessage 来满足 DeepSeek-v4-pro 不支持连续同 role 的约束。

**Architecture:** 通道 A（memory 行为指令，已在 system prompt）保持不变。新增通道 B-1：boot 时 `build_memory_context()`（内层，生成 `Contents of {path} (desc):\n\n{索引}` value）+ `render_user_context()`（外层，遍历 context dict 渲染 `# {key}\n{value}` + `<system-reminder>` 包装，key 数据驱动）→ 作为 `is_synthetic=True` 的 UserMessage dispatch 到 store。方案 C：`api_view` 在合并 assistant 之后，新增 post-pass 合并连续 HumanMessage，store 保留真实语义、合并隔离在 API 边界（= CC `normalizeMessagesForAPI` 哲学），同时去掉 task_state 既有的"配对假 assistant"。

**Tech Stack:** Python 3.11, pydantic v2, langchain_core messages (HumanMessage/AIMessage/ToolMessage/SystemMessage), pytest. 测试用 `tmp_path` + `monkeypatch` + `asyncio.run()`，不需要 LLM mock（boot 只 dispatch、injection/truncate/merge 是纯函数）。

**Task 依赖顺序:** Task 1（B-0，独立）→ Task 2（truncate）→ Task 3（injection，用 truncate）→ Task 4（api_view 合并，独立）→ Task 5（boot 注入，用 injection）→ Task 6（端到端 + 回归）。

---

### Task 1: B-0 — 主 system prompt 加 `<system-reminder>` 协议声明

**Files:**
- Modify: `src/mini_cc/prompts.py:75-83`（`agent_section`）
- Test: `tests/test_prompts_protocol.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_prompts_protocol.py
"""B-0: the main system prompt must declare the <system-reminder> protocol,
so the model treats injected context messages (channel B-1) as system
metadata rather than user speech. CC parity: prompts.ts:190."""
from mini_cc.prompts import build_system_prompt


def test_system_prompt_declares_system_reminder_protocol():
    prompt = build_system_prompt(available_tools=set())
    assert "<system-reminder>" in prompt
    assert "information from the system" in prompt
    assert "bear no direct relation" in prompt
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd "D:\coding projects\build-mini-cc" && uv run pytest tests/test_prompts_protocol.py -v`
Expected: FAIL — current `agent_section` has no `<system-reminder>` text.

- [ ] **Step 3: Add the protocol declaration to `agent_section`**

In `src/mini_cc/prompts.py`, replace the `agent_section` block (currently lines 75-83):

```python
    # ── Section 1: # Agent ────────────────────────────────────────────────
    agent_section = "\n\n".join([
        "# Agent",
        f"You are a coding agent in: {config.CWD}\n{_platform_line()}",
        (
            "Answer questions and chat normally without tools. Use tools only when "
            "the task actually requires running code, reading files, or taking action "
            "— not for simple questions or greetings."
        ),
        # B-0: declare the <system-reminder> protocol ONCE here. Channel B-1
        # (memory index) and any future injected context reuse this contract
        # instead of re-declaring it. Without this, the model can mistake the
        # injected context message for something the user actually typed.
        (
            "Tool results and user messages may include <system-reminder> or other "
            "tags. Tags contain information from the system. They bear no direct "
            "relation to the specific tool results or user messages in which they appear."
        ),
    ])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_prompts_protocol.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/mini_cc/prompts.py tests/test_prompts_protocol.py
git commit -m "feat(memory): declare <system-reminder> protocol in system prompt (B-0)"
```

---

### Task 2: `truncate_entrypoint` — MEMORY.md 索引截断

**Files:**
- Create: `src/mini_cc/memdir/truncate.py`
- Test: `tests/test_memdir_retrievable.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_memdir_retrievable.py
"""可取 (retrievable) attribute: 双通道 messages[0] 注入 + 方案C 合并."""
from mini_cc.memdir.truncate import truncate_entrypoint, MAX_ENTRYPOINT_LINES


def test_truncate_under_caps_returns_trimmed():
    assert truncate_entrypoint("- a\n- b\n") == "- a\n- b"


def test_truncate_over_line_cap_appends_warning():
    raw = "\n".join(f"- line{i}" for i in range(MAX_ENTRYPOINT_LINES + 50))
    out = truncate_entrypoint(raw)
    # Partition on the full "\n\n> WARNING" separator so the blank line isn't
    # counted as content; `content` is then exactly the truncated index.
    content, sep, warning = out.partition("\n\n> WARNING")
    assert sep  # warning block present, with its blank-line separator
    assert content.count("\n") + 1 <= MAX_ENTRYPOINT_LINES  # N lines = N-1 newlines
    assert str(MAX_ENTRYPOINT_LINES) in warning


def test_truncate_over_byte_cap_under_line_cap():
    # 10 long lines that blow the 25KB byte cap while staying under 200 lines.
    raw = "\n".join("x" * 4000 for _ in range(10))
    out = truncate_entrypoint(raw)
    assert "> WARNING" in out
    assert len(out.encode("utf-8")) < 26_000
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_memdir_retrievable.py -v`
Expected: FAIL — `ModuleNotFoundError: mini_cc.memdir.truncate`

- [ ] **Step 3: Implement `truncate.py` (复刻 CC `truncateEntrypointContent`, why 注释)**

```python
# src/mini_cc/memdir/truncate.py
"""Truncate MEMORY.md index content to line + byte caps.

Why two caps (mirrors CC memdir.ts:truncateEntrypointContent):
  - line cap (200) bounds a normal index — one entry per line.
  - byte cap (25KB) catches the pathological case the line cap misses:
    a few very long lines. CC observed a 197KB index that was still under
    200 lines. Without the byte cap that whole 197KB would hit every
    request as cache-prefix tokens.
Line-truncate first (a natural boundary), THEN byte-truncate at the last
newline before the cap, so a line is never cut mid-way.
"""
from __future__ import annotations

MAX_ENTRYPOINT_LINES = 200
MAX_ENTRYPOINT_BYTES = 25_000


def truncate_entrypoint(raw: str) -> str:
    trimmed = raw.strip()
    lines = trimmed.split("\n")
    line_count = len(lines)
    # Measure the ORIGINAL byte count: long lines are exactly the failure
    # mode the byte cap targets, so measuring after line-truncation would
    # understate the size and suppress the warning.
    byte_count = len(trimmed.encode("utf-8"))

    was_line_truncated = line_count > MAX_ENTRYPOINT_LINES
    was_byte_truncated = byte_count > MAX_ENTRYPOINT_BYTES

    if not was_line_truncated and not was_byte_truncated:
        return trimmed

    truncated = (
        "\n".join(lines[:MAX_ENTRYPOINT_LINES]) if was_line_truncated else trimmed
    )

    if len(truncated.encode("utf-8")) > MAX_ENTRYPOINT_BYTES:
        # Slice on bytes, then back up to the last newline so we don't split
        # a UTF-8 line; decode with errors="ignore" to drop any partial
        # multibyte char left at the cut.
        head = truncated.encode("utf-8")[:MAX_ENTRYPOINT_BYTES]
        cut = head.rfind(b"\n")
        head = head[:cut] if cut > 0 else head
        truncated = head.decode("utf-8", errors="ignore")

    if was_byte_truncated and not was_line_truncated:
        reason = f"{byte_count} bytes (limit: {MAX_ENTRYPOINT_BYTES}) — index entries are too long"
    elif was_line_truncated and not was_byte_truncated:
        reason = f"{line_count} lines (limit: {MAX_ENTRYPOINT_LINES})"
    else:
        reason = f"{line_count} lines and {byte_count} bytes"

    return (
        truncated
        + f"\n\n> WARNING: MEMORY.md is {reason}. Only part of it was loaded. "
        "Keep index entries to one line under ~200 chars; move detail into topic files."
    )
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_memdir_retrievable.py -v`
Expected: PASS (3 truncate tests)

- [ ] **Step 5: Commit**

```bash
git add src/mini_cc/memdir/truncate.py tests/test_memdir_retrievable.py
git commit -m "feat(memory): add truncate_entrypoint for MEMORY.md index (D1/D3)"
```

---

### Task 3: `injection.py` — `build_memory_context` + `render_user_context`（两层）

**Files:**
- Create: `src/mini_cc/memdir/injection.py`
- Test: `tests/test_memdir_retrievable.py`

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_memdir_retrievable.py
from pathlib import Path

from mini_cc.memdir.injection import build_memory_context, render_user_context


def _write_memory_index(memdir: Path, body: str) -> None:
    memdir.mkdir(parents=True, exist_ok=True)
    (memdir / "MEMORY.md").write_text(body, encoding="utf-8")


def test_build_memory_context_returns_none_when_missing(tmp_path):
    # No MEMORY.md written → None (缺失=缺省, 学 CC if file.content)
    assert build_memory_context(tmp_path) is None


def test_build_memory_context_returns_none_when_empty(tmp_path):
    _write_memory_index(tmp_path, "   \n  ")
    assert build_memory_context(tmp_path) is None


def test_build_memory_context_wraps_index_as_value(tmp_path):
    _write_memory_index(tmp_path, "- [User Role](user_role.md) — Java engineer")
    value = build_memory_context(tmp_path)
    assert value is not None
    # Inner format: Contents of {abs path} ({desc}):\n\n{index}
    assert str((tmp_path / "MEMORY.md")) in value
    assert "persists across conversations" in value
    assert "Java engineer" in value


def test_build_memory_context_neutralizes_reminder_tags(tmp_path):
    # An index that literally mentions the tag (very likely here — this very
    # project's memory is ABOUT memory internals) must not close the block.
    _write_memory_index(tmp_path, "- note: a literal </system-reminder> in the index")
    value = build_memory_context(tmp_path)
    assert value is not None
    assert "</system-reminder>" not in value          # neutralized in the value
    assert "&lt;/system-reminder&gt;" in value


def test_render_user_context_empty_dict_returns_empty():
    assert render_user_context({}) == ""


def test_render_user_context_iterates_keys_data_driven():
    out = render_user_context({"memory": "MEM-VALUE", "currentDate": "Today's date is 2026-05-27."})
    assert "<system-reminder>" in out and "</system-reminder>" in out
    # key 数据驱动: 每个 key 渲染成 '# {key}', 顺序保持
    assert "# memory\nMEM-VALUE" in out
    assert "# currentDate\nToday's date is 2026-05-27." in out
    assert out.index("# memory") < out.index("# currentDate")
    assert "IMPORTANT: this context may or may not be relevant" in out
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_memdir_retrievable.py -v`
Expected: FAIL — `ModuleNotFoundError: mini_cc.memdir.injection`

- [ ] **Step 3: Implement `injection.py` (两层, 对照 CC getClaudeMds + prependUserContext)**

```python
# src/mini_cc/memdir/injection.py
"""Channel B-1: inject the MEMORY.md index into messages[0] as a
<system-reminder> context message.

Two layers, mirroring CC's split:
  build_memory_context  ↔  getClaudeMds (claudemd.ts:1185) — inner value
  render_user_context   ↔  prependUserContext (api.ts:461) — outer wrapper

Keeping them separate is what makes the wrapper data-driven: the caller
(boot) assembles a {key: value} dict, and render_user_context iterates it.
Adding gitStatus/userEmail later = one more dict key, zero changes here.
"""
from __future__ import annotations

from pathlib import Path

from mini_cc.memdir.truncate import truncate_entrypoint

# Fixed description string, like CC's per-type `description` (claudemd.ts:1176).
# It is content-about-content, not data — so it lives here, not in the dict.
_MEMORY_DESC = "your persistent memory, persists across conversations"


def _neutralize_reminder_tags(text: str) -> str:
    """Stop index content that literally contains a system-reminder tag from
    prematurely closing the injected <system-reminder> block. Escape the
    angle brackets so the model reads it as literal text, not structure.
    Not hypothetical: this project's own memory is ABOUT memory internals,
    so '</system-reminder>' really can appear in the index."""
    return (
        text.replace("</system-reminder>", "&lt;/system-reminder&gt;")
            .replace("<system-reminder>", "&lt;system-reminder&gt;")
    )


def build_memory_context(memory_dir: Path) -> str | None:
    """Inner layer: produce the `memory` value, or None if there's nothing
    to inject. None (not "") so the caller's `if value:` drops the key
    entirely — an empty index must not emit a bare `# memory` heading."""
    entrypoint = memory_dir / "MEMORY.md"
    try:
        raw = entrypoint.read_text(encoding="utf-8")
    except (FileNotFoundError, NotADirectoryError):
        return None
    # Neutralize AFTER truncate so we escape only what actually gets injected.
    content = _neutralize_reminder_tags(truncate_entrypoint(raw))
    if not content.strip():
        return None
    return f"Contents of {entrypoint} ({_MEMORY_DESC}):\n\n{content}"


def render_user_context(context: dict[str, str]) -> str:
    """Outer layer: wrap a {key: value} dict into one <system-reminder>
    block, iterating keys in insertion order. Empty dict → "" so the caller
    can skip dispatching an empty message."""
    if not context:
        return ""
    body = "\n".join(f"# {key}\n{value}" for key, value in context.items())
    return (
        "<system-reminder>\n"
        "As you answer the user's questions, you can use the following context:\n"
        f"{body}\n\n"
        "IMPORTANT: this context may or may not be relevant to your tasks. You should "
        "not respond to this context unless it is highly relevant to your task.\n"
        "</system-reminder>\n"
    )
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_memdir_retrievable.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/mini_cc/memdir/injection.py tests/test_memdir_retrievable.py
git commit -m "feat(memory): add build_memory_context + render_user_context (B-1, two-layer)"
```

---

### Task 4: `api_view` 合并连续 HumanMessage（方案 C）

**Files:**
- Modify: `src/mini_cc/engine/store.py`（`api_view` 末尾 + 新增 `_merge_consecutive_human`）
- Modify: `src/mini_cc/engine/compact.py:98-101`（`_group_by_api_round` docstring）
- Test: `tests/test_store_merge.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_store_merge.py
"""方案 C: api_view merges consecutive HumanMessages at the API boundary
so DeepSeek-v4-pro never sees two user turns in a row. ToolMessages
(role=tool) must NOT be swept into the merge."""
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from mini_cc.engine.store import _merge_consecutive_human


def test_merge_two_consecutive_humans():
    out = _merge_consecutive_human([HumanMessage(content="a"), HumanMessage(content="b")])
    assert len(out) == 1
    assert out[0].content == "a\n\nb"


def test_merge_preserves_order_and_breaks_on_ai():
    out = _merge_consecutive_human([
        HumanMessage(content="ctx"),
        HumanMessage(content="q1"),
        AIMessage(content="a1"),
        HumanMessage(content="q2"),
    ])
    assert [type(m).__name__ for m in out] == ["HumanMessage", "AIMessage", "HumanMessage"]
    assert out[0].content == "ctx\n\nq1"
    assert out[2].content == "q2"


def test_tool_message_is_not_merged():
    # ToolMessage sits between an AI tool call and the next human turn; it
    # must stay its own message (role=tool), never merged into a human.
    msgs = [
        AIMessage(content="calling"),
        ToolMessage(content="result", tool_call_id="t1"),
        HumanMessage(content="next"),
    ]
    out = _merge_consecutive_human(msgs)
    assert [type(m).__name__ for m in out] == ["AIMessage", "ToolMessage", "HumanMessage"]


def test_single_and_empty():
    assert _merge_consecutive_human([]) == []
    one = [HumanMessage(content="solo")]
    assert _merge_consecutive_human(one)[0].content == "solo"
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_store_merge.py -v`
Expected: FAIL — `ImportError: cannot import name '_merge_consecutive_human'`

- [ ] **Step 3: Add `_merge_consecutive_human` and call it from `api_view`**

In `src/mini_cc/engine/store.py`, add the helper near `api_view` and change the final `return out` to run it:

```python
def _merge_consecutive_human(msgs: list[BaseMessage]) -> list[BaseMessage]:
    """Merge runs of consecutive HumanMessages into one (content joined by
    "\\n\\n"). DeepSeek-v4-pro rejects successive same-role turns; this is
    the boundary-level fix (方案 C) so the store can keep synthetic context
    and real input as separate, honest messages. Only HumanMessage merges —
    ToolMessage (role=tool) and AIMessage pass through untouched, so a
    tool_result is never folded into a user turn.

    ASSUMPTION (serial input): today every consecutive-Human run is
    {synthetic context}* + {one real user input}, because mini-cc processes
    one user turn to completion before accepting the next. If concurrent /
    queued input is added later, two *real* user turns could land in a row
    and wrongly merge. EXTENSION POINT: when that feature lands, do the merge
    at the store layer (mini-cc Message still carries is_synthetic) or pass a
    'mergeable' predicate here, so only synthetic-adjacent runs collapse.

    ASSUMPTION (str content): the join assumes HumanMessage.content is str
    (true while UserMessage.content is str). Channel B-2 multimodal
    attachments (list content) would break the '+' — out of scope now (L5)."""
    out: list[BaseMessage] = []
    for m in msgs:
        if isinstance(m, HumanMessage) and out and isinstance(out[-1], HumanMessage):
            out[-1] = HumanMessage(content=out[-1].content + "\n\n" + m.content)
        else:
            out.append(m)
    return out
```

Then in `api_view`, change the last line from `return out` to:

```python
        flush()
        return _merge_consecutive_human(out)
```

Ensure `HumanMessage` is imported at the top of `store.py` (it is used by
`_merge_consecutive_human`); add to the existing langchain_core import if missing:

```python
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
```

- [ ] **Step 4: Update the `_group_by_api_round` docstring in `compact.py:98-101`**

```python
    """Split a LangChain message list into API rounds.

    api_view() already merges consecutive AssistantMessages by turn_id AND
    consecutive HumanMessages (方案 C boundary merge), so each AIMessage in
    the input represents exactly one assistant round, and the preamble holds
    a single merged Human turn (never multiple). That makes the boundary
    trivial: any AIMessage after a non-empty `current` buffer flushes it.
    """
```

- [ ] **Step 5: Run tests + commit**

Run: `uv run pytest tests/test_store_merge.py -v`
Expected: PASS

```bash
git add src/mini_cc/engine/store.py src/mini_cc/engine/compact.py tests/test_store_merge.py
git commit -m "feat(memory): merge consecutive HumanMessages in api_view (方案 C)"
```

---

### Task 5: boot 注入 context message + 去 task_state 配对

**Files:**
- Modify: `src/mini_cc/engine/query_engine.py`（`boot`, 现 193-212）
- Test: `tests/test_boot_memory_injection.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_boot_memory_injection.py
"""boot injects the channel B-1 context message and drops the task_state
paired 'Acknowledged' assistant. boot only dispatches to the store (no LLM
call), so a MagicMock llm_base + asyncio.run() is enough."""
import asyncio
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from mini_cc.engine import query_engine as qe_mod
from mini_cc.engine.query_engine import QueryEngine


def _engine() -> QueryEngine:
    return QueryEngine(
        llm_base=MagicMock(),
        main_tools=[],
        sub_tools=[],
        model_name="test-model",
        system_prompt_builder=lambda: "SYS",
    )


def test_boot_injects_memory_context(monkeypatch, fresh_tasks):
    # Patch where boot looks it up (function-local import target).
    monkeypatch.setattr(qe_mod, "build_memory_context",
                        lambda _dir: "Contents of MEMORY.md (...):\n\n- [Role](r.md) — x")
    engine = _engine()
    asyncio.run(engine.boot())
    view = engine.store.api_view()
    text = "\n".join(m.content for m in view if isinstance(m, HumanMessage))
    assert "Contents of MEMORY.md" in text
    assert "# memory" in text
    assert "# currentDate" in text
    assert "Today's date is" in text


def test_boot_skips_memory_when_none(monkeypatch, fresh_tasks):
    monkeypatch.setattr(qe_mod, "build_memory_context", lambda _dir: None)
    engine = _engine()
    asyncio.run(engine.boot())
    view = engine.store.api_view()
    human = "\n".join(m.content for m in view if isinstance(m, HumanMessage))
    assert "Contents of MEMORY.md" not in human
    # currentDate still injected (context dict non-empty) when there is any task state,
    # but with no task state and no memory the dict may be empty → no human message.


def test_boot_task_state_has_no_paired_assistant(monkeypatch, fresh_tasks):
    # Give task_state some content so it gets injected.
    monkeypatch.setattr(qe_mod, "build_memory_context", lambda _dir: None)
    fresh_tasks.add_todo("do a thing")  # produces non-empty state_summary()
    engine = _engine()
    asyncio.run(engine.boot())
    # No synthetic 'Acknowledged' assistant should exist.
    from mini_cc.engine.messages import AssistantMessage
    assistants = [m for m in engine.store._messages if isinstance(m, AssistantMessage)]
    assert all("Acknowledged" not in (getattr(m.content, "text", "") or "") for m in assistants)
```

> Note for the implementer: `fresh_tasks.add_todo(...)` is illustrative — use
> whatever the real `TaskManager` API is to make `state_summary()` non-empty
> (check `mini_cc/state/tasks.py`). If no simple adder exists, monkeypatch
> `tasks._tasks.state_summary` to return a fixed string instead.

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_boot_memory_injection.py -v`
Expected: FAIL — boot doesn't inject memory; `build_memory_context` not referenced in `query_engine`.

- [ ] **Step 3: Modify `boot` in `query_engine.py`**

Add the import near the top of `query_engine.py` (module level, so the test's
`monkeypatch.setattr(qe_mod, "build_memory_context", ...)` patches the name
boot resolves):

```python
from mini_cc.memdir import get_auto_mem_path
from mini_cc.memdir.injection import build_memory_context, render_user_context
```

Replace the body of `boot` (current 193-212) with:

```python
        for sub in self._subscriptions:
            await sub.start()
        await self._dispatch(
            SystemPromptMessage(content=self._build_system_prompt(), source="boot")
        )

        # Channel B-1: assemble the user-context dict (key-driven, extensible)
        # and inject as ONE synthetic UserMessage. Kept in the messages region
        # (not the system prompt) so DeepSeek's prefix cache over the system
        # prompt stays stable, and so the index can evolve without a cache bust.
        from datetime import date
        context: dict[str, str] = {}
        if mem := build_memory_context(get_auto_mem_path()):
            context["memory"] = mem
        context["currentDate"] = f"Today's date is {date.today().isoformat()}."
        rendered = render_user_context(context)
        if rendered:
            await self._dispatch(
                UserMessage(content=rendered, is_synthetic=True, source="memory")
            )

        # task_state: inject the synthetic user message ONLY. No paired
        # 'Acknowledged' assistant — 方案 C lets api_view merge consecutive
        # users, so we no longer fabricate an assistant turn in the store.
        state = tasks._tasks.state_summary()
        if state:
            await self._dispatch(
                UserMessage(content=state, is_synthetic=True, source="task_state")
            )
```

(Delete the old `AssistantMessage(... "Acknowledged. I have the current task plan." ...)` dispatch entirely.)

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_boot_memory_injection.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/mini_cc/engine/query_engine.py tests/test_boot_memory_injection.py
git commit -m "feat(memory): boot injects B-1 context message, drop task_state paired assistant"
```

---

### Task 6: 端到端 — boot 后 api_view 是单条合并 Human preamble

**Files:**
- Test: `tests/test_boot_memory_injection.py`（追加）

- [ ] **Step 1: Write the failing/integration test**

```python
# append to tests/test_boot_memory_injection.py
def test_boot_then_api_view_merges_into_single_human_preamble(monkeypatch, fresh_tasks):
    """End-to-end: after boot, the synthetic memory + currentDate (and any
    task_state) collapse to ONE HumanMessage in api_view — proving方案C
    keeps the preamble a single user turn (DeepSeek-safe, compact-safe)."""
    monkeypatch.setattr(qe_mod, "build_memory_context",
                        lambda _dir: "Contents of MEMORY.md (...):\n\n- [Role](r.md) — x")
    monkeypatch.setattr(qe_mod.tasks._tasks, "state_summary", lambda: "TASK STATE BLOCK")
    engine = _engine()
    asyncio.run(engine.boot())
    view = engine.store.api_view()
    # First message is the system prompt, then exactly one merged Human turn.
    assert isinstance(view[0], SystemMessage)
    humans = [m for m in view if isinstance(m, HumanMessage)]
    assert len(humans) == 1, f"expected 1 merged human, got {len(humans)}"
    merged = humans[0].content
    assert "Contents of MEMORY.md" in merged
    assert "TASK STATE BLOCK" in merged
    # No assistant turn fabricated between the two synthetic users.
    assert not any(isinstance(m, AIMessage) for m in view)
```

- [ ] **Step 2: Run to verify it fails (or passes if Task 4+5 already cover it)**

Run: `uv run pytest tests/test_boot_memory_injection.py::test_boot_then_api_view_merges_into_single_human_preamble -v`
Expected: PASS if Task 4 (merge) + Task 5 (boot) are both done; FAIL otherwise.

- [ ] **Step 3: (No new implementation — this validates Task 4 + Task 5 compose correctly.)**

If it fails, the bug is in the interaction: verify `api_view` calls
`_merge_consecutive_human` and boot dispatches both synthetic users with no
assistant between them.

- [ ] **Step 4: Run the full suite**

Run: `uv run pytest tests/ -v`
Expected: all PASS (new + existing test_memdir_storable.py regression).

- [ ] **Step 5: Commit**

```bash
git add tests/test_boot_memory_injection.py
git commit -m "test(memory): end-to-end boot → single merged human preamble"
```

---

## Self-Review

**1. Spec coverage** (against L2-spec M6/M9 方案 Y):
- B-0 协议声明 → Task 1 ✓
- truncate (D1 单独文件 / D3 完整复刻+why) → Task 2 ✓
- build_memory_context + render_user_context (两层, key 数据驱动) → Task 3 ✓
- api_view 合并连续 Human (方案 C) + compact docstring → Task 4 ✓
- boot 注入 + 去 task_state 配对 → Task 5 ✓
- T17 合并 / T18 包装 / T19 task_state 回归 → Task 4 / Task 3 / Task 6 ✓
- D2 current_date 参数传入（boot 构造）✓  D5 本地 date.today() ✓  D4 memory 在 task_state 前 ✓

**2. Placeholder scan:** Task 5 的 `fresh_tasks.add_todo` 标注了"用真实 TaskManager API 或 monkeypatch state_summary"——Task 6 已用 monkeypatch state_summary 的确定写法，实现者照此即可。无其它 TBD/模糊项。

**3. Type consistency:** `build_memory_context(memory_dir) -> str | None`、`render_user_context(dict) -> str`、`_merge_consecutive_human(list[BaseMessage]) -> list[BaseMessage]`、`truncate_entrypoint(str) -> str`、`MAX_ENTRYPOINT_LINES` 常量——跨 Task 1-6 引用一致。`QueryEngine(llm_base, main_tools, sub_tools, model_name, system_prompt_builder)` 与源码 `__init__` 签名一致。

**未决小项（实现时确认，不阻塞）:** `tasks.TaskManager` 让 `state_summary()` 非空的最简 API（Task 5 step1 注释已给 fallback：monkeypatch state_summary）。

---

## Edge Cases & Known Limits

| Edge case | 处理 |
|-----------|------|
| 🔴 MEMORY.md 含 `</system-reminder>` 提前闭合注入块 | **已处理**：`_neutralize_reminder_tags` 转义角括号 (Task 3) + 专项测试 |
| 🟡 truncate 输出略超 25KB（WARNING 文字是额外的）| 接受：25KB 是索引内容 cap，WARNING 是元信息（CC 同款）|
| 🟡 并发/排队输入致两条真实 user 被合并 | 当前串行不发生；`_merge_consecutive_human` docstring 标了扩展点（并发功能落地时在 store 层按 is_synthetic 合并）|
| 🟢 多模态 attachment（list content）破坏 `+` 拼接 | 不做（L5 范围）；docstring 标了 str-content 假设 |
| 🟢 session resume 后 boot 重注入 → 双份 memory | L2「重启=新 session」不恢复消息，不发生；resume 功能落地时 boot 加 dedup |
