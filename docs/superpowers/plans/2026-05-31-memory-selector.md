# Memory Selector (Layer 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the Layer-1 LLM memory selector to mini-cc: a generic `side_query` primitive plus `find_relevant_memories`, which picks the top-N memories relevant to a user query via a single cheap DeepSeek call.

**Architecture:** `side_query` is a dumb single-shot primitive (`deepseek-v4-flash`, non-thinking, json_object) returning raw text; it owns no parsing/validation. `find_relevant_memories` consumes the existing `scan.py` (scan + manifest), calls `side_query`, then parses → validates shape → filters to real filenames → maps back to `MemoryHeader` in model order → caps. Best-effort: any failure returns `[]`. No Layer-2 wiring (async prefetch, attachment injection, dedup derivation) here.

**Tech Stack:** Python 3.11, `langchain-deepseek` (OpenAI-format `/v1` endpoint), `pytest` with `asyncio_mode=auto`, `uv`.

**Spec:** `docs/superpowers/specs/2026-05-31-memory-selector-design.md` (side_query §2 locked; selector §3).

**Reuses (do not modify):** `src/mini_cc/memdir/scan.py` (`scan_memory_files`, `format_memory_manifest`), `src/mini_cc/memdir/types.py` (`MemoryHeader`).

---

## Task 0: Create feature branch

The repo is on `master` (15 commits ahead of origin). Do not implement or commit on `master`.

- [ ] **Step 1: Create and switch to the branch**

```bash
cd "D:/coding projects/build-mini-cc"
git checkout -b feat/memory-selector
```

Expected: `Switched to a new branch 'feat/memory-selector'`

---

## Task 1: `side_query` primitive

**Files:**
- Create: `src/mini_cc/side_query.py`
- Test: `tests/test_side_query.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_side_query.py`:

```python
"""side_query 是 API 边界的单测：假 _llm_flash 记录 bind kwargs + 返回 canned AIMessage，
全程不打 API（对照 test_boot_memory_injection.py 的 MagicMock-llm 模式）。"""
from langchain_core.messages import AIMessage

from mini_cc import side_query as sq_mod


class _FakeBound:
    def __init__(self, content):
        self._content = content

    async def ainvoke(self, messages):
        return AIMessage(content=self._content)


class _FakeFlash:
    """记录 .bind(**kwargs)，.ainvoke 返回固定 AIMessage。"""

    def __init__(self, content='{"selected_memories": []}'):
        self.recorded: dict = {}
        self._content = content

    def bind(self, **kwargs):
        self.recorded = kwargs
        return _FakeBound(self._content)


async def test_json_mode_binds_response_format(monkeypatch):
    fake = _FakeFlash(content='{"selected_memories": ["a.md"]}')
    monkeypatch.setattr(sq_mod, "_llm_flash", fake)
    out = await sq_mod.side_query("sys (json)", "usr", json_mode=True, max_tokens=256)
    assert fake.recorded["response_format"] == {"type": "json_object"}
    assert fake.recorded["max_tokens"] == 256
    assert out == '{"selected_memories": ["a.md"]}'


async def test_no_json_mode_omits_response_format(monkeypatch):
    fake = _FakeFlash()
    monkeypatch.setattr(sq_mod, "_llm_flash", fake)
    await sq_mod.side_query("sys", "usr", json_mode=False)
    assert "response_format" not in fake.recorded
    assert fake.recorded["max_tokens"] == 256


async def test_non_str_content_returns_empty(monkeypatch):
    fake = _FakeFlash(content=[{"type": "text", "text": "x"}])
    monkeypatch.setattr(sq_mod, "_llm_flash", fake)
    out = await sq_mod.side_query("sys", "usr")
    assert out == ""
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_side_query.py -v`
Expected: collection error / FAIL — `ModuleNotFoundError: No module named 'mini_cc.side_query'`.

- [ ] **Step 3: Write the implementation**

Create `src/mini_cc/side_query.py`:

```python
"""单发 side query 原语：在主对话循环之外调一次便宜、快、确定的副模型，返回原始文本。

dumb 通用原语：不解析、不校验、不吞异常——prompt / 解析 / best-effort 策略全归调用方
(让"失败→[]"只活在一处)。唯一调用方见 memdir/selector.py；它也是 selector 测试的
monkeypatch 缝。
"""
from __future__ import annotations

import logging
import os

from dotenv import load_dotenv

load_dotenv()  # 构造 _llm_flash 前确保 env 已读(幂等，同 llm.py)

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_deepseek import ChatDeepSeek

logger = logging.getLogger(__name__)

_FLASH_MODEL = "deepseek-v4-flash"
"""副模型名。select 质量不够时改这一行升 deepseek-v4-pro(仍非思考)，不碰思考轴。"""

# 策略焊死在实例上：每个 side query 都"便宜/快/确定/非思考"。
# 朴素 ChatDeepSeek(非 llm._ChatDeepSeekRoundTrip)：单发 [system,user] 无历史 assistant，
# reasoning_content round-trip 补丁是 no-op。
_llm_flash = ChatDeepSeek(
    model=_FLASH_MODEL,
    temperature=0,                                   # 分类任务 → 确定性
    timeout=10,                                      # 防挂死兜底(→ request_timeout)
    max_retries=1,                                   # best-effort 快路径
    extra_body={"thinking": {"type": "disabled"}},   # 非思考：省钱/快 + 不让思考吃 max_tokens→空content
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)


async def side_query(
    system: str,
    user: str,
    *,
    json_mode: bool = False,
    max_tokens: int = 256,
) -> str:
    """跑一轮 [system, user]，返回模型原始文本。

    - 不解析/校验/吞异常(API 错、timeout、CancelledError 全向上抛)。
    - json_mode=True 置 response_format json_object；调用方 prompt 须含字面 "json"+示例+空目标。
    - system=稳定(缓存前缀) / user=变化(miss 尾)，拆分即 prompt-cache 边界。
    """
    binds: dict = {"max_tokens": max_tokens}
    if json_mode:
        binds["response_format"] = {"type": "json_object"}
    resp = await _llm_flash.bind(**binds).ainvoke(
        [SystemMessage(content=system), HumanMessage(content=user)]
    )
    # 缓存命中率诊断(方案 A)：debug 级，返回类型不变；确切 metadata key 见包内 usage 字段。
    logger.debug("side_query usage: %s", resp.response_metadata)
    content = resp.content
    return content if isinstance(content, str) else ""
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_side_query.py -v`
Expected: `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/mini_cc/side_query.py tests/test_side_query.py
git commit -m "feat: add side_query primitive for cheap off-loop LLM calls"
```

---

## Task 2: `find_relevant_memories` selector

**Files:**
- Create: `src/mini_cc/memdir/selector.py`
- Test: `tests/test_memdir_selector.py`

Depends on Task 1 (imports `from mini_cc.side_query import side_query`).

- [ ] **Step 1: Write the failing test**

Create `tests/test_memdir_selector.py`:

```python
"""find_relevant_memories Layer 1 单测：monkeypatch selector.side_query，全程不打 API。"""
import os
from pathlib import Path

from mini_cc.memdir import selector as sel_mod
from mini_cc.memdir.selector import find_relevant_memories


def _write(memdir: Path, name: str, *, type_: str = "user", desc: str = "d") -> None:
    memdir.mkdir(parents=True, exist_ok=True)
    (memdir / name).write_text(
        f"---\nname: {name}\ndescription: {desc}\ntype: {type_}\n---\n\nbody\n",
        encoding="utf-8",
    )


class _FakeSideQuery:
    """记录最近一次 (system, user) + 调用次数；返回固定 JSON 文本。"""

    def __init__(self, returns: str):
        self.returns = returns
        self.system: str | None = None
        self.user: str | None = None
        self.calls = 0

    async def __call__(self, system, user, *, json_mode=False, max_tokens=256):
        self.calls += 1
        self.system, self.user = system, user
        return self.returns


async def test_returns_selected_headers_in_model_order(tmp_path, monkeypatch):
    _write(tmp_path, "a.md")
    _write(tmp_path, "b.md")
    # 让 a.md 比 b.md 新 → scan(mtime 倒序)给 [a, b]；模型却返回 [b, a]。
    # 输出 == [b, a] 证明是"模型序"而非 scan 的 mtime 序。
    os.utime(tmp_path / "a.md", (2000, 2000))
    os.utime(tmp_path / "b.md", (1000, 1000))
    monkeypatch.setattr(sel_mod, "side_query",
                        _FakeSideQuery('{"selected_memories": ["b.md", "a.md"]}'))
    out = await find_relevant_memories("q", tmp_path)
    assert [h.filename for h in out] == ["b.md", "a.md"]


async def test_caps_at_max_results(tmp_path, monkeypatch):
    for n in ["a.md", "b.md", "c.md", "d.md", "e.md", "f.md"]:
        _write(tmp_path, n)
    monkeypatch.setattr(sel_mod, "side_query", _FakeSideQuery(
        '{"selected_memories": ["a.md","b.md","c.md","d.md","e.md","f.md"]}'))
    out = await find_relevant_memories("q", tmp_path)
    assert len(out) == 5


async def test_already_surfaced_filtered_before_side_query(tmp_path, monkeypatch):
    _write(tmp_path, "a.md")
    _write(tmp_path, "b.md")
    fake = _FakeSideQuery('{"selected_memories": []}')
    monkeypatch.setattr(sel_mod, "side_query", fake)
    await find_relevant_memories("q", tmp_path, already_surfaced=frozenset({"a.md"}))
    assert "a.md" not in fake.system   # 在 manifest(system) 里就没了
    assert "b.md" in fake.system


async def test_recent_tools_in_prompt(tmp_path, monkeypatch):
    _write(tmp_path, "a.md")
    fake = _FakeSideQuery('{"selected_memories": []}')
    monkeypatch.setattr(sel_mod, "side_query", fake)
    await find_relevant_memories("q", tmp_path, recent_tools=("file_edit",))
    assert "file_edit" in fake.user


async def test_hallucinated_filename_dropped(tmp_path, monkeypatch):
    _write(tmp_path, "a.md")
    monkeypatch.setattr(sel_mod, "side_query",
                        _FakeSideQuery('{"selected_memories": ["ghost.md", "a.md"]}'))
    out = await find_relevant_memories("q", tmp_path)
    assert [h.filename for h in out] == ["a.md"]   # ghost.md 不存在被丢


async def test_broken_json_returns_empty(tmp_path, monkeypatch):
    _write(tmp_path, "a.md")
    monkeypatch.setattr(sel_mod, "side_query", _FakeSideQuery("not json at all"))
    out = await find_relevant_memories("q", tmp_path)
    assert out == []


async def test_side_query_raises_returns_empty(tmp_path, monkeypatch):
    _write(tmp_path, "a.md")

    async def _boom(*a, **k):
        raise RuntimeError("api down")

    monkeypatch.setattr(sel_mod, "side_query", _boom)
    out = await find_relevant_memories("q", tmp_path)
    assert out == []


async def test_empty_memdir_short_circuits(tmp_path, monkeypatch):
    fake = _FakeSideQuery('{"selected_memories": []}')
    monkeypatch.setattr(sel_mod, "side_query", fake)
    out = await find_relevant_memories("q", tmp_path)   # 空目录
    assert out == []
    assert fake.calls == 0   # 短路，不调 side_query
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_memdir_selector.py -v`
Expected: collection error / FAIL — `ModuleNotFoundError: No module named 'mini_cc.memdir.selector'`.

- [ ] **Step 3: Write the implementation**

Create `src/mini_cc/memdir/selector.py`:

```python
"""LLM 记忆选择器(Layer 1 纯召回)。

CC 对照：src/memdir/findRelevantMemories.ts。流程：扫 memdir → 滤已展示 → manifest →
side_query(flash, json_object) → 解析 → 校验形状 → 过滤真实文件名 → 按模型序映射回 header
→ 截断。Best-effort：任何失败 → []。

Layer 2(异步 prefetch、attachment 注入、三重去重、session 字节预算、recent_tools /
already_surfaced 的消息派生)不在此处。
"""
from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path

from mini_cc.memdir.scan import format_memory_manifest, scan_memory_files
from mini_cc.memdir.types import MemoryHeader
from mini_cc.side_query import side_query

logger = logging.getLogger(__name__)

_SELECT_SYSTEM_PROMPT = """You are selecting memories useful to a personal assistant as it answers a user's query. You are given the query and a list of available memory files (filename + description).

## Selection rules
Return the filenames of memories that will clearly help (up to 5). Only include ones you are certain about, based on name and description.
- If unsure a memory is useful, do not include it. Be selective.
- If none are clearly useful, return an empty list.
- If recently-used tools are listed, do NOT select usage/API-reference memories for those tools (the assistant is already using them). DO still select memories with warnings, gotchas, or known issues about those tools.

## Output format
Respond with a single JSON object and NOTHING else, in exactly this shape:
{"selected_memories": ["<filename copied verbatim>", ...]}
- At most 5 filenames, each copied verbatim from the list above.
- If nothing is clearly useful, you MUST still return valid JSON with an empty array: {"selected_memories": []}.
- Never return an empty message, prose, or only whitespace — always a JSON object.

Examples:
{"selected_memories": []}
{"selected_memories": ["feedback_testing.md", "user_role.md"]}

## Available memories
"""


async def find_relevant_memories(
    query: str,
    memdir: Path,
    *,
    recent_tools: Sequence[str] = (),
    already_surfaced: frozenset[str] = frozenset(),
    max_results: int = 5,
) -> list[MemoryHeader]:
    """选出至多 max_results 个与 query 相关的 memory。Best-effort：任何失败返回 []。"""
    try:
        headers = [
            h for h in scan_memory_files(memdir)
            if h.filename not in already_surfaced
        ]
        if not headers:
            return []

        system = _SELECT_SYSTEM_PROMPT + format_memory_manifest(headers)
        user = _build_user_prompt(query, recent_tools)

        names = _parse_selected(await side_query(system, user, json_mode=True))

        by_name = {h.filename: h for h in headers}
        seen: set[str] = set()
        selected: list[MemoryHeader] = []
        for name in names:  # 按模型返回顺序；去重 + 过滤幻觉(CC findRelevantMemories.ts:130)
            header = by_name.get(name)
            if header is not None and name not in seen:
                seen.add(name)
                selected.append(header)
        return selected[:max_results]
    except Exception:  # best-effort：绝不崩 turn(CancelledError 是 BaseException，不被吞)
        logger.debug("find_relevant_memories failed", exc_info=True)
        return []


def _build_user_prompt(query: str, recent_tools: Sequence[str]) -> str:
    user = f"Query: {query}"
    if recent_tools:
        user += f"\n\nRecently used tools: {', '.join(recent_tools)}"
    return user


def _parse_selected(raw: str) -> list[str]:
    """side_query 的 JSON 文本 → 文件名字符串列表。形状不对 → []（坏 JSON 由外层 try 兜）。"""
    data = json.loads(raw)
    if not isinstance(data, dict):
        return []
    names = data.get("selected_memories")
    if not isinstance(names, list):
        return []
    return [n for n in names if isinstance(n, str)]
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_memdir_selector.py -v`
Expected: `8 passed`.

- [ ] **Step 5: Run the full suite (no regressions)**

Run: `uv run pytest -q`
Expected: all tests pass (existing + 11 new).

- [ ] **Step 6: Commit**

```bash
git add src/mini_cc/memdir/selector.py tests/test_memdir_selector.py
git commit -m "feat(memdir): add LLM memory selector (find_relevant_memories)"
```

---

## Notes / out of scope

- **Layer 2 (not here):** async prefetch, attachment injection, triple dedup, session byte budget, and the message-history derivation of `recent_tools` / `already_surfaced`. Layer 1 takes those as params (defaults empty); a later plan wires the producers.
- **Deferred optimizations (§3 of spec):** manifest re-ordering for cache stability, dropping/coarsening the per-row timestamp, and #15 (ISO vs age-words). Decide later with the cache-hit telemetry (`logger.debug` in side_query). The selector consumes the existing CC-faithful `scan.py` manifest as-is.
- **Signature note:** supersedes the `L2-spec.md` M5 sketch — `already_surfaced` is now `frozenset[str]` (filenames) keyword-only, and `recent_tools` is added.
- **No push.** Local commits only; pushing `feat/memory-selector` needs explicit user confirmation.

## Self-Review

1. **Spec coverage:** side_query §2 (instance policy, signature, json_mode→response_format, non-str→"", debug cache log) → Task 1. selector §3 (scan→filter already_surfaced→short-circuit→manifest→side_query→parse→validate→filter real filenames→model-order map→cap; best-effort `[]`; cache layout system/user; SELECT_PROMPT with json+example+empty-target; recent_tools section) → Task 2. Covered.
2. **Placeholder scan:** no TBD/TODO; every code step has full code; commands have expected output. Clean.
3. **Type consistency:** `find_relevant_memories(query, memdir, *, recent_tools, already_surfaced, max_results) -> list[MemoryHeader]` used identically in test and impl; `side_query(system, user, *, json_mode, max_tokens) -> str` identical across Task 1 impl, Task 1 test, and Task 2's `_FakeSideQuery.__call__`. `MemoryHeader.filename` matches `scan.py`/`types.py`. Consistent.

