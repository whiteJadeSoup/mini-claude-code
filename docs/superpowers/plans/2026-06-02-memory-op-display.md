# 记忆读写操作折叠展示（第②面）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** agent 主动 `file_read`/`file_edit`/`file_write`/`grep`/`glob` 命中 memdir 时，TUI 把连续操作折叠成一行语义计数行（`Recalling/Recalled/Wrote/Searched N memories`），复刻 Claude Code 的 inline 记忆展示。

**Architecture:** 两层——(1) `consumers/tui/memory_ops.py` 一组纯函数 + 纯累加器（`classify_memory_op` / `memory_run_summary` / `MemoryRun`），不依赖 Textual，独立单测；(2) `ToolStatus`（`app.py`）持一个 `MemoryRun` 实例，在 `add_tool`/`_tick`/`end_turn` 处转发，把摘要包成 markup 行。游程归属锚定 `add_tool` 发出顺序，对未来并行执行前向兼容。

**Tech Stack:** Python 3.11、Textual≥0.80、pytest + pytest-asyncio（`asyncio_mode=auto`）、ruff line-length 100。

**Spec:** `docs/superpowers/specs/2026-06-02-memory-op-display-design.md`

**执行前**：在 mini-cc 仓库新建特性分支（如 `feat/memory-op-display`），不要直接在默认分支上做。

---

## File Structure

| 文件 | 动作 | 职责 |
|---|---|---|
| `src/mini_cc/memdir/paths.py` | 改 | 新增 `is_memory_path()`（纯比较，抽自重复代码） |
| `src/mini_cc/memdir/__init__.py` | 改 | 导出 `is_memory_path` |
| `src/mini_cc/tools/file_read/__init__.py` | 改 | `_frozen_memory_staleness` 改用 `is_memory_path`（去重） |
| `src/mini_cc/config.py` | 改 | `safe_path` 的 memdir 判定改用 `is_memory_path`（去重） |
| `src/mini_cc/consumers/tui/memory_ops.py` | 建 | `classify_memory_op` + `memory_run_summary` + `MemoryRun`（纯，无 widget） |
| `src/mini_cc/consumers/tui/app.py` | 改 | `ToolStatus` 接入 `MemoryRun` 折叠展示 |
| `tests/test_memory_op_display.py` | 建 | T1–T16 全部用例 |

> **不动**：`consumers/tui/renderers.py` 的 `_render_relevant_memory` / `_recalled_markup`（第①面自动注入，保留）。

---

## Task 1: 重构 — 抽取 `is_memory_path`（零行为变更）

memdir 成员判定现在重复在 `file_read/__init__.py:37` 和 `config.py:56`。先抽成一个函数，两处复用。**必须惰性从包属性取 `get_auto_mem_path`**，否则现有 `test_file_read_memory.py` 会因 monkeypatch 失效而全挂。

**Files:**
- Create: `tests/test_memory_op_display.py`
- Modify: `src/mini_cc/memdir/paths.py`（末尾追加函数）
- Modify: `src/mini_cc/memdir/__init__.py:17-47`
- Modify: `src/mini_cc/tools/file_read/__init__.py:27-39`
- Modify: `src/mini_cc/config.py:52-59`

- [ ] **Step 1: 写失败测试（含共享 fixture）**

新建 `tests/test_memory_op_display.py`：

```python
"""Surface ② — agent-initiated memory op display (classify + collapse).
See docs/superpowers/specs/2026-06-02-memory-op-display-design.md.
"""
import os
from pathlib import Path

import pytest

from mini_cc import config


@pytest.fixture
def memdir(tmp_path, monkeypatch):
    """Point the session memdir at a tmp dir and set CWD to tmp_path.

    Patches the PACKAGE attribute mini_cc.memdir.get_auto_mem_path, because
    is_memory_path / safe_path resolve it lazily through the package at call
    time (mirrors tests/test_file_read_memory.py:_point_memdir_at). realpath
    so it matches safe_path's realpath'd `resolved`.
    """
    monkeypatch.setattr(config, "CWD", os.path.realpath(tmp_path))
    d = tmp_path / "mem"
    d.mkdir()
    real = Path(os.path.realpath(d))
    monkeypatch.setattr("mini_cc.memdir.get_auto_mem_path", lambda: real)
    return real


def test_is_memory_path_true_for_file_in_memdir(memdir):
    from mini_cc.memdir import is_memory_path
    assert is_memory_path(str(memdir / "user_nickname.md")) is True


def test_is_memory_path_false_for_project_file(memdir):
    from mini_cc.memdir import is_memory_path
    assert is_memory_path(os.path.join(config.CWD, "src", "foo.py")) is False


def test_is_memory_path_true_for_memdir_itself(memdir):
    from mini_cc.memdir import is_memory_path
    assert is_memory_path(str(memdir)) is True


def test_is_memory_path_true_for_nested_file(memdir):
    from mini_cc.memdir import is_memory_path
    assert is_memory_path(str(memdir / "sub" / "deep.md")) is True
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/test_memory_op_display.py -v`
Expected: 4 个 FAIL，`ImportError: cannot import name 'is_memory_path'`。

- [ ] **Step 3: 实现 `is_memory_path`**

在 `src/mini_cc/memdir/paths.py` **末尾**追加：

```python
def is_memory_path(resolved: str) -> bool:
    """Whether an already-resolved absolute path lives inside this session's
    memdir (the memdir directory itself counts).

    `resolved` MUST already be resolved (e.g. via os.path.realpath or
    config.safe_path) — this is a pure comparison; it does not resolve or
    expand. Kept config-free to avoid a config<->memdir import cycle: the
    caller owns the resolve step.

    The lazy `from mini_cc.memdir import get_auto_mem_path` routes through the
    package attribute (not the module-local name) so tests that monkeypatch
    `mini_cc.memdir.get_auto_mem_path` take effect here too — matching how
    config.safe_path and file_read already resolve it.
    """
    from mini_cc.memdir import get_auto_mem_path
    memdir = os.path.normcase(str(get_auto_mem_path()))
    rp = os.path.normcase(resolved)
    return rp == memdir or rp.startswith(memdir + os.sep)
```

在 `src/mini_cc/memdir/__init__.py` 导出它——在 `from mini_cc.memdir.paths import (...)` 块加一行、`__all__` 加一项：

```python
from mini_cc.memdir.paths import (
    get_auto_mem_daily_log_path,
    get_auto_mem_entrypoint,
    get_auto_mem_path,
    is_memory_path,
    validate_memory_path,
)
```

`__all__` 中加入 `"is_memory_path",`（保持字母序，放在 `"get_auto_mem_path",` 之后）。

- [ ] **Step 4: 跑测试确认通过**

Run: `pytest tests/test_memory_op_display.py -v`
Expected: 4 个 PASS。

- [ ] **Step 5: 两处去重改用 `is_memory_path`**

`src/mini_cc/tools/file_read/__init__.py` 的 `_frozen_memory_staleness`（约 27-39 行）整体替换为：

```python
def _frozen_memory_staleness(resolved: str, mtime_ms: int) -> str:
    """Frozen freshness note for a memory-file read. Computed ONCE here (not in
    to_api_str), so the tool-result bytes stay stable across api_view rebuilds —
    age derives from the clock and would otherwise drift on day boundaries and
    bust the prompt cache (same reason SurfacedMemory.header is frozen). Returns
    '' for non-memory paths or memories <=1 day old."""
    from mini_cc.memdir import is_memory_path
    if is_memory_path(resolved):
        return memory_freshness_note(mtime_ms)
    return ""
```

`src/mini_cc/config.py` 的 `safe_path` 中，把 52-57 行的 memdir 比较块：

```python
    # Lazy: avoid import-time git rev-parse subprocess. get_auto_mem_path is
    # @cache'd so this is one subprocess per process; ~0ms after the first call.
    from mini_cc.memdir import get_auto_mem_path
    memdir_norm = os.path.normcase(str(get_auto_mem_path()))
    if resolved_norm == memdir_norm or resolved_norm.startswith(memdir_norm + os.sep):
        return resolved
```

替换为：

```python
    # Lazy: avoid import-time git rev-parse subprocess. is_memory_path does the
    # normcase memdir comparison (get_auto_mem_path is @cache'd: ~0ms after first).
    from mini_cc.memdir import is_memory_path
    if is_memory_path(resolved):
        return resolved
```

（`resolved` 是 45 行 realpath 的结果；`is_memory_path` 内部自己 normcase，故传原始 `resolved`，不传 `resolved_norm`。）

- [ ] **Step 6: 跑全量测试，证明零行为变更**

Run: `pytest -q`
Expected: 全绿，尤其 `tests/test_file_read_memory.py` 5 个用例（safe_path + staleness 经由 `is_memory_path` 仍按 monkeypatch 的 memdir 工作）全 PASS。

- [ ] **Step 7: 提交**

```bash
git add src/mini_cc/memdir/paths.py src/mini_cc/memdir/__init__.py \
        src/mini_cc/tools/file_read/__init__.py src/mini_cc/config.py \
        tests/test_memory_op_display.py
git commit -m "refactor: extract is_memory_path, dedupe memdir membership check

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: `memory_ops.py` — 分类 + 摘要 + 游程累加器（纯，无 widget）

三件纯逻辑：`classify_memory_op`（路径 → 操作类型）、`memory_run_summary`（计数 → 文案）、`MemoryRun`（连续游程状态机）。全部不依赖 Textual，独立单测。

**Files:**
- Create: `src/mini_cc/consumers/tui/memory_ops.py`
- Modify: `tests/test_memory_op_display.py`（追加用例）

- [ ] **Step 1: 写失败测试（分类 + 摘要 + 游程）**

在 `tests/test_memory_op_display.py` **末尾**追加：

```python
# ---- classify_memory_op (T5-T8) ----

def test_classify_read_write_search(memdir):
    from mini_cc.consumers.tui.memory_ops import classify_memory_op
    p = str(memdir / "m.md")
    assert classify_memory_op("file_read", {"path": p}) == "read"
    assert classify_memory_op("file_edit", {"path": p}) == "write"
    assert classify_memory_op("file_write", {"path": p}) == "write"
    assert classify_memory_op("grep", {"pattern": "x", "path": str(memdir)}) == "search"
    assert classify_memory_op("glob", {"pattern": "*.md", "path": str(memdir)}) == "search"


def test_classify_none_for_project_file(memdir):
    from mini_cc.consumers.tui.memory_ops import classify_memory_op
    p = os.path.join(config.CWD, "src", "a.py")
    assert classify_memory_op("file_read", {"path": p}) is None


def test_classify_none_for_search_without_path(memdir):
    from mini_cc.consumers.tui.memory_ops import classify_memory_op
    assert classify_memory_op("grep", {"pattern": "x"}) is None


def test_classify_none_for_escape_path(memdir):
    from mini_cc.consumers.tui.memory_ops import classify_memory_op
    # safe_path rejects cwd+memdir escapes → ValueError → not a memory op
    assert classify_memory_op("file_read", {"path": "../../../../etc/passwd"}) is None


# ---- memory_run_summary (T9-T11) ----

def test_summary_multi_kind_join():
    from mini_cc.consumers.tui.memory_ops import memory_run_summary
    assert memory_run_summary(2, 1, 0, active=False) == "Recalled 2 memories · wrote 1 memory"


def test_summary_search_has_no_count():
    from mini_cc.consumers.tui.memory_ops import memory_run_summary
    assert memory_run_summary(0, 0, 1, active=True) == "Searching memories"


def test_summary_singular():
    from mini_cc.consumers.tui.memory_ops import memory_run_summary
    assert memory_run_summary(1, 0, 0, active=False) == "Recalled 1 memory"


def test_summary_empty_is_blank():
    from mini_cc.consumers.tui.memory_ops import memory_run_summary
    assert memory_run_summary(0, 0, 0, active=False) == ""


# ---- MemoryRun state machine (T12-T14, T16) ----

def test_run_collapses_consecutive_reads(memdir):
    from mini_cc.consumers.tui.memory_ops import MemoryRun
    r = MemoryRun()
    p = str(memdir / "m.md")
    for _ in range(3):
        assert r.absorb("file_read", {"path": p}) is True
    assert r.is_open is True
    assert r.flush() == "Recalled 3 memories"
    assert r.is_open is False
    assert r.flush() is None  # idempotent close


def test_run_breaks_on_non_memory_tool(memdir):
    from mini_cc.consumers.tui.memory_ops import MemoryRun
    r = MemoryRun()
    p = str(memdir / "m.md")
    r.absorb("file_read", {"path": p})
    r.absorb("file_read", {"path": p})
    assert r.absorb("execute_command", {"command": "ls"}) is False
    assert r.flush() == "Recalled 2 memories"
    assert r.absorb("file_read", {"path": p}) is True
    assert r.flush() == "Recalled 1 memory"


def test_run_merges_across_steps(memdir):
    # No non-memory tool between reads → one run regardless of LLM step boundary.
    from mini_cc.consumers.tui.memory_ops import MemoryRun
    r = MemoryRun()
    p = str(memdir / "m.md")
    for _ in range(3):
        r.absorb("file_read", {"path": p})
    assert r.flush() == "Recalled 3 memories"


def test_run_counts_op_regardless_of_later_error(memdir):
    # absorb happens at add-time; a later failed result doesn't change the count.
    from mini_cc.consumers.tui.memory_ops import MemoryRun
    r = MemoryRun()
    r.absorb("file_read", {"path": str(memdir / "m.md")})
    assert r.flush() == "Recalled 1 memory"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/test_memory_op_display.py -v`
Expected: 新增用例全 FAIL，`ModuleNotFoundError: No module named 'mini_cc.consumers.tui.memory_ops'`。

- [ ] **Step 3: 实现 `memory_ops.py`**

新建 `src/mini_cc/consumers/tui/memory_ops.py`：

```python
"""Classify + summarize agent-initiated memory tool operations (surface ②).

See docs/superpowers/specs/2026-06-02-memory-op-display-design.md. Pure
functions + a pure accumulator (no widget state) so they unit-test directly,
mirroring CC's getSearchReadSummaryText and renderers._recalled_markup.
"""
from __future__ import annotations

from mini_cc import config
from mini_cc.memdir import is_memory_path

_READ_TOOLS = frozenset({"file_read"})
_WRITE_TOOLS = frozenset({"file_edit", "file_write"})
_SEARCH_TOOLS = frozenset({"grep", "glob"})


def classify_memory_op(name: str, args: dict) -> str | None:
    """Return "read" | "write" | "search" if this tool call targets a file
    inside the memory directory, else None.

    The path is resolved with config.safe_path — the same resolution the tools
    themselves apply — so classification matches what actually gets read/written.
    Paths outside cwd+memdir raise in safe_path and yield None. grep/glob with no
    explicit `path` search the project, not memory.
    """
    if name in _READ_TOOLS:
        kind = "read"
    elif name in _WRITE_TOOLS:
        kind = "write"
    elif name in _SEARCH_TOOLS:
        kind = "search"
    else:
        return None

    path = args.get("path")
    if not path:
        return None
    try:
        resolved = config.safe_path(path)
    except ValueError:
        return None
    return kind if is_memory_path(resolved) else None


def memory_run_summary(read: int, write: int, search: int, *, active: bool) -> str:
    """Collapsed line text for a memory run. Verb order read · search · write;
    the first part's verb is capitalized, the rest lowercased (CC parity). Read
    and write carry a count (singular/plural); search does not. Returns "" when
    all counts are zero.
    """
    parts: list[str] = []

    def verb(present: str, past: str) -> str:
        word = present if active else past
        return word if not parts else word.lower()

    if read:
        noun = "memory" if read == 1 else "memories"
        parts.append(f"{verb('Recalling', 'Recalled')} {read} {noun}")
    if search:
        parts.append(f"{verb('Searching', 'Searched')} memories")
    if write:
        noun = "memory" if write == 1 else "memories"
        parts.append(f"{verb('Writing', 'Wrote')} {write} {noun}")

    return " · ".join(parts)


class MemoryRun:
    """Open memory-run accumulator (surface ②). Pure state machine — no widget —
    so the consecutive-grouping logic unit-tests without importing the TUI.
    ToolStatus owns one instance and renders its summaries as markup rows.

    Membership is decided at absorb()-time (emit order), NOT at completion time,
    so counts/grouping stay correct once tool execution becomes parallel; only
    flushed-row ordering would then depend on completion (a pre-existing TUI
    property shared by all tools).
    """

    def __init__(self) -> None:
        self._counts: dict[str, int] | None = None

    def absorb(self, name: str, args: dict) -> bool:
        """Fold a memory op into the open run (opening one if needed) and return
        True. Return False for a non-memory op — the caller should flush() the
        run and handle the tool normally."""
        kind = classify_memory_op(name, args)
        if kind is None:
            return False
        if self._counts is None:
            self._counts = {"read": 0, "write": 0, "search": 0}
        self._counts[kind] += 1
        return True

    @property
    def is_open(self) -> bool:
        return self._counts is not None

    def live_summary(self) -> str | None:
        """Present-tense summary for the live row, or None if no run is open."""
        if self._counts is None:
            return None
        c = self._counts
        return memory_run_summary(c["read"], c["write"], c["search"], active=True)

    def flush(self) -> str | None:
        """Close the run and return the past-tense summary to persist, or None
        if no run was open."""
        if self._counts is None:
            return None
        c = self._counts
        self._counts = None
        return memory_run_summary(c["read"], c["write"], c["search"], active=False)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `pytest tests/test_memory_op_display.py -v`
Expected: T5–T16 对应 13 个新用例全 PASS（连同 Task 1 的 4 个共 17 个）。

- [ ] **Step 5: 提交**

```bash
git add src/mini_cc/consumers/tui/memory_ops.py tests/test_memory_op_display.py
git commit -m "feat(tui): classify + summarize agent memory tool operations

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: 接入 `ToolStatus` — 折叠展示

`ToolStatus` 持一个 `MemoryRun`；`add_tool` 吸收记忆 op（不进逐行 blink），非记忆顶层工具/turn 末 flush 一条折叠行；`_tick` 渲染 live 行。

**Files:**
- Modify: `src/mini_cc/consumers/tui/app.py`（import + `ToolStatus` 的 `__init__`/`start_turn`/`end_turn`/`add_tool`/`_tick`，新增 `_flush_mem_run`）
- Modify: `tests/test_memory_op_display.py`（追加 T12 端到端 + T15 sub-tool）

- [ ] **Step 1: 写失败测试（ToolStatus 接线）**

在 `tests/test_memory_op_display.py` **末尾**追加：

```python
# ---- ToolStatus wiring (T12 end-to-end, T15 sub-tool) ----

def _make_tool_status():
    """Instantiate ToolStatus unmounted; stub post_message to capture flushes.
    The exercised paths (add_tool / _flush_mem_run) only touch post_message —
    no Textual screen needed."""
    from mini_cc.consumers.tui.app import ToolStatus
    ts = ToolStatus()
    captured = []
    ts.post_message = lambda m: captured.append(m)
    return ts, captured


def test_toolstatus_flushes_one_memory_row_when_broken(memdir):
    from mini_cc.consumers.tui.app import ToolFlushed
    ts, captured = _make_tool_status()
    p = str(memdir / "m.md")
    ts.add_tool("c0", "file_read", {"path": p}, "", "a0", None)
    ts.add_tool("c1", "file_read", {"path": p}, "", "a1", None)
    assert ts._mem_run.is_open is True
    assert captured == []  # nothing flushed mid-run
    # A non-memory top-level tool breaks the run → exactly one collapsed row.
    ts.add_tool("c2", "execute_command", {"command": "ls"}, "", "a2", None)
    flushed = [m for m in captured if isinstance(m, ToolFlushed)]
    assert len(flushed) == 1
    assert "Recalled 2 memories" in flushed[0].markup
    assert ts._mem_run.is_open is False


def test_toolstatus_subtool_memory_read_not_folded(memdir):
    ts, captured = _make_tool_status()
    # parent_id set → sub-tool path; must NOT open a memory run.
    ts.add_tool("s1", "file_read", {"path": str(memdir / "m.md")},
                "  ", "a1", "missing-parent")
    assert ts._mem_run.is_open is False
    assert captured == []
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/test_memory_op_display.py -k toolstatus -v`
Expected: 2 个 FAIL（`AttributeError: 'ToolStatus' object has no attribute '_mem_run'`）。

- [ ] **Step 3: 接入 `ToolStatus`**

`src/mini_cc/consumers/tui/app.py` 顶部 import 区（约 35-38 行附近）加一行：

```python
from mini_cc.consumers.tui.memory_ops import MemoryRun
```

`ToolStatus.__init__` 加一行 `self._mem_run`：

```python
    def __init__(self) -> None:
        super().__init__("")
        self._tools: dict[str, dict] = {}
        self._tool_order: list[str] = []
        self._groups: dict[str, dict] = {}
        self._mem_run = MemoryRun()
        self._frame = 0
```

`start_turn` 重置 run：

```python
    def start_turn(self) -> None:
        self._tools.clear()
        self._tool_order.clear()
        self._groups.clear()
        self._mem_run = MemoryRun()
        self._frame = 0
        self.add_class("active")
```

`end_turn` 先 flush 再清：

```python
    def end_turn(self) -> None:
        self._flush_mem_run()
        self._tools.clear()
        self._tool_order.clear()
        self._groups.clear()
        self._mem_run = MemoryRun()
        self.update("")
        self.remove_class("active")
```

`add_tool` 顶部加记忆分流（仅顶层）：

```python
    def add_tool(
        self,
        call_id: str,
        name: str,
        args: dict,
        prefix: str,
        asst_id: str,
        parent_id: str | None,
    ) -> None:
        # Top-level agent memory ops collapse into one rolling run instead of a
        # per-call blink row (surface ②; see consumers/tui/memory_ops.py).
        if parent_id is None:
            if self._mem_run.absorb(name, args):
                return
            # A non-memory top-level tool ends the current run (CC's flushGroup).
            self._flush_mem_run()

        mini = get_tool(name)
        args_repr = mini.render_received(args) if mini else ""
        if parent_id is None:
            self._tools[call_id] = {
                "name": name,
                "args": args,
                "args_repr": args_repr,
                "prefix": prefix,
                "asst_id": asst_id,
                "started_at": time.monotonic(),
            }
            self._tool_order.append(call_id)
        else:
            main_call_id = next(
                (cid for cid, t in self._tools.items() if t["asst_id"] == parent_id),
                None,
            )
            if main_call_id is None:
                return
            if main_call_id not in self._groups:
                self._groups[main_call_id] = {
                    "tool_count": 0,
                    "current_label": f"{name}({args_repr})",
                    "started_at": time.monotonic(),
                }
            g = self._groups[main_call_id]
            g["tool_count"] += 1
            g["current_label"] = f"{name}({args_repr})"
```

新增 `_flush_mem_run`（放在 `complete_tool` 之后、`_tick` 之前）：

```python
    def _flush_mem_run(self) -> None:
        """Close the open memory run (if any) and persist one collapsed row."""
        summary = self._mem_run.flush()
        if summary:
            self.post_message(
                ToolFlushed(f"[green]●[/green] [cyan]{summary}[/cyan]")
            )
```

`_tick` 改早退条件 + 末尾追加 live 记忆行：

```python
    def _tick(self) -> None:
        self._frame = (self._frame + 1) % len(_BLINK)
        if not self._tools and not self._mem_run.is_open:
            return
        b = _BLINK[self._frame]
        lines = []
        for cid in self._tool_order:
            t = self._tools[cid]
            mini = get_tool(t["name"])
            exec_repr = (
                mini.render_executing(t["args"]) if mini else t["args_repr"]
            )
            lines.append(
                f"{t['prefix']}[grey50]{b}[/grey50] "
                f"[cyan]{t['name']}[/cyan]({exec_repr})"
            )
            if cid in self._groups:
                g = self._groups[cid]
                elapsed = int(time.monotonic() - g["started_at"])
                lines.append(
                    f"{t['prefix']}│  [grey50]{b}[/grey50] "
                    f"[cyan]{g['current_label']}[/cyan]"
                    f" · {g['tool_count']} · {elapsed}s"
                )
        summary = self._mem_run.live_summary()
        if summary:
            lines.append(f"[grey50]{b}[/grey50] [cyan]{summary}[/cyan]")
        self.update("\n".join(lines))
```

> `complete_tool` 不改：记忆 op 从不进 `self._tools`，故其 `if call_id not in self._tools: return` 对记忆结果天然 no-op（含失败的红 ●，符合 T16）。

- [ ] **Step 4: 跑测试确认通过**

Run: `pytest tests/test_memory_op_display.py -v`
Expected: 全部 19 个 PASS。

- [ ] **Step 5: 跑全量回归**

Run: `pytest -q`
Expected: 全绿。

- [ ] **Step 6: 手动冒烟（可选但推荐）**

启动 mini-cc，问一句会触发记忆读取的话（如「我的小名叫什么」后再「记住我的新小名 X」），确认：读时 live 行显示 `Recalling N memories…`，完成后 ChatLog 出现 `● Recalled N memories` / `● Wrote N memories` 单行，而非多条裸 `file_read/file_edit`。

- [ ] **Step 7: 提交**

```bash
git add src/mini_cc/consumers/tui/app.py tests/test_memory_op_display.py
git commit -m "feat(tui): collapse agent memory reads/writes into one status row

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review（计划对照 spec）

- **Spec 覆盖**：G1 读/G2 写/G3 搜 → Task 2 `classify_memory_op` + Task 3 `add_tool` 分流；G4 折叠/打断 → `MemoryRun` + `_flush_mem_run`（T12/T13/T14）；G5 并行前向兼容 → `MemoryRun` 锚定 absorb（注释 + 设计）；§3 文案 → `memory_run_summary`（T9-T11）；§4 全 16 用例 → 映射齐（T1-T4 Task1 / T5-T11 Task2 / T12-T16 Task2+3）；§7 提交划分 → 3 个 commit。
- **占位符**：无 TBD/TODO；每个改码 step 给完整代码。
- **类型/命名一致**：`MemoryRun.absorb/flush/live_summary/is_open`、`classify_memory_op`、`memory_run_summary`、`is_memory_path`、`_flush_mem_run`、`_mem_run` 在各 Task 间一致。
- **已知取舍**（spec §6）：sub-tool 不折叠（T15）；并行行序延后；搜索只认 `path`；不列文件名。

