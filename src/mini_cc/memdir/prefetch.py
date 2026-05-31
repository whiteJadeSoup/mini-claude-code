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
        # 停止条件：遇到上一个真实 user turn（非 synthetic、非当前）
        if isinstance(m, UserMessage) and not m.is_synthetic and m.id != current_user_id:
            break
        if isinstance(m, AssistantMessage) and isinstance(m.content, ToolUseBlock):
            use_to_name[m.content.call_id] = m.content.name
        elif isinstance(m, ToolResultMessage):
            errored[m.tool_call_id] = bool(m.output is not None and m.output.is_error)
    # 保留 use_to_name 的插入顺序（倒序扫描 = reverse-chronological），结果确定。
    # `set - set` 会按哈希打乱、跨进程不稳定（PYTHONHASHSEED）→ recent_tools 进
    # selector prompt 的字节漂移、破坏 prompt cache。镜像 CC attachments.ts:2502
    # `[...succeeded].filter(t => !failed.has(t))`：dict 当有序集，filter 掉曾报错的。
    failed: set[str] = set()
    succeeded: dict[str, None] = {}
    for cid, name in use_to_name.items():
        e = errored.get(cid)
        if e is None:        # 该 tool_use 没有配对的 result → 跳过（同 CC undefined）
            continue
        if e:
            failed.add(name)
        else:
            succeeded.setdefault(name, None)
    return tuple(n for n in succeeded if n not in failed)


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
        header=memory_header(h.filename, mtime),
    )


async def surface_relevant(
    query: str, memdir: Path, *,
    recent_tools: Sequence[str] = (),
    already_surfaced: frozenset[str] = frozenset(),
) -> list[SurfacedMemory]:
    """① 的 coro：选 top-N（Layer 1）→ 读+截断（⑤）→ 结构化 SurfacedMemory。

    find_relevant_memories 是模块级名字（from selector import），
    测试用 monkeypatch.setattr(pf_mod, "find_relevant_memories", ...) 替换。
    """
    headers = await find_relevant_memories(
        query, memdir, recent_tools=recent_tools, already_surfaced=already_surfaced)
    out: list[SurfacedMemory] = []
    for h in headers:
        sm = read_and_truncate(h)
        if sm is not None:
            out.append(sm)
    return out
