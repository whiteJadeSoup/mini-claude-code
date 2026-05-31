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
