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
