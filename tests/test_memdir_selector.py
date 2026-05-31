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
