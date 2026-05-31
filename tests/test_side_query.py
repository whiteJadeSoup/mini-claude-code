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
