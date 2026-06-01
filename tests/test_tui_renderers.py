"""renderers.py：声明式 dispatch 注册表 + 纯呈现 helper（refactor 行为保真）。"""
from unittest.mock import MagicMock

from mini_cc.consumers.tui.renderers import _recalled_markup, dispatch
from mini_cc.engine.messages import (
    RelevantMemoryMessage,
    SurfacedMemory,
    UserMessage,
)


def _sm(filename="user_role.md"):
    return SurfacedMemory(filename=filename, path=f"/abs/{filename}",
                          content="x", mtime_ms=1, line_count=1,
                          header=f"Memory: {filename}:")


def test_recalled_markup_plural():
    assert _recalled_markup(["user_role.md", "feedback_testing.md"]) == (
        "[dim]※ recalled 2 memories: user_role.md, feedback_testing.md[/dim]"
    )


def test_recalled_markup_singular():
    assert _recalled_markup(["user_role.md"]) == (
        "[dim]※ recalled 1 memory: user_role.md[/dim]"
    )


def test_dispatch_routes_registered_type():
    # 注册类型 → 路由到对应 renderer（用 MagicMock app，无需起 Textual）。
    app = MagicMock()
    dispatch(app, RelevantMemoryMessage(memories=[_sm("a.md")]))
    app.chat_log.append_markup.assert_called_once_with(_recalled_markup(["a.md"]))


def test_dispatch_unregistered_type_is_noop():
    # 未注册类型 → no-op（等价旧 if-elif 落空，不报错、不渲染）。
    app = MagicMock()
    dispatch(app, UserMessage(content="hi", source="user"))
    app.chat_log.append_markup.assert_not_called()
