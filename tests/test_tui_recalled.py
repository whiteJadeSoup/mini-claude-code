"""TUI 的「召回 N 条记忆」单行反馈（CC parity）。纯函数，不挂 Textual app。"""
from mini_cc.consumers.tui.app import _recalled_markup


def test_recalled_markup_plural():
    assert _recalled_markup(["user_role.md", "feedback_testing.md"]) == (
        "[dim]※ recalled 2 memories: user_role.md, feedback_testing.md[/dim]"
    )


def test_recalled_markup_singular():
    assert _recalled_markup(["user_role.md"]) == (
        "[dim]※ recalled 1 memory: user_role.md[/dim]"
    )
