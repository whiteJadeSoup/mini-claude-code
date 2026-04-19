"""Unit tests for persistence.py — _cwd_slug, _on_append, JSONL output."""
import json
from pathlib import Path

import pytest

from mini_cc import config
from mini_cc.consumers import persistence
from mini_cc.engine.messages import (
    AssistantMessage, CompactBoundaryMessage, SystemPromptMessage,
    TextBlock, ToolResultMessage, ToolUseBlock, UserMessage,
)


@pytest.fixture
def isolated_home(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(persistence, "SESSION_ID", "test-session-uuid")
    monkeypatch.setattr(config, "CWD", "/test/cwd")
    return tmp_path


# ---------------------------------------------------------------------------
# _cwd_slug
# ---------------------------------------------------------------------------

class TestCwdSlug:
    def test_windows_style_path(self, monkeypatch):
        monkeypatch.setattr(config, "CWD", r"D:\coding projects\build-mini-cc")
        assert persistence._cwd_slug() == "D--coding-projects-build-mini-cc"

    def test_posix_style_path(self, monkeypatch):
        monkeypatch.setattr(config, "CWD", "/home/user/my project")
        assert persistence._cwd_slug() == "home-user-my-project"

    def test_idempotent(self, monkeypatch):
        monkeypatch.setattr(config, "CWD", "/tmp/foo")
        assert persistence._cwd_slug() == persistence._cwd_slug()


# ---------------------------------------------------------------------------
# _on_append — subscriber
# ---------------------------------------------------------------------------

class TestOnAppend:
    def test_user_message_written(self, isolated_home):
        msg = UserMessage(content="hello")
        persistence._on_append(msg)
        path = persistence.transcript_path()
        assert path.exists()
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["type"] == "user"
        assert record["content"] == "hello"

    def test_system_prompt_written(self, isolated_home):
        msg = SystemPromptMessage(content="you are helpful", source="boot")
        persistence._on_append(msg)
        lines = persistence.transcript_path().read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["type"] == "system_prompt"

    def test_assistant_text_block_written(self, isolated_home):
        msg = AssistantMessage(turn_id="t1", model="m", content=TextBlock(text="hi"))
        persistence._on_append(msg)
        lines = persistence.transcript_path().read_text(encoding="utf-8").splitlines()
        record = json.loads(lines[0])
        assert record["type"] == "assistant"
        assert record["content"]["type"] == "text"
        assert record["content"]["text"] == "hi"

    def test_assistant_tool_use_block_written(self, isolated_home):
        msg = AssistantMessage(
            turn_id="t1", model="m",
            content=ToolUseBlock(call_id="c1", name="ls", args={"path": "."}),
        )
        persistence._on_append(msg)
        lines = persistence.transcript_path().read_text(encoding="utf-8").splitlines()
        record = json.loads(lines[0])
        assert record["content"]["type"] == "tool_use"
        assert record["content"]["call_id"] == "c1"

    def test_tool_result_written(self, isolated_home):
        msg = ToolResultMessage(content="output", tool_call_id="c1")
        persistence._on_append(msg)
        lines = persistence.transcript_path().read_text(encoding="utf-8").splitlines()
        record = json.loads(lines[0])
        assert record["type"] == "tool_result"
        assert record["tool_call_id"] == "c1"

    def test_compact_boundary_written(self, isolated_home):
        msg = CompactBoundaryMessage(pre_count=42, auto=True, source="compact")
        persistence._on_append(msg)
        lines = persistence.transcript_path().read_text(encoding="utf-8").splitlines()
        record = json.loads(lines[0])
        assert record["type"] == "compact_boundary"
        assert record["pre_count"] == 42
        assert record["auto"] is True

    def test_multiple_appends_accumulate(self, isolated_home):
        persistence._on_append(UserMessage(content="first"))
        persistence._on_append(AssistantMessage(turn_id="t1", model="m",
                                                 content=TextBlock(text="second")))
        persistence._on_append(ToolResultMessage(content="third", tool_call_id="c1"))
        lines = persistence.transcript_path().read_text(encoding="utf-8").splitlines()
        assert len(lines) == 3
        assert json.loads(lines[0])["content"] == "first"
        assert json.loads(lines[1])["content"]["text"] == "second"
        assert json.loads(lines[2])["content"] == "third"

    def test_is_synthetic_flag_preserved(self, isolated_home):
        msg = UserMessage(content="summary", is_synthetic=True, source="compact")
        persistence._on_append(msg)
        lines = persistence.transcript_path().read_text(encoding="utf-8").splitlines()
        record = json.loads(lines[0])
        assert record["is_synthetic"] is True

    def test_parent_id_preserved(self, isolated_home):
        msg = UserMessage(content="sub", parent_id="abc-123")
        persistence._on_append(msg)
        lines = persistence.transcript_path().read_text(encoding="utf-8").splitlines()
        assert json.loads(lines[0])["parent_id"] == "abc-123"

    def test_unicode_round_trip(self, isolated_home):
        msg = UserMessage(content="中文 🎉")
        persistence._on_append(msg)
        line = persistence.transcript_path().read_text(encoding="utf-8").splitlines()[0]
        assert json.loads(line)["content"] == "中文 🎉"
        assert "中文" in line  # ensure_ascii=False

    def test_parent_dir_created(self, isolated_home):
        persistence._on_append(UserMessage(content="x"))
        assert persistence.transcript_path().parent.exists()

    def test_io_error_silent(self, isolated_home, monkeypatch, capsys):
        def boom(*a, **kw):
            raise PermissionError("nope")
        monkeypatch.setattr(Path, "open", boom)
        persistence._on_append(UserMessage(content="x"))
        captured = capsys.readouterr()
        assert "persistence" in captured.err
