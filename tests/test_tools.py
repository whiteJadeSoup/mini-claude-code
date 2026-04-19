"""Tests for the MiniTool framework: Output types, registry, execute wrapping, render."""
import json
import pytest
from langchain_core.tools import StructuredTool

from mini_cc.tools.base import (
    ToolOutput, ToolErrorOutput,
    CommandOutput, FileWriteOutput, FileEditOutput,
    TodoPlanOutput, TodoUpdateOutput, TaskPlanOutput, TaskUpdateOutput,
    RunSkillOutput, SubTaskOutput,
    MiniTool, register, get_tool, _REGISTRY,
)
from mini_cc.engine.messages import ToolResultMessage


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------

class TestToolOutput:
    def test_base_is_not_error(self):
        out = ToolOutput()
        assert out.is_error is False

    def test_error_output_is_error(self):
        out = ToolErrorOutput(message="boom")
        assert out.is_error is True

    def test_error_output_to_api_str(self):
        out = ToolErrorOutput(message="something went wrong")
        assert out.to_api_str() == "something went wrong"

    def test_command_output_to_api_str_with_content(self):
        out = CommandOutput(stdout="hello\nworld", returncode=0)
        assert out.to_api_str() == "hello\nworld"

    def test_command_output_to_api_str_empty(self):
        out = CommandOutput(stdout="", returncode=0)
        assert out.to_api_str() == "(no output)"

    def test_file_write_output_to_api_str(self):
        out = FileWriteOutput(path="app.py", bytes_written=1024)
        assert "1024" in out.to_api_str()
        assert "app.py" in out.to_api_str()

    def test_file_edit_output_replaced(self):
        out = FileEditOutput(path="app.py", replaced=True)
        assert "Edited" in out.to_api_str()
        assert "app.py" in out.to_api_str()

    def test_file_edit_output_not_replaced(self):
        out = FileEditOutput(path="app.py", replaced=False)
        assert "not found" in out.to_api_str()

    def test_todo_plan_output(self):
        out = TodoPlanOutput(count=3, rendered="- a\n- b\n- c")
        assert out.to_api_str() == "- a\n- b\n- c"

    def test_todo_update_output(self):
        out = TodoUpdateOutput(item="write tests", status="done", rendered="[x] write tests")
        assert out.to_api_str() == "[x] write tests"

    def test_task_plan_output(self):
        out = TaskPlanOutput(count=2, rendered="tasks rendered")
        assert out.to_api_str() == "tasks rendered"

    def test_task_update_output(self):
        out = TaskUpdateOutput(task_id="t1", status="completed", rendered="done")
        assert out.to_api_str() == "done"

    def test_run_skill_output(self):
        out = RunSkillOutput(skill_name="search", result="found it")
        assert out.to_api_str() == "found it"

    def test_sub_task_output(self):
        out = SubTaskOutput(result="subtask done")
        assert out.to_api_str() == "subtask done"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestRegistry:
    def _make_tool(self, name: str) -> MiniTool:
        class _Tool(MiniTool):
            concurrency = False
            description = "test"
            prompt = "Use when: testing."

            async def _run(self, x: str) -> ToolOutput:
                return ToolOutput()

        t = _Tool()
        t.name = name
        return t

    def test_register_and_get(self):
        t = self._make_tool("_test_reg_basic")
        register(t)
        assert get_tool("_test_reg_basic") is t

    def test_get_unknown_returns_none(self):
        assert get_tool("_nonexistent_xyz") is None

    def test_register_overrides_existing(self):
        t1 = self._make_tool("_test_override")
        t2 = self._make_tool("_test_override")
        register(t1)
        register(t2)
        assert get_tool("_test_override") is t2


# ---------------------------------------------------------------------------
# as_langchain_tool — schema generation
# ---------------------------------------------------------------------------

class TestAsLangchainTool:
    def _make_tool(self):
        class _Tool(MiniTool):
            name = "_lc_test_tool"
            description = "test"
            prompt = "Use when: testing.\nDon't use for: production."

            async def _run(self, command: str, timeout: int = 120) -> CommandOutput:
                return CommandOutput(stdout="ok", returncode=0)

        return _Tool()

    def test_returns_structured_tool(self):
        t = self._make_tool()
        lc = t.as_langchain_tool()
        assert isinstance(lc, StructuredTool)

    def test_name_matches(self):
        t = self._make_tool()
        lc = t.as_langchain_tool()
        assert lc.name == "_lc_test_tool"

    def test_description_uses_prompt(self):
        t = self._make_tool()
        lc = t.as_langchain_tool()
        assert "Use when" in lc.description

    def test_args_schema_has_correct_fields(self):
        t = self._make_tool()
        lc = t.as_langchain_tool()
        fields = lc.args_schema.model_fields
        assert "command" in fields
        assert "timeout" in fields

    def test_args_schema_default_timeout(self):
        t = self._make_tool()
        lc = t.as_langchain_tool()
        timeout_field = lc.args_schema.model_fields["timeout"]
        assert timeout_field.default == 120


# ---------------------------------------------------------------------------
# execute() — exception wrapping
# ---------------------------------------------------------------------------

class TestExecuteWrapper:
    def _make_tool(self, raises=None, returns=None):
        _raises = raises
        _returns = returns

        class _Tool(MiniTool):
            name = "_exec_test"
            description = "test"
            prompt = "Use when: testing."

            async def _run(self, x: str) -> ToolOutput:
                if _raises is not None:
                    raise _raises
                return _returns or ToolOutput()

        return _Tool()

    @pytest.mark.asyncio
    async def test_normal_run_returns_output(self):
        out = CommandOutput(stdout="hello", returncode=0)
        t = self._make_tool(returns=out)
        result = await t.execute(x="hi")
        assert result is out

    @pytest.mark.asyncio
    async def test_exception_returns_tool_error_output(self):
        t = self._make_tool(raises=RuntimeError("test failure"))
        result = await t.execute(x="hi")
        assert isinstance(result, ToolErrorOutput)
        assert result.is_error is True
        assert "test failure" in result.message

    @pytest.mark.asyncio
    async def test_execute_never_raises(self):
        t = self._make_tool(raises=ValueError("bad value"))
        try:
            result = await t.execute(x="hi")
        except Exception:
            pytest.fail("execute() should not raise")
        assert result.is_error is True

    @pytest.mark.asyncio
    async def test_handle_error_can_be_overridden(self):
        class _CustomTool(MiniTool):
            name = "_custom_err"
            description = "test"
            prompt = "Use when: testing."

            async def _run(self, x: str) -> ToolOutput:
                raise KeyError("missing key")

            def handle_error(self, args: dict, error: Exception) -> ToolOutput:
                return ToolErrorOutput(message="custom error handler")

        t = _CustomTool()
        result = await t.execute(x="hi")
        assert result.message == "custom error handler"


# ---------------------------------------------------------------------------
# Render methods
# ---------------------------------------------------------------------------

class TestRenderMethods:
    def _make_tool(self):
        class _Tool(MiniTool):
            name = "_render_test"
            description = "test"
            prompt = "Use when: testing."

            async def _run(self, command: str, timeout: int = 120) -> CommandOutput:
                return CommandOutput(stdout="ok", returncode=0)

        return _Tool()

    def test_render_received_does_not_raise(self):
        t = self._make_tool()
        result = t.render_received({"command": "ls -la", "timeout": 30})
        assert isinstance(result, str)

    def test_render_executing_does_not_raise(self):
        t = self._make_tool()
        result = t.render_executing({"command": "ls"})
        assert isinstance(result, str)

    def test_render_complete_does_not_raise(self):
        t = self._make_tool()
        out = CommandOutput(stdout="ok", returncode=0)
        result = t.render_complete({"command": "ls"}, out)
        assert isinstance(result, str)

    def test_render_error_does_not_raise(self):
        t = self._make_tool()
        out = ToolErrorOutput(message="something failed")
        result = t.render_error({"command": "ls"}, out)
        assert isinstance(result, str)

    def test_render_error_includes_message(self):
        t = self._make_tool()
        out = ToolErrorOutput(message="permission denied")
        result = t.render_error({"command": "ls"}, out)
        assert "permission denied" in result

    def test_render_received_truncates_long_arg(self):
        t = self._make_tool()
        long_cmd = "x" * 100
        result = t.render_received({"command": long_cmd})
        assert len(result) <= 65   # 60 chars + "…" with some slack


# ---------------------------------------------------------------------------
# ToolResultMessage — output field
# ---------------------------------------------------------------------------

class TestToolResultMessageOutput:
    def test_output_defaults_to_none(self):
        msg = ToolResultMessage(content="ok", tool_call_id="c1")
        assert msg.output is None

    def test_output_accepts_tool_output(self):
        out = CommandOutput(stdout="hello", returncode=0)
        msg = ToolResultMessage(content="hello", tool_call_id="c1", output=out)
        assert msg.output is out
        assert msg.output.is_error is False

    def test_old_jsonl_without_output_deserializes(self):
        """Backward compat: records without 'output' key must still deserialize."""
        raw = {
            "id": "msg-1", "session_id": "s", "cwd": "/",
            "created_at": "2025-01-01T00:00:00+00:00",
            "type": "tool_result", "content": "done",
            "tool_call_id": "c99", "source": "agent",
        }
        msg = ToolResultMessage.model_validate(raw)
        assert msg.output is None
        assert msg.content == "done"

    def test_output_serializes_to_dict_in_jsonl(self):
        out = CommandOutput(stdout="hi", returncode=0)
        msg = ToolResultMessage(content="hi", tool_call_id="c1", output=out)
        dumped = json.dumps(msg.model_dump(mode="json"))
        loaded = json.loads(dumped)
        assert loaded["output"]["stdout"] == "hi"
        assert loaded["output"]["returncode"] == 0
