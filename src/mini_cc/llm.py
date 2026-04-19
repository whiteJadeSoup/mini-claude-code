"""LLM + tool-set configuration."""
from dotenv import load_dotenv
load_dotenv()  # must run before anything that reads env vars

import os

from langchain_openai import ChatOpenAI

from mini_cc import prompts
import mini_cc.tools.builtins  # noqa: F401 — side-effect: registers all MiniTools
from mini_cc.tools.base import get_tool
from mini_cc.tools.skills import _skill_manager

_MODEL_NAME = "deepseek-reasoner"


def _build_system_prompt() -> str:
    """Dynamic — rebuilt each turn so newly discovered skills appear."""
    return prompts.build_system_prompt(_skill_manager.prompt_section())


def _lc(name: str):
    t = get_tool(name)
    assert t is not None, f"Tool '{name}' not registered"
    return t.as_langchain_tool()


_llm_base = ChatOpenAI(
    model=_MODEL_NAME,
    base_url="https://api.deepseek.com",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    stream_usage=True,
)

SUB_TOOLS = [_lc(n) for n in (
    "plan_todos", "update_todo", "plan_tasks", "update_task",
    "execute_command", "write_file", "edit_file",
)]
MAIN_TOOLS = [_lc("task"), _lc("run_skill")] + SUB_TOOLS
