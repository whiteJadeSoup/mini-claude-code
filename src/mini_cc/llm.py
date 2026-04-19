"""LLM + tool-set configuration."""
from dotenv import load_dotenv
load_dotenv()  # must run before anything that reads env vars

import os

from langchain_openai import ChatOpenAI

from mini_cc import prompts
from mini_cc.tools.builtins import (
    edit_file,
    execute_command,
    plan_tasks,
    plan_todos,
    run_skill,
    task,
    update_task,
    update_todo,
    write_file,
)
from mini_cc.tools.skills import _skill_manager

_MODEL_NAME = "deepseek-reasoner"


def _build_system_prompt() -> str:
    """Dynamic — rebuilt each turn so newly discovered skills appear."""
    return prompts.build_system_prompt(_skill_manager.prompt_section())


_llm_base = ChatOpenAI(
    model=_MODEL_NAME,
    base_url="https://api.deepseek.com",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    stream_usage=True,
)

SUB_TOOLS = [
    plan_todos, update_todo, plan_tasks, update_task,
    execute_command, write_file, edit_file,
]
MAIN_TOOLS = [task, run_skill] + SUB_TOOLS
