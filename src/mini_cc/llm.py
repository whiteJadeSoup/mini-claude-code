"""LLM + tool-set configuration."""
from dotenv import load_dotenv
load_dotenv()  # must run before anything that reads env vars

import os
from typing import Any

from langchain_core.language_models import LanguageModelInput
from langchain_deepseek import ChatDeepSeek

from mini_cc import config, prompts
import mini_cc.tools  # noqa: F401 — side-effect: registers all MiniTools
from mini_cc.tools.base import get_tool
from mini_cc.skills import _skill_manager

_MODEL_NAME = "deepseek-v4-pro"


class _ChatDeepSeekRoundTrip(ChatDeepSeek):
    """ChatDeepSeek that always emits `reasoning_content` on outbound assistant messages.

    DeepSeek's thinking-mode models (deepseek-v4-pro, deepseek-reasoner) reject
    follow-up requests with HTTP 400 ("The `reasoning_content` in the thinking
    mode must be passed back to the API.") when a prior assistant message in
    the history is missing the field. The API is lenient about the value —
    empty string is accepted — but the field must be present.

    `langchain_openai._convert_message_to_dict` does not propagate
    `additional_kwargs.reasoning_content` to outbound dicts (it filters to a
    fixed allowlist of OpenAI-spec keys, see langchain_openai/chat_models/base.py
    module docstring), so we patch `_get_request_payload` to inject it.
    Defaults to "" when no upstream reasoning_content was captured.
    """

    def _get_request_payload(
        self,
        input_: LanguageModelInput,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict:
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        # Single-pass over outbound messages — no zip with input_, so this
        # works regardless of input shape (list[BaseMessage], PromptValue,
        # str, dict-form messages, etc.) and survives any future langchain
        # change that filters or coalesces messages between input and payload.
        # `setdefault` preserves any reasoning_content already present (e.g.
        # if a future engine change starts threading it through), otherwise
        # falls back to "" — DeepSeek validates field presence, not content.
        for dst in payload.get("messages", []):
            if isinstance(dst, dict) and dst.get("role") == "assistant":
                dst.setdefault("reasoning_content", "")
        return payload


def _build_system_prompt() -> str:
    """Dynamic — rebuilt each turn so newly discovered skills appear."""
    return prompts.build_system_prompt(_skill_manager.prompt_section())


def _lc(name: str):
    t = get_tool(name)
    assert t is not None, f"Tool '{name}' not registered"
    return t.as_langchain_tool()


_llm_base = _ChatDeepSeekRoundTrip(
    model=_MODEL_NAME,
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    stream_usage=True,
)

_sub_tool_names = [
    "plan_todos", "update_todo", "plan_tasks", "update_task",
    "execute_command", "file_read", "file_write", "file_edit",
]
if config.RG_PATH:
    _sub_tool_names += ["grep", "glob"]

SUB_TOOLS = [_lc(n) for n in _sub_tool_names]
MAIN_TOOLS = [_lc("task"), _lc("run_skill")] + SUB_TOOLS
