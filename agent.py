# /// script
# requires-python = ">=3.11"
# dependencies = ["langchain-openai", "python-dotenv", "pyyaml", "rich"]
# ///
import sys
sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
load_dotenv()  # must run before importing anything that reads env vars

import os, json, re
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
from langchain_openai import ChatOpenAI

import config
import todos
import usage
from skills import _skill_manager
from printer import StreamPrinter
from tools import (execute_command, edit_file,
                   plan_todos, update_todo, run_skill, task, _sub_agent_scope)

# --- Prompts ---


def _build_system_prompt() -> str:
    """Dynamic — rebuilt each turn so newly discovered skills appear."""
    return (
        f"You are a coding agent in: {config.CWD}\n"
        f"Answer questions and chat normally without tools. Use tools only when the task actually requires running code, reading files, or taking action — not for simple questions or greetings.\n"
        f"\n"
        f"## execute_command\n"
        f"Use execute_command for all file operations and shell tasks. Examples (not exhaustive):\n"
        f"- Read files: cat, head, tail\n"
        f"- Search content: grep, rg, awk\n"
        f"- Find files: find, ls, tree\n"
        f"- Write/create files: cat <<'EOF' > file, echo, tee, mkdir\n"
        f"- Run scripts, installs, git, and any other commands\n"
        f"\n"
        f"## edit_file\n"
        f"Use edit_file only for targeted string replacements in existing files.\n"
        f"\n"
        f"## Workflow\n"
        f"Complex tasks (3+ steps): plan_todos → update_todo(in_progress) → work → update_todo(done).\n"
        f"Independent subtasks: use the task tool.\n"
        f"Domain knowledge: use run_skill to execute skills in isolated context."
        + _skill_manager.prompt_section()
    )
SUB_SYSTEM_PROMPT = (
    f"You are a sub-agent in: {config.CWD}\n"
    f"Complete the task using tools. Plan with plan_todos, track with update_todo.\n"
    f"When done, briefly summarize what was accomplished."
)

# --- LLM & Tools ---

# stream_usage=True: required for custom base_url — LangChain won't
# auto-enable usage tracking when not hitting OpenAI's own endpoint.
_llm_base = ChatOpenAI(
    model="deepseek-reasoner",
    base_url="https://api.deepseek.com",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    stream_usage=True,
)

# Sub-agents get basic tools only — no task or run_skill to prevent nesting.
SUB_TOOLS = [plan_todos, update_todo, execute_command, edit_file]
SUB_TOOLS_BY_NAME = {t.name: t for t in SUB_TOOLS}

MAIN_TOOLS = [task, run_skill] + SUB_TOOLS
MAIN_TOOLS_BY_NAME = {t.name: t for t in MAIN_TOOLS}
llm = _llm_base.bind_tools(MAIN_TOOLS)

# --- Context Management ---


def _clear_old_tool_results(history: list) -> None:
    """Layer 1: replace old ToolMessage content with [Cleared] before each LLM call.

    Keeps the most recent group of tool results intact so the LLM can see the
    outcome of its latest tool calls. Earlier results are large and no longer needed.
    """
    last_ai_idx = -1
    for i, msg in enumerate(history):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            last_ai_idx = i
    if last_ai_idx <= 0:
        return
    for i in range(1, last_ai_idx):
        msg = history[i]
        if isinstance(msg, ToolMessage) and not msg.content.startswith("[Cleared]"):
            history[i] = ToolMessage(content="[Cleared]", tool_call_id=msg.tool_call_id)


def _load_compact_prompt() -> str:
    path = os.path.join(config.CWD, "prompts", "compact.md")
    with open(path, encoding="utf-8") as f:
        return f.read()


def _format_history_for_summary(messages: list) -> str:
    parts = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            parts.append(f"Human: {msg.content}")
        elif isinstance(msg, ToolMessage):
            parts.append(f"Tool result: {msg.content}")
        elif isinstance(msg, AIMessage):
            if msg.content:
                parts.append(f"Assistant: {msg.content}")
            for tc in (msg.tool_calls or []):
                parts.append(f"  → {tc['name']}({json.dumps(tc.get('args', {}), ensure_ascii=False)})")
    return "\n\n".join(parts)


def _extract_summary(content: str) -> str:
    m = re.search(r'<summary>(.*?)</summary>', content, re.DOTALL)
    return m.group(1).strip() if m else content


def compact(history: list, custom_instructions: str = "") -> int:
    """Layer 2: summarize conversation via sub-agent and rebuild history.

    Returns the number of messages removed. Raises on LLM failure so the
    caller can decide whether to abort or continue with unmodified history.
    """
    prompt = _load_compact_prompt()
    formatted = _format_history_for_summary(history[1:])  # skip SystemMessage
    if custom_instructions:
        formatted += f"\n\n## Compact Instructions\n{custom_instructions}"
    system_msg = history[0]
    usage._tracker.reset()
    response = None
    with _sub_agent_scope("compact"):
        messages = [SystemMessage(content=prompt), HumanMessage(content=formatted)]
        response = _llm_base.invoke(messages)
        usage._tracker.record("compact", response.usage_metadata,
                              getattr(response, "response_metadata", None))
    summary = _extract_summary(response.content)
    original_len = len(history)
    history.clear()
    history.append(system_msg)
    history.append(HumanMessage(content=f"[Previous conversation summary]\n\n{summary}"))
    history.append(AIMessage(content="Understood. I have the context. How can I help?"))
    return max(0, original_len - len(history))


# --- Loop ---


def _run_loop(bound_llm, history, tools_by_name, prefix="", source="agent"):
    while True:
        _clear_old_tool_results(history)
        if usage._tracker.context_tokens_used() > usage._tracker.context_limit * 0.8:
            print(f"\n{prefix}[Context at 80%, compacting...]", flush=True)
            try:
                n = compact(history)
                print(f"{prefix}[Compacted: {n} messages removed]")
                history.append(HumanMessage(content="Continue with the task described above."))
            except Exception as e:
                print(f"{prefix}[Auto-compact failed: {e}]")
        full_response = None
        tool_call_started = False
        printer = StreamPrinter(prefix)

        try:
            for chunk in bound_llm.stream(history):
                # LangChain chunks support __add__ — accumulates content + tool calls
                full_response = chunk if full_response is None else full_response + chunk
                if chunk.content:
                    printer.write(chunk.content)
                for tc in chunk.tool_call_chunks:
                    if tc.get("name"):
                        tool_call_started = True
                        printer.tool_start(tc["name"])
                    if tc.get("args"):
                        printer.tool_args(tc["args"])
        except Exception as e:
            printer.newline()
            raise RuntimeError(f"LLM stream error: {e}") from e

        if full_response is None:
            printer.newline()
            raise RuntimeError("LLM returned empty response")

        if tool_call_started:
            printer.tool_end()
        history.append(full_response)
        usage._tracker.record(source, full_response.usage_metadata,
                              getattr(full_response, "response_metadata", None))

        if not full_response.tool_calls:
            printer.newline()
            return full_response

        for tc in full_response.tool_calls:
            name = tc["name"]
            tool_fn = tools_by_name.get(name)
            if tool_fn is None:
                result = f"Error: unknown tool '{name}'. Available: {', '.join(tools_by_name)}"
            else:
                try:
                    result = tool_fn.invoke(tc["args"])
                except Exception as e:
                    result = f"Error running {name}: {e}"
            printer.tool_result(str(result))
            history.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

        # Next LLM call is a continuation after tool results
        names = [tc["name"] for tc in full_response.tool_calls]
        source = "tool: " + ", ".join(names)


def run_agent(user_message: str, history: list):
    history.append(HumanMessage(content=user_message))
    try:
        _run_loop(llm, history, MAIN_TOOLS_BY_NAME)
    except Exception as e:
        print(f"\n[Error: {e}]")


# --- Main ---

def _refresh_skills(history):
    """Rescan skills dir; sync commands and system prompt if anything changed."""
    from commands import sync_skill_commands
    old_names = set(_skill_manager.names())
    _skill_manager.rescan()
    new_names = set(_skill_manager.names())
    changed = sync_skill_commands(_skill_manager)
    if changed:
        history[0] = SystemMessage(content=_build_system_prompt())
        added = new_names - old_names
        removed = old_names - new_names
        if added:
            print(f"  [skills added: {', '.join(sorted(added))}]")
        if removed:
            print(f"  [skills removed: {', '.join(sorted(removed))}]")


if __name__ == "__main__":
    from commands import registry, sync_skill_commands, CommandContext
    sync_skill_commands(_skill_manager)

    history = [SystemMessage(content=_build_system_prompt())]
    ctx = CommandContext(
        history=history,
        tracker=usage._tracker,
        run_agent=run_agent,
        compact=compact,
        system_prompt_builder=_build_system_prompt,
    )

    if _skill_manager.names():
        print(f"Skills: {', '.join(f'/{n}' for n in _skill_manager.names())}")

    try:
        while True:
            _refresh_skills(history)
            user_msg = input("You: ").strip()
            if not user_msg:
                continue
            if user_msg.startswith("/"):
                cmd_name, *rest = user_msg[1:].split(maxsplit=1)
                args = rest[0] if rest else ""
                if not registry.handle(cmd_name, args, ctx):
                    print(f"Unknown command: /{cmd_name}")
                if ctx.should_exit:
                    break
                continue
            if usage._tracker.context_tokens_used() > usage._tracker.context_limit * 0.8:
                print("\n[Context at 80%, compacting...]", flush=True)
                try:
                    n = compact(history)
                    print(f"[Compacted: {n} messages removed]")
                except Exception as e:
                    print(f"[Auto-compact failed: {e}]")
            print("Agent: ", end="", flush=True)
            run_agent(user_msg, history)
    except (KeyboardInterrupt, EOFError):
        print()
