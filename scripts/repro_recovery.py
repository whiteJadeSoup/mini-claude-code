"""Targeted reproduction: ask for non-existent files (LLM's training-data
guesses) and watch whether file_read NOT FOUND triggers shell-fallback or
glob-discovery.

Mimics the user's screenshot 3 failure mode.
"""
import asyncio
import sys

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from mini_cc import llm
from mini_cc.tools.base import get_tool


# Query designed to make the LLM guess non-existent file names. The pattern
# "memory_tools / task_tools / app_state" is what the LLM picked in screenshot 3
# — names plausible from generic agent-codebase priors but not in mini-cc.
QUERY = (
    "I want to understand how this codebase manages app state, memory tools, "
    "and the task subsystem. Can you walk me through the implementation?"
)


def _sum(tc) -> str:
    name = tc.get("name") if isinstance(tc, dict) else tc.name
    args = tc.get("args") if isinstance(tc, dict) else tc.args
    s = str(args)
    if len(s) > 180:
        s = s[:180] + "…"
    return f"{name}({s})"


def main() -> None:
    chat = llm._llm_base.bind_tools(llm.MAIN_TOOLS)
    sys_msg = SystemMessage(content=llm._build_system_prompt())
    msgs = [sys_msg, HumanMessage(content=QUERY)]

    for step in range(1, 12):
        print(f"--- Step {step} ---")
        resp = chat.invoke(msgs)
        msgs.append(resp)

        tcs = getattr(resp, "tool_calls", None) or []
        if not tcs:
            print(f"  (text — done): {str(resp.content)[:200]}")
            break

        for tc in tcs:
            print(f"  → {_sum(tc)}")

        for tc in tcs:
            name = tc.get("name") if isinstance(tc, dict) else tc.name
            args = tc.get("args") if isinstance(tc, dict) else tc.args
            tcid = tc.get("id") if isinstance(tc, dict) else tc.id
            tool = get_tool(name)
            if tool is None:
                msgs.append(ToolMessage(tool_call_id=tcid, content=f"Tool {name!r} not found."))
                continue
            out = asyncio.run(tool.execute(**args))
            content = out.to_api_str()
            if len(content) > 4000:
                content = content[:4000] + "\n…[truncated]"
            msgs.append(ToolMessage(tool_call_id=tcid, content=content))

    print()
    print("=== TOOL USAGE TALLY ===")
    used: dict[str, int] = {}
    for m in msgs:
        if isinstance(m, AIMessage):
            for tc in getattr(m, "tool_calls", []) or []:
                name = tc.get("name") if isinstance(tc, dict) else tc.name
                used[name] = used.get(name, 0) + 1
    for name, n in sorted(used.items(), key=lambda x: -x[1]):
        print(f"  {n:3}× {name}")


if __name__ == "__main__":
    sys.exit(main())
