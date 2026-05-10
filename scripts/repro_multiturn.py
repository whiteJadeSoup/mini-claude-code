"""Multi-turn reproduction: simulate the user's session — broad exploration
task, multiple turns. See if grep/glob compliance degrades with conversation
length.
"""
import sys

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from mini_cc import config, llm
from mini_cc.tools.base import _REGISTRY, get_tool


# Simulate a multi-step exploration task
SCRIPT = [
    # Turn 1 — broad opening (user's typical "give me an overview" query)
    "Give me a comprehensive overview of this codebase: directory structure, "
    "core modules and their relationships, and how the agent loop works.",
]


def _summarise_tool_call(tc) -> str:
    name = tc.get("name") if isinstance(tc, dict) else tc.name
    args = tc.get("args") if isinstance(tc, dict) else tc.args
    arg_str = str(args)
    if len(arg_str) > 200:
        arg_str = arg_str[:200] + "…"
    return f"{name}({arg_str})"


def main() -> None:
    print("=== ENVIRONMENT ===")
    print(f"SUB_TOOLS order: {[t.name for t in llm.SUB_TOOLS]}")
    print()

    chat = llm._llm_base.bind_tools(llm.MAIN_TOOLS)
    sys_msg = SystemMessage(content=llm._build_system_prompt())

    msgs = [sys_msg, HumanMessage(content=SCRIPT[0])]

    # Loop until LLM stops calling tools or we hit a step cap
    for step in range(1, 12):
        print(f"--- Step {step} ---")
        resp = chat.invoke(msgs)
        msgs.append(resp)

        tcs = getattr(resp, "tool_calls", None) or []
        if not tcs:
            text = str(resp.content)
            print(f"  (text response — done) {text[:200]}")
            break

        for tc in tcs:
            print(f"  → {_summarise_tool_call(tc)}")

        # Execute tool calls so the conversation can continue (we want to
        # see whether long-running sessions cause drift, not just the first turn)
        for tc in tcs:
            name = tc.get("name") if isinstance(tc, dict) else tc.name
            args = tc.get("args") if isinstance(tc, dict) else tc.args
            tool = get_tool(name)
            if tool is None:
                msgs.append(ToolMessage(
                    tool_call_id=tc.get("id") if isinstance(tc, dict) else tc.id,
                    content=f"Tool {name!r} not found.",
                ))
                continue
            import asyncio
            out = asyncio.run(tool.execute(**args))
            content = out.to_api_str()
            if len(content) > 4000:
                content = content[:4000] + "\n…[truncated]"
            msgs.append(ToolMessage(
                tool_call_id=tc.get("id") if isinstance(tc, dict) else tc.id,
                content=content,
            ))

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
