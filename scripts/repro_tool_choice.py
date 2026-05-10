"""One-shot reproduction: send a query the LLM should answer with grep/glob,
print every tool call it makes. Bypasses the TUI and engine — just the LLM
+ tools + system prompt.

Usage:
    UV_NO_SYNC=1 uv run python scripts/repro_tool_choice.py
"""
import sys

from langchain_core.messages import HumanMessage, SystemMessage

from mini_cc import config, llm
from mini_cc.tools.base import _REGISTRY


QUERIES = [
    "List all Python files under src/mini_cc/.",
    "Find every place in the project where a class named *Output is defined.",
    "Show me the structure of src/mini_cc/ recursively.",
]


def main() -> None:
    print("=== ENVIRONMENT ===")
    print(f"RG_PATH:        {config.RG_PATH}")
    print(f"Registry tools: {sorted(_REGISTRY.keys())}")
    print(f"SUB_TOOLS:      {[t.name for t in llm.SUB_TOOLS]}")
    print()

    system_text = llm._build_system_prompt()
    chat = llm._llm_base.bind_tools(llm.MAIN_TOOLS)

    for i, q in enumerate(QUERIES, 1):
        print(f"=== QUERY {i}: {q!r} ===")
        msgs = [SystemMessage(content=system_text), HumanMessage(content=q)]
        try:
            resp = chat.invoke(msgs)
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            print()
            continue

        tcs = getattr(resp, "tool_calls", None) or []
        if not tcs:
            print(f"  (no tool calls — text response: {str(resp.content)[:200]!r})")
            print()
            continue

        for tc in tcs:
            name = tc.get("name") if isinstance(tc, dict) else tc.name
            args = tc.get("args") if isinstance(tc, dict) else tc.args
            arg_str = str(args)
            if len(arg_str) > 200:
                arg_str = arg_str[:200] + "…"
            print(f"  → {name}({arg_str})")
        print()


if __name__ == "__main__":
    sys.exit(main())
