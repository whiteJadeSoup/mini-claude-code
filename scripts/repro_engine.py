"""Engine-based reproduction: drives QueryEngine end-to-end so the
_prepare_messages → _clear_old_tool_results path actually fires.

Sample query designed to make the LLM repeatedly file_read the same set
of files (mimics user's screenshot 4-6 failure mode).
"""
import asyncio
import sys

from mini_cc.engine.query_engine import QueryEngine
from mini_cc import llm
from mini_cc.engine.messages import AssistantMessage, ToolResultMessage, ToolUseBlock


QUERY = (
    "Give me a comprehensive overview of this codebase: directory structure, "
    "core modules and their relationships, and how the agent loop works. "
    "Be thorough — read every important source file at least once."
)


async def main() -> None:
    engine = QueryEngine(
        llm_base=llm._llm_base,
        main_tools=llm.MAIN_TOOLS,
        sub_tools=llm.SUB_TOOLS,
        model_name="deepseek-v4-pro",
        system_prompt_builder=llm._build_system_prompt,
    )
    await engine.boot()
    await engine.query(QUERY)
    await engine.shutdown()

    # Final tally
    print("\n=== FINAL STATE ===", file=sys.stderr, flush=True)
    msgs = engine.store.all()
    tool_call_counter: dict[str, int] = {}
    file_read_results_full = 0
    file_read_results_unchanged_full = 0  # post-fix: dedup hit but content embedded
    file_read_results_unchanged_stub = 0  # legacy stub (should be 0 after fix)
    file_read_results_cleared = 0
    execute_command_calls = 0
    for m in msgs:
        if isinstance(m, AssistantMessage) and isinstance(m.content, ToolUseBlock):
            tool_call_counter[m.content.name] = tool_call_counter.get(m.content.name, 0) + 1
            if m.content.name == "execute_command":
                execute_command_calls += 1
        elif isinstance(m, ToolResultMessage):
            content = str(m.content)
            if content.startswith("[Cleared]"):
                file_read_results_cleared += 1
                continue
            out = m.output
            if out and getattr(out, "type", "") == "file_read":
                if getattr(out, "unchanged", False):
                    if "File unchanged since last read" in content:
                        file_read_results_unchanged_stub += 1
                    else:
                        file_read_results_unchanged_full += 1
                else:
                    file_read_results_full += 1
    print("Tool call distribution:", file=sys.stderr)
    for n, c in sorted(tool_call_counter.items(), key=lambda x: -x[1]):
        print(f"  {c:3}× {n}", file=sys.stderr)
    print(file=sys.stderr)
    print(f"file_read FULL fresh content results:        {file_read_results_full}", file=sys.stderr)
    print(f"file_read UNCHANGED with embedded content:   {file_read_results_unchanged_full}  (post-fix)", file=sys.stderr)
    print(f"file_read UNCHANGED stub-only (legacy):      {file_read_results_unchanged_stub}  (must be 0)", file=sys.stderr)
    print(f"tool_results cleared by engine:              {file_read_results_cleared}", file=sys.stderr)
    print(f"execute_command calls (shell fallback):      {execute_command_calls}", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
