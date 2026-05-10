# scripts/ — Diagnostic harness

One-shot Python scripts for reproducing tool-routing and engine bugs
without the TUI. Each one boots `mini_cc.llm` (and optionally
`QueryEngine`) and drives a scripted user query end-to-end so behavior
can be observed in stderr/stdout.

Run any of them via:

```bash
UV_NO_SYNC=1 uv run python scripts/<name>.py
```

(`UV_NO_SYNC=1` skips the editable-install rebuild — useful when a long-
running mini-cc TUI process is holding `mini-cc.exe` open on Windows.)

## When to use which

| Script | When | What it shows |
|---|---|---|
| `repro_tool_choice.py` | Quick sanity check: "does the LLM pick the right tool for a single ask?" | One-shot LLM call; prints the tool_calls returned for 3 representative queries. No engine, no state. |
| `repro_multiturn.py` | Multi-turn drift: "does behavior degrade after many turns?" | Drives 11+ steps of an exploration query, executes tool calls, tallies tool usage. Goes through `llm._llm_base.bind_tools` but not through QueryEngine — i.e. **no `_clear_old_tool_results`**. |
| `repro_recovery.py` | Targeted failure mode: "what does the LLM do when file_read returns NOT FOUND?" | Same harness as multiturn but with a query designed to make the LLM guess non-existent file names. Use to verify the Discovery-first prompt prevents shell-fallback. |
| `repro_engine.py` | Full path verification: "what happens with the real `_clear_old_tool_results` + `_prepare_messages` flow?" | Boots a real `QueryEngine` and runs a single `query()`. Prints clear-hook firings, api_view summaries, and a final tally of file_read / glob / execute_command counts. **The right script for verifying engine-level fixes (compact, clear, dedup).** |

## Adding a new repro

Drop a new `repro_<name>.py` here, mirror the existing harness pattern
(System+Human messages → `chat.invoke` or `engine.query`), and add a
row to the table above. Keep them as throw-away verification harnesses
— not part of the test suite.
