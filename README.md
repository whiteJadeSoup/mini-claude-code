# mini-claude-code

A minimal coding agent built as a manual agentic loop — no framework wrappers.

## Running

```bash
# One-time (or after dependency changes)
uv venv
uv pip install -e ".[dev]"

# Run
uv run mini-cc

# Tests
uv run pytest -q
```

## Module Structure

```
src/mini_cc/
├─ __init__.py              — package version
├─ __main__.py              — entry point; amain() + main loop
├─ config.py                — CWD, safe_path, PLATFORM, BASH_PATH
├─ llm.py                   — LLM client setup + MAIN/SUB tool lists
├─ commands.py              — CommandRegistry, CommandContext, sync_skill_commands
├─ prompts.py               — build_system_prompt(), SUB_SYSTEM_PROMPT, COMPACT_PROMPT
├─ skills.py                — SkillManager + _skill_manager singleton
│
├─ engine/                  — core event loop (pure: no UI, no persistence)
│  ├─ query_engine.py       — QueryEngine: dispatch, query, run_sidechain, compact
│  ├─ agent_loop.py         — AgentLoop: per-turn LLM streaming + tool dispatch
│  ├─ messages.py           — Pydantic message types (Layer 1 API + Layer 2 UI)
│  ├─ store.py              — MessageStore (api_view, adjacency, layer classification)
│  ├─ subscription.py       — pub-sub consumer protocol
│  ├─ predicates.py         — message classification predicates
│  └─ transforms.py         — api_view construction helpers
│
├─ consumers/               — engine subscribers (read events, produce side effects)
│  ├─ persistence.py        — PersistenceConsumer; writes JSONL transcripts
│  └─ tui/
│     └─ app.py             — Textual TUI; renders chat, tool status, input
│
├─ state/                   — mutable app state (swapped per sub-agent scope)
│  ├─ tasks.py              — TaskManager + _tasks singleton (DAG-aware)
│  ├─ todos.py              — TodoManager + _todos singleton
│  └─ usage.py              — UsageTracker + _tracker
│
└─ tools/                   — tool definitions + MiniTool framework
   ├─ base.py               — MiniTool ABC, ToolOutput hierarchy, registry
   └─ builtins.py           — all 9 MiniTool implementations
```

Data dirs (not Python): `skills/<name>/SKILL.md`, `prompts/`.
