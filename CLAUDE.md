# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Agent

```bash
# One-time (or after dependency changes)
uv venv
uv pip install -e ".[dev]"

# Run
uv run mini-cc
# or equivalently
uv run python -m mini_cc

# Tests
uv run pytest -q
```

Dependencies are declared in `pyproject.toml`. `uv` is installed at `C:\Users\dzshu\.local\bin\uv.exe`.

## Design Overview

A minimal coding agent built as a manual agentic loop — no framework wrappers. Uses the `src/` layout; imports resolve against the installed package, not the working copy.

### Module structure

```
src/mini_cc/
├─ __init__.py              — package version
├─ __main__.py              — `python -m mini_cc` entry; amain() + main loop
├─ config.py                — CWD, safe_path, PLATFORM, BASH_PATH
├─ llm.py                   — LLM client setup + MAIN/SUB tool lists
├─ commands.py              — CommandRegistry, CommandContext, sync_skill_commands
├─ prompts.py               — build_system_prompt(), SUB_SYSTEM_PROMPT, COMPACT_PROMPT
├─ engine/
│  ├─ query_engine.py       — QueryEngine: dispatch, query, run_sidechain, compact
│  ├─ agent_loop.py         — AgentLoop: per-turn LLM streaming + tool dispatch
│  ├─ messages.py           — Pydantic message types
│  └─ store.py              — MessageStore (api_view, adjacency, layer classification)
├─ consumers/
│  └─ persistence.py        — PersistenceConsumer, transcript_path()
├─ state/
│  ├─ tasks.py              — TaskManager + _tasks singleton (DAG-aware)
│  ├─ todos.py              — TodoManager + _todos singleton
│  └─ usage.py              — UsageTracker + _tracker
└─ tools/
   ├─ builtins.py           — all @tool functions, _sub_agent_scope
   └─ skills.py             — SkillManager + _skill_manager singleton
```

Data dirs (not Python): `skills/<name>/SKILL.md`, `prompts/`.

### Key conventions

- **`task` lazy import**: `task` in `tools/builtins.py` does `from mini_cc import llm` inside the function body to break the circular dep with `llm.py` importing tool lists.
- **`todos._todos` / `tasks._tasks` / `usage._tracker` access**: all are reassigned (not just mutated) by `_sub_agent_scope` for sub-agent isolation. Always access as `module._name`, never via `from state.X import _name`.
- **`load_dotenv()` ordering**: must run before any project imports that read env vars — see `llm.py` and `__main__.py`.
- **Tool docstrings**: Google-style with "Use when / Don't use for" guidance and concrete examples.
- **Skill system**: skills live in `skills/<name>/SKILL.md` with YAML frontmatter (`name`, `description`). Frontmatter is rescanned each turn. `run_skill` executes skills in a sub-agent (body as system prompt, never enters main history). Users invoke with `/skill-name <request>`.
- **Module vs data dir naming**: `mini_cc.tools.skills` (Python module) and `./skills/` (data dir) share a name but live in different namespaces.

## Planning

Follow the Design Plan Guide in `~/.claude/CLAUDE.md`. Required structure:

**Overview Design** (6 sections in order):
1. Problem Statement — pain point only, no solution
2. Goals & Non-Goals — goals must be verifiable; non-goals prevent scope creep
3. Architecture Overview — diagram with solid/dashed/gray for existing/new/removed; group by responsibility
4. Core Flows — sequence/flow diagrams; module-level participants, no function names
5. Key Decisions & Alternatives — title states what was chosen; body explains rationale; include rejected alternatives
6. Risks & Open Questions

**Detail Design** (7 sections):
1. Module Responsibilities & Interfaces — full contract, mark incremental changes
2. Data Model / Schema — fields, constraints, query patterns, migration
3. Core Algorithms / State Machines — only non-obvious logic; include invariants
4. Error Handling — per external interaction point: failure modes, handling strategy
5. Key Flows — same as overview but with function-level detail
6. Performance Estimation — only when goals have quantitative targets
7. Testing Strategy — what/how/pass criteria; mock boundaries and why

**Writing rules**: every section traces to a design goal; diagrams over prose; record *why* not *what*.

## Code Style

Keep implementation cohesive and extensible. Group related operations into manager classes (e.g. `SkillManager` for skill load/lazy-load/unload). Encapsulate distinct logic blocks into focused functions. Prefer Pydantic models over plain dicts for structures with defined fields that are passed across functions. Comments explain **why**, not what or how — let the code speak for itself.
