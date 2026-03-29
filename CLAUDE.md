# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Agent

```bash
# With uv (preferred — handles dependencies via PEP 723 inline metadata)
uv run agent.py

# Or with a traditional venv
pip install langchain-openai python-dotenv pyyaml
python agent.py
```

uv is installed at `C:\Users\dzshu\.local\bin\uv.exe`. Add `$env:USERPROFILE\.local\bin` to PATH if `uv` is not found.

## Design Overview

A minimal coding agent built as a manual agentic loop — no framework wrappers.

### Module structure

| File | Responsibility |
|------|---------------|
| `agent.py` | Entry point — LLM setup, `_run_loop()`, `run_agent()`, main loop |
| `commands.py` | `CommandRegistry` + `CommandContext` + built-in commands + `sync_skill_commands()` |
| `tools.py` | All `@tool` functions including `task` (uses lazy `import agent` to break circular dep) |
| `skills.py` | `SkillManager` class + `_skill_manager` singleton |
| `todos.py` | `TodoManager` class + `_todos` instance |
| `prompts.py` | All prompt templates: `build_system_prompt()`, `SUB_SYSTEM_PROMPT`, `COMPACT_PROMPT` |
| `printer.py` | `StreamPrinter` + `ThinkingIndicator` + CJK spacing utilities |
| `config.py` | `CWD`, `safe_path()`, `PLATFORM`, `BASH_PATH` |

### Key conventions

- **`task` lazy import**: `task` in `tools.py` does `import agent` inside the function body (not at module level) to avoid the circular dependency with `agent.py` importing from `tools.py`.
- **`todos._todos` access**: `_todos` is reassigned (not just mutated) by `task` for sub-agent isolation. Always access via `todos._todos` (module attribute), never `from todos import _todos`.
- **`load_dotenv()` ordering**: must be the first statement in `agent.py`, before any project imports that read env vars.
- **Tool docstrings**: Google-style with "Use when / Don't use for" guidance and concrete examples.
- **Skill system**: skills live in `skills/<name>/SKILL.md` with YAML frontmatter (`name`, `description`). Frontmatter is rescanned each turn for auto-discovery. `run_skill` executes skills in a sub-agent (body as system prompt, never enters main history). Users invoke with `/skill-name <request>`.

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
