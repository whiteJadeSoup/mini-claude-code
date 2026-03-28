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
| `agent.py` | Entry point — LLM setup, prompts, `_run_loop()`, `run_agent()`, main loop |
| `commands.py` | `CommandRegistry` + `CommandContext` + built-in commands + `sync_skill_commands()` |
| `tools.py` | All `@tool` functions including `task` (uses lazy `import agent` to break circular dep) |
| `skills.py` | `SkillManager` class + `_skill_manager` singleton |
| `todos.py` | `TodoManager` class + `_todos` instance |
| `printer.py` | `StreamPrinter` + CJK spacing utilities |
| `config.py` | `CWD`, `safe_path()` |

### Key conventions

- **`task` lazy import**: `task` in `tools.py` does `import agent` inside the function body (not at module level) to avoid the circular dependency with `agent.py` importing from `tools.py`.
- **`todos._todos` access**: `_todos` is reassigned (not just mutated) by `task` for sub-agent isolation. Always access via `todos._todos` (module attribute), never `from todos import _todos`.
- **`load_dotenv()` ordering**: must be the first statement in `agent.py`, before any project imports that read env vars.
- **Tool docstrings**: Google-style with "Use when / Don't use for" guidance and concrete examples.
- **Skill system**: skills live in `skills/<name>/SKILL.md` with YAML frontmatter (`name`, `description`). Frontmatter is rescanned each turn for auto-discovery. `run_skill` executes skills in a sub-agent (body as system prompt, never enters main history). Users invoke with `/skill-name <request>`.

## Planning

Plans must lead with overview (with diagrams), design goals, and design principles before implementation details. Use diagrams liberally: architecture diagrams in overview, flow diagrams in implementation sections.

## Code Style

Keep implementation cohesive and extensible. Group related operations into manager classes (e.g. `SkillManager` for skill load/lazy-load/unload). Encapsulate distinct logic blocks into focused functions. Prefer Pydantic models over plain dicts for structures with defined fields that are passed across functions. Comments explain **why**, not what or how — let the code speak for itself.
