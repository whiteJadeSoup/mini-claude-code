"""Centralized prompt templates for the agent."""

from mini_cc import config


def _platform_line() -> str:
    shell = "bash" if config.BASH_PATH else ("cmd.exe" if config.PLATFORM == "Windows" else "sh")
    return f"Platform: {config.PLATFORM} ({shell})"


def _priority_table(available_tools: set[str]) -> str:
    """Layer-1 tool selection table. Conditional on rg-gated tools.

    Format keeps row labels at fixed indent so the model parses it as a table
    rather than prose. Bottom row is the residual fallback.
    """
    rows: list[tuple[str, str]] = [
        ("Read text",         "file_read"),
        ("Create / overwrite","file_write"),
        ("Edit",              "file_edit"),
    ]
    if "grep" in available_tools:
        rows.append(("Content search", "grep"))
    if "glob" in available_tools:
        rows.append(("Filename search", "glob"))
    rows.append(("Everything else", "execute_command"))

    width = max(len(label) for label, _ in rows)
    body = "\n".join(f"  {label:<{width}}  →  {tool}" for label, tool in rows)
    return f"Tool selection priority:\n{body}\n"


def _missing_rg_hint() -> str:
    """Install hint shown when rg is absent — keeps the LLM from chasing a
    grep/glob tool that isn't there.
    """
    if config.PLATFORM == "Windows":
        cmd = "winget install BurntSushi.ripgrep.MSVC"
    elif config.PLATFORM == "Darwin":
        cmd = "brew install ripgrep"
    else:
        cmd = "apt install ripgrep   # or your distro's equivalent"
    return (
        "Note: ripgrep is not detected on this system. Install it "
        f"(`{cmd}`) so that the dedicated `grep` and `glob` tools become "
        "available. Until then, fall back to `execute_command(\"rg ...\")`.\n"
    )


def build_system_prompt(
    skill_section: str = "",
    available_tools: set[str] | None = None,
) -> str:
    """Build the main agent system prompt.

    Rebuilt each turn so newly discovered skills appear.

    `available_tools` controls Layer-1 priority table rendering and per-tool
    section gating. Default reads from the live registry — pass an explicit
    set in tests.
    """
    if available_tools is None:
        from mini_cc.tools.base import _REGISTRY
        available_tools = set(_REGISTRY.keys())

    has_grep = "grep" in available_tools
    has_glob = "glob" in available_tools

    sections: list[str] = []
    sections.append(f"You are a coding agent in: {config.CWD}")
    sections.append(_platform_line())
    sections.append(
        "Answer questions and chat normally without tools. Use tools only when "
        "the task actually requires running code, reading files, or taking action "
        "— not for simple questions or greetings."
    )
    sections.append("")
    sections.append(_priority_table(available_tools))

    sections.append(
        "## file_read\n"
        "Use file_read to read text files with cat -n line numbers. Default "
        "reads up to 2000 lines from the start; use offset and limit for "
        "chunked continuation. Output line-number prefix is for navigation "
        "only — never include it in old_string for file_edit. NEVER read text "
        "files via execute_command(\"cat ...\") — file_read enforces the "
        "read-gate that downstream file_edit/file_write rely on."
    )
    sections.append(
        "## file_write\n"
        "Use file_write to create new files or fully overwrite existing files. "
        "For existing files, you MUST call file_read first — the read-gate "
        "will reject otherwise. Don't use it for small edits (use file_edit). "
        "NEVER overwrite via shell redirection (`echo > file`, `cat <<EOF`) — "
        "those bypass the read-gate."
    )
    sections.append(
        "## file_edit\n"
        "Use file_edit for targeted string replacements in existing files. You "
        "MUST call file_read first — the read-gate will reject otherwise. "
        "old_string must be unique in the file; if multiple matches exist, "
        "either add surrounding context to make it unique, or pass "
        "replace_all=true. NEVER edit via `sed -i` or `awk -i inplace` — "
        "those skip the uniqueness check and CRLF-aware staleness detection."
    )

    if has_grep:
        sections.append(
            "## grep\n"
            "Use grep for content search (regex via ripgrep). Three output_modes: "
            "files_with_matches (default, mtime-sorted paths), content "
            "(matching lines with -n/context), count (per-file totals). Filter "
            "by `type` (e.g. \"py\", \"js\") or `glob` (e.g. \"*.ts\"). "
            "head_limit defaults to 100 — pass 0 only when you genuinely need "
            "every match. NEVER invoke `grep` or `rg` via execute_command — "
            "the dedicated tool gives VCS exclusion, head_limit, and "
            "structured output that raw shell calls don't."
        )
    if has_glob:
        sections.append(
            "## glob\n"
            "Use glob for filename search (shell glob via ripgrep). Pattern "
            "uses `**`, `{a,b}` alternation, etc. Results are mtime-sorted "
            "(recent first), capped at 100 with `truncated` flag. NEVER use "
            "`find`, `ls -R`, or `git ls-files` via execute_command — glob "
            "handles VCS exclusion and capping that raw commands lack."
        )

    sections.append(
        "## execute_command\n"
        "Use only for shell tasks NOT covered by the dedicated tools above: "
        "scripts, installs, git, environment inspection, build commands."
    )
    if not (has_grep and has_glob):
        sections.append(_missing_rg_hint())

    sections.append(
        "## Workflow\n"
        "Simple checklists (no dependencies): plan_todos → update_todo(in_progress) → work → update_todo(done).\n"
        "Complex tasks with dependencies: plan_tasks (declare depends_on) → update_task(in_progress) → work → update_task(done).\n"
        "Independent subtasks: use the task tool.\n"
        "Domain knowledge: use run_skill to execute skills in isolated context."
    )

    return "\n\n".join(s.rstrip() for s in sections if s) + skill_section


SUB_SYSTEM_PROMPT = (
    f"You are a sub-agent in: {config.CWD}\n"
    f"{_platform_line()}\n"
    f"Complete the task using tools.\n"
    f"Simple checklists: plan_todos → update_todo. Tasks with dependencies: plan_tasks → update_task.\n"
    f"When done, briefly summarize what was accomplished."
)

COMPACT_PROMPT = """\
CRITICAL: Respond with TEXT ONLY. Do NOT call any tools.

- Do NOT use Read, Bash, Grep, Glob, Edit, Write, or ANY other tool.
- You already have all the context you need in the conversation above.
- Tool calls will be REJECTED and will waste your only turn — you will fail the task.
- Your entire response must be plain text: an <analysis> block followed by a <summary> block.

---

Your task is to create a detailed summary of the conversation so far, paying close attention to the user's explicit requests and your previous actions.
This summary should be thorough in capturing technical details, code patterns, and architectural decisions that would be essential for continuing development work without losing context.

Before providing your final summary, wrap your analysis in <analysis> tags to organize your thoughts and ensure you've covered all necessary points. In your analysis process:

1. Chronologically analyze each message and section of the conversation. For each section thoroughly identify:
   - The user's explicit requests and intents
   - Your approach to addressing the user's requests
   - Key decisions, technical concepts and code patterns
   - Specific details like:
     - file names
     - full code snippets
     - function signatures
     - file edits
   - Errors that you ran into and how you fixed them
   - Pay special attention to specific user feedback that you received, especially if the user told you to do something differently.
2. Double-check for technical accuracy and completeness, addressing each required element thoroughly.

Your summary should include the following sections:

1. Primary Request and Intent: Capture all of the user's explicit requests and intents in detail
2. Key Technical Concepts: List all important technical concepts, technologies, and frameworks discussed.
3. Files and Code Sections: Enumerate specific files and code sections examined, modified, or created. Pay special attention to the most recent messages and include full code snippets where applicable and include a summary of why this file read or edit is important.
4. Errors and fixes: List all errors that you ran into, and how you fixed them. Pay special attention to specific user feedback that you received, especially if the user told you to do something differently.
5. Problem Solving: Document problems solved and any ongoing troubleshooting efforts.
6. All user messages: List ALL user messages that are not tool results. These are critical for understanding the users' feedback and changing intent.
7. Pending Tasks: Outline any pending tasks that you have explicitly been asked to work on.
8. Current Work: Describe in detail precisely what was being worked on immediately before this summary request, paying special attention to the most recent messages from both user and assistant. Include file names and code snippets where applicable.
9. Optional Next Step: List the next step that you will take that is related to the most recent work you were doing. IMPORTANT: ensure that this step is DIRECTLY in line with the user's most recent explicit requests, and the task you were working on immediately before this summary request. If your last task was concluded, then only list next steps if they are explicitly in line with the users request. Do not start on tangential requests or really old requests that were already completed without confirming with the user first.
                       If there is a next step, include direct quotes from the most recent conversation showing exactly what task you were working on and where you left off. This should be verbatim to ensure there's no drift in task interpretation.

Here's an example of how your output should be structured:

<example>
<analysis>
[Your thought process, ensuring all points are covered thoroughly and accurately]
</analysis>

<summary>
1. Primary Request and Intent:
   [Detailed description]

2. Key Technical Concepts:
   - [Concept 1]
   - [Concept 2]
   - [...]

3. Files and Code Sections:
   - [File Name 1]
      - [Summary of why this file is important]
      - [Summary of the changes made to this file, if any]
      - [Important Code Snippet]
   - [File Name 2]
      - [Important Code Snippet]
   - [...]

4. Errors and fixes:
    - [Detailed description of error 1]:
      - [How you fixed the error]
      - [User feedback on the error if any]
    - [...]

5. Problem Solving:
   [Description of solved problems and ongoing troubleshooting]

6. All user messages:
    - [Detailed non tool use user message]
    - [...]

7. Pending Tasks:
   - [Task 1]
   - [Task 2]
   - [...]

8. Current Work:
   [Precise description of current work]

9. Optional Next Step:
   [Optional Next step to take]

</summary>
</example>

Please provide your summary based on the conversation so far, following this structure and ensuring precision and thoroughness in your response.

There may be additional summarization instructions provided in the included context. If so, remember to follow these instructions when creating the above summary. Examples of instructions include:
<example>
## Compact Instructions
When summarizing the conversation focus on typescript code changes and also remember the mistakes you made and how you fixed them.
</example>

<example>
# Summary instructions
When you are using compact - please focus on test output and code changes. Include file reads verbatim.
</example>

---

REMINDER: Do NOT call any tools. Respond with plain text only — an <analysis> block followed by a <summary> block. Tool calls will be rejected and you will fail the task."""
