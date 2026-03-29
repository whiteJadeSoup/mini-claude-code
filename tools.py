import subprocess, os, locale, re
from contextlib import contextmanager
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage

import config
import todos
import skills
import usage
import prompts
from todos import TodoManager
from usage import UsageTracker


@contextmanager
def _sub_agent_scope(label: str):
    """Isolate a sub-agent by swapping global singletons; merges usage on exit."""
    saved_todos, saved_tracker = todos._todos, usage._tracker
    todos._todos = TodoManager()
    usage._tracker = UsageTracker()
    try:
        yield
    finally:
        sub_tracker = usage._tracker
        todos._todos = saved_todos
        usage._tracker = saved_tracker
        usage._tracker.merge_sub(label, sub_tracker)


_SENSITIVE_PATTERNS = [
    r'\.env\b',
    r'credentials\.json\b',
    r'\.pem\b',
    r'id_rsa\b',
    r'id_ed25519\b',
]
_SENSITIVE_RE = re.compile("|".join(_SENSITIVE_PATTERNS), re.I)


@tool
def execute_command(command: str, timeout: int = 120) -> str:
    """Run a shell command and return stdout + stderr.

    Use when: running scripts, installs, git operations, reading/writing/searching files, or any shell task.
    Don't use for: targeted edits to existing files — use edit_file instead.

    Args:
        command: The shell command to execute.
        timeout: Max seconds to wait (default 120). Increase for long-running commands like installs.

    Examples:
        execute_command("cat src/main.py")
        execute_command("head -50 src/main.py")
        execute_command("grep -rn 'TODO' src/")
        execute_command("find . -name '*.py'")
        execute_command("cat <<'EOF' > hello.py\\nprint('hello')\\nEOF")
        execute_command("python script.py")
        execute_command("pip install requests", timeout=300)
    """
    if _SENSITIVE_RE.search(command):
        return "Error: command references a sensitive file and was blocked."
    try:
        if config.BASH_PATH:
            result = subprocess.run(
                [config.BASH_PATH, "-c", command],
                cwd=config.CWD, capture_output=True, timeout=timeout,
            )
        else:
            result = subprocess.run(
                command, shell=True, cwd=config.CWD,
                capture_output=True, timeout=timeout,
            )
    except subprocess.TimeoutExpired:
        return f"Error: command timed out after {timeout}s"
    except subprocess.SubprocessError as e:
        return f"Error: {e}"
    raw = result.stdout + result.stderr
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode(locale.getpreferredencoding(False), errors="replace")
    return text.strip() or "(no output)"


@tool
def write_file(path: str, content: str) -> str:
    """Create or fully overwrite a file with the given content.

    Use when: creating a new file, or replacing an entire file's content.
    Don't use for: small targeted edits to existing files — use edit_file instead.

    Args:
        path: Relative path to the file to create or overwrite.
        content: The full text content to write to the file.

    Examples:
        write_file("hello.py", "print('hello')")
        write_file("config/settings.json", '{"debug": true}')
    """
    p = config.safe_path(path)
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Written to {path}"


@tool
def edit_file(path: str, old_string: str, new_string: str) -> str:
    """Replace the first occurrence of old_string with new_string in a file.

    Use when: making a targeted edit to an existing file (fix a bug, rename, update a value).
    Don't use for: creating new files or rewriting entire content — use write_file instead.

    Args:
        path: Relative path to the file to edit.
        old_string: Exact text to find — must match including whitespace and indentation.
        new_string: Text to replace old_string with.

    Examples:
        edit_file("app.py", "def foo():", "def bar():")
        edit_file("config.json", '"debug": false', '"debug": true')
        edit_file("README.md", "## Old Title", "## New Title")
    """
    p = config.safe_path(path)
    with open(p, "r", encoding="utf-8") as f:
        content = f.read()
    if old_string not in content:
        return f"Error: old_string not found in {path}"
    with open(p, "w", encoding="utf-8") as f:
        f.write(content.replace(old_string, new_string, 1))
    return f"Edited {path}"


@tool
def plan_todos(items: list[str]) -> str:
    """Set the full TODO list for the current task, replacing any existing list.

    Use when: at the start of a task, to lay out all planned steps before executing.
    Don't use for: updating progress mid-task — use update_todo instead.

    Args:
        items: Ordered list of step descriptions, all set to pending.

    Examples:
        plan_todos(["Read existing code", "Add feature", "Test the change"])
        plan_todos(["List files", "Write script", "Run script"])
    """
    return todos._todos.plan(items)


@tool
def update_todo(item: str, status: str) -> str:
    """Update the status of an existing TODO item.

    Use when: marking a step in_progress before starting it, or done after completing it.
    Don't use for: adding new steps — use plan_todos to set the full list upfront.

    Args:
        item: Exact description of the step (must match an item in the list).
        status: One of "pending", "in_progress", or "done".

    Examples:
        update_todo("Read existing code", "in_progress")
        update_todo("Read existing code", "done")
    """
    return todos._todos.update(item, status)


@tool
def run_skill(name: str, request: str, context: str = "") -> str:
    """Execute a skill in a sub-agent with isolated context.

    Use when: the user's request matches a skill listed in the system prompt.
    Don't use for: tasks not covered by available skills; don't guess skill names.

    Args:
        name: Skill name exactly as listed in Available Skills.
        request: What the user wants to accomplish.
        context: Optional conversation context the sub-agent needs but can't see.
            Summarize relevant details from the conversation. Omit if the request is self-contained.

    Examples:
        run_skill("wenyan", "写一篇关于春天的散文")
        run_skill("wenyan", "写一篇技术总结", context="项目使用 FastAPI + PostgreSQL，实现了用户认证模块")
    """
    body = skills._skill_manager.body(name)
    if body is None:
        available = ", ".join(skills._skill_manager.names()) or "(none)"
        return f"Skill '{name}' not found. Available: {available}"

    import agent  # lazy import — same circular-dep pattern as task
    with _sub_agent_scope(f"skill:{name}"):
        sub_llm = agent._llm_base.bind_tools(agent.SUB_TOOLS)
        user_msg = f"{request}\n\n## Context\n{context}" if context else request
        history = [SystemMessage(content=body), HumanMessage(content=user_msg)]
        print(f"\n  [skill: {name}]")
        response = agent._run_loop(sub_llm, history, agent.SUB_TOOLS_BY_NAME, prefix="  ")
    return response.content or "(completed, no summary)"


@tool
def task(description: str) -> str:
    """Delegate a self-contained subtask to a sub-agent with fresh context.

    Use when: the subtask is independent and doesn't need the current conversation history.
    Don't use for: simple single-step actions you can do directly.

    Args:
        description: Clear description of what the sub-agent should accomplish.

    Examples:
        task("Write a hello.py file and run it")
        task("Find all TODO comments in the codebase and list them")
    """
    import agent  # lazy import — agent imports tools, so we break the cycle here
    with _sub_agent_scope(description):
        sub_llm = agent._llm_base.bind_tools(agent.SUB_TOOLS)
        history = [SystemMessage(content=prompts.SUB_SYSTEM_PROMPT), HumanMessage(content=description)]
        print(f"\n  [sub-agent: {description}]")
        response = agent._run_loop(sub_llm, history, agent.SUB_TOOLS_BY_NAME, prefix="  ")
    return response.content or "(completed, no summary)"
