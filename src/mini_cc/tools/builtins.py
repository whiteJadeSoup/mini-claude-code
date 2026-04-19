import asyncio, subprocess, os, locale, re, sys
from contextlib import contextmanager
from langchain_core.tools import tool

from mini_cc import config, prompts
from mini_cc.state import tasks, todos, usage
from mini_cc.state.tasks import TaskManager
from mini_cc.state.todos import TodoManager
from mini_cc.state.usage import UsageTracker
from mini_cc.tools import skills


@contextmanager
def _sub_agent_scope(label: str):
    """Isolate a sub-agent by swapping global singletons; merges usage on exit."""
    saved_todos, saved_tasks, saved_tracker = todos._todos, tasks._tasks, usage._tracker
    todos._todos = TodoManager()
    tasks._tasks = TaskManager()  # no persist_path → in-memory only, no DAG state leak
    usage._tracker = UsageTracker()
    try:
        yield
    finally:
        sub_tracker = usage._tracker
        todos._todos = saved_todos
        tasks._tasks = saved_tasks
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


def _kill_tree(pid: int) -> None:
    """Kill a process and all its descendants."""
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(pid)],
            capture_output=True,
        )
    else:
        try:
            os.killpg(os.getpgid(pid), 9)
        except (ProcessLookupError, OSError):
            pass


@tool
async def execute_command(command: str, timeout: int = 120) -> str:
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
            proc = await asyncio.create_subprocess_exec(
                config.BASH_PATH, "--login", "-c", command,
                cwd=config.CWD,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        else:
            proc = await asyncio.create_subprocess_shell(
                command,
                cwd=config.CWD,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
    except OSError as e:
        return f"Error: {e}"

    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        _kill_tree(proc.pid)
        proc.kill()
        return f"Error: command timed out after {timeout}s"
    except BaseException:
        # CancelledError or any unexpected error — kill process tree then re-raise
        _kill_tree(proc.pid)
        try:
            proc.kill()
        except Exception:
            pass
        raise

    raw = (stdout or b"") + (stderr or b"")
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
def plan_tasks(tasks_list: list[dict]) -> str:
    """Set a dependency-aware task graph, replacing any existing plan.

    Use when: complex work with 3+ steps where some steps have prerequisites.
    Don't use for: simple checklists with no dependencies — use plan_todos instead.

    Args:
        tasks_list: List of task dicts. Each dict must have:
            - id (str): unique identifier, used in depends_on references
            - description (str): what the task does
            - depends_on (list[str], optional): ids that must be done first

    Examples:
        plan_tasks([
            {"id": "read", "description": "Read existing code"},
            {"id": "impl", "description": "Implement feature", "depends_on": ["read"]},
            {"id": "test", "description": "Write tests", "depends_on": ["impl"]},
        ])
    """
    return tasks._tasks.plan(tasks_list)


@tool
def update_task(task_id: str, status: str) -> str:
    """Update the status of a task in the dependency graph.

    Use when: marking a task in_progress before starting it, or done after completing it.
    Don't use for: adding tasks — use plan_tasks to set the full graph upfront.

    Args:
        task_id: The id string of the task (as declared in plan_tasks).
        status: One of "pending", "in_progress", or "done".

    Note: starting a task whose dependencies aren't done will return an error.

    Examples:
        update_task("read", "in_progress")
        update_task("read", "done")
        update_task("impl", "in_progress")  # only valid after "read" is done
    """
    return tasks._tasks.update(task_id, status)


@tool
async def run_skill(name: str, request: str, context: str = "") -> str:
    """Execute a skill in a sub-agent with isolated context.

    Use when: the user's request matches a skill listed in the system prompt.
    Don't use for: tasks not covered by available skills; don't guess skill names.

    Args:
        name: Skill name exactly as listed in Available Skills.
        request: What the user wants to accomplish.
        context: Optional conversation context the sub-agent needs but can't see.
            Summarize relevant details from the conversation. Omit if the request is self-contained.
    """
    body = skills._skill_manager.body(name)
    if body is None:
        available = ", ".join(skills._skill_manager.names()) or "(none)"
        return f"Skill '{name}' not found. Available: {available}"

    # Lazy imports: query_engine imports tools indirectly via tool lists, so
    # importing it at module load would cycle. Same rationale as the previous
    # lazy `import agent` pattern.
    from mini_cc.engine.query_engine import get_engine
    from mini_cc.engine.store import _triggering_asst_id

    parent_id = _triggering_asst_id.get()
    if parent_id is None:
        return "Error: run_skill called outside a tool-dispatch context"

    user_content = f"{request}\n\n## Context\n{context}" if context else request
    return await get_engine().run_sidechain(
        parent_id=parent_id,
        system_prompt=body,
        user_prompt=user_content,
        label=f"skill:{name}",
    )


@tool
async def task(description: str) -> str:
    """Delegate a self-contained subtask to a sub-agent with fresh context.

    Use when: the subtask is independent and doesn't need the current conversation history.
    Don't use for: simple single-step actions you can do directly.

    Args:
        description: Clear description of what the sub-agent should accomplish.
    """
    from mini_cc.engine.query_engine import get_engine
    from mini_cc.engine.store import _triggering_asst_id

    parent_id = _triggering_asst_id.get()
    if parent_id is None:
        return "Error: task called outside a tool-dispatch context"

    return await get_engine().run_sidechain(
        parent_id=parent_id,
        system_prompt=prompts.SUB_SYSTEM_PROMPT,
        user_prompt=description,
        label="task",
    )
