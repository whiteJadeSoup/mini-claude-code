import asyncio, subprocess, os, locale, re, sys
from contextlib import contextmanager

from mini_cc import config, prompts
from mini_cc.state import tasks, todos, usage
from mini_cc.state.tasks import TaskManager
from mini_cc.state.todos import TodoManager
from mini_cc.state.usage import UsageTracker
from mini_cc.tools import skills
from mini_cc.tools.base import (
    MiniTool, ToolOutput,
    CommandOutput, FileWriteOutput, FileEditOutput,
    TodoPlanOutput, TodoUpdateOutput, TaskPlanOutput, TaskUpdateOutput,
    RunSkillOutput, SubTaskOutput,
    register,
)


# ---------------------------------------------------------------------------
# Module-level utilities (not part of any tool)
# ---------------------------------------------------------------------------

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


def _size_str(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    return f"{n / 1024:.1f} KB"


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

class ExecuteCommandTool(MiniTool):
    name = "execute_command"
    description = "Run a shell command"
    prompt = """\
Use when: running scripts, installs, git operations, reading/writing/searching files, or any shell task.
Don't use for: targeted edits to existing files — use edit_file instead.

Examples:
    execute_command("cat src/main.py")
    execute_command("grep -rn 'TODO' src/")
    execute_command("python script.py")
    execute_command("pip install requests", timeout=300)
"""

    async def _run(self, command: str, timeout: int = 120) -> CommandOutput:
        if _SENSITIVE_RE.search(command):
            return CommandOutput(
                stdout="Error: command references a sensitive file and was blocked.",
                returncode=1,
            )
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
            return CommandOutput(stdout=f"Error: {e}", returncode=1)

        try:
            stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            _kill_tree(proc.pid)
            proc.kill()
            return CommandOutput(stdout=f"Error: command timed out after {timeout}s", returncode=1)
        except BaseException:
            # CancelledError or unexpected — kill then re-raise so execute() can catch
            _kill_tree(proc.pid)
            try:
                proc.kill()
            except Exception:
                pass
            raise

        raw = (stdout_b or b"") + (stderr_b or b"")
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode(locale.getpreferredencoding(False), errors="replace")
        return CommandOutput(stdout=text.strip() or "(no output)", returncode=proc.returncode)

    def render_received(self, args: dict) -> str:
        cmd = args.get("command", "")
        return (cmd[:57] + "…") if len(cmd) > 60 else cmd

    def render_complete(self, args: dict, output: ToolOutput | None) -> str:
        base = self.render_received(args)
        if not isinstance(output, CommandOutput):
            return base
        lines = [ln for ln in output.stdout.splitlines() if ln.strip()]
        if output.returncode != 0:
            return f"{base} · exit {output.returncode}"
        return f"{base} · {len(lines)} lines"


class WriteFileTool(MiniTool):
    name = "write_file"
    description = "Create or overwrite a file"
    prompt = """\
Use when: creating a new file, or replacing an entire file's content.
Don't use for: small targeted edits to existing files — use edit_file instead.

Examples:
    write_file("hello.py", "print('hello')")
    write_file("config/settings.json", '{"debug": true}')
"""

    async def _run(self, path: str, content: str) -> FileWriteOutput:
        p = config.safe_path(path)
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(content)
        return FileWriteOutput(path=path, bytes_written=len(content.encode("utf-8")))

    def render_received(self, args: dict) -> str:
        return args.get("path", "")

    def render_complete(self, args: dict, output: ToolOutput | None) -> str:
        path = args.get("path", "")
        if not isinstance(output, FileWriteOutput):
            return path
        return f"{path} · {_size_str(output.bytes_written)}"


class EditFileTool(MiniTool):
    name = "edit_file"
    description = "Edit a file in place"
    prompt = """\
Use when: making a targeted edit to an existing file (fix a bug, rename, update a value).
Don't use for: creating new files or rewriting entire content — use write_file instead.

Examples:
    edit_file("app.py", "def foo():", "def bar():")
    edit_file("config.json", '"debug": false', '"debug": true')
"""

    async def _run(self, path: str, old_string: str, new_string: str) -> FileEditOutput:
        p = config.safe_path(path)
        with open(p, "r", encoding="utf-8") as f:
            content = f.read()
        if old_string not in content:
            return FileEditOutput(path=path, replaced=False)
        with open(p, "w", encoding="utf-8") as f:
            f.write(content.replace(old_string, new_string, 1))
        return FileEditOutput(path=path, replaced=True)

    def render_received(self, args: dict) -> str:
        return args.get("path", "")

    def render_complete(self, args: dict, output: ToolOutput | None) -> str:
        path = args.get("path", "")
        if not isinstance(output, FileEditOutput):
            return path
        return f"{path} · edited" if output.replaced else f"{path} · not found"


class PlanTodosTool(MiniTool):
    name = "plan_todos"
    description = "Set the TODO list"
    prompt = """\
Use when: at the start of a task, to lay out all planned steps before executing.
Don't use for: updating progress mid-task — use update_todo instead.

Examples:
    plan_todos(["Read existing code", "Add feature", "Test the change"])
"""

    async def _run(self, items: list[str]) -> TodoPlanOutput:
        rendered = todos._todos.plan(items)
        return TodoPlanOutput(count=len(items), rendered=rendered)

    def render_received(self, args: dict) -> str:
        items = args.get("items", [])
        return f"{len(items)} items"

    def render_complete(self, args: dict, output: ToolOutput | None) -> str:
        if not isinstance(output, TodoPlanOutput):
            return self.render_received(args)
        return f"{output.count} todos"


class UpdateTodoTool(MiniTool):
    name = "update_todo"
    description = "Update a TODO item status"
    prompt = """\
Use when: marking a step in_progress before starting it, or done after completing it.
Don't use for: adding new steps — use plan_todos to set the full list upfront.

Examples:
    update_todo("Read existing code", "in_progress")
    update_todo("Read existing code", "done")
"""

    async def _run(self, item: str, status: str) -> TodoUpdateOutput:
        rendered = todos._todos.update(item, status)
        return TodoUpdateOutput(item=item, status=status, rendered=rendered)

    def render_received(self, args: dict) -> str:
        item = args.get("item", "")
        status = args.get("status", "")
        short = (item[:30] + "…") if len(item) > 30 else item
        return f"{short} → {status}"

    def render_complete(self, args: dict, output: ToolOutput | None) -> str:
        return self.render_received(args)


class PlanTasksTool(MiniTool):
    name = "plan_tasks"
    description = "Set a dependency-aware task graph"
    prompt = """\
Use when: complex work with 3+ steps where some steps have prerequisites.
Don't use for: simple checklists with no dependencies — use plan_todos instead.

Examples:
    plan_tasks([
        {"id": "read", "description": "Read existing code"},
        {"id": "impl", "description": "Implement feature", "depends_on": ["read"]},
    ])
"""

    async def _run(self, tasks_list: list[dict]) -> TaskPlanOutput:
        rendered = tasks._tasks.plan(tasks_list)
        return TaskPlanOutput(count=len(tasks_list), rendered=rendered)

    def render_received(self, args: dict) -> str:
        tlist = args.get("tasks_list", [])
        return f"{len(tlist)} tasks"

    def render_complete(self, args: dict, output: ToolOutput | None) -> str:
        if not isinstance(output, TaskPlanOutput):
            return self.render_received(args)
        return f"{output.count} tasks"


class UpdateTaskTool(MiniTool):
    name = "update_task"
    description = "Update a task status"
    prompt = """\
Use when: marking a task in_progress before starting it, or done after completing it.
Don't use for: adding tasks — use plan_tasks to set the full graph upfront.

Note: starting a task whose dependencies aren't done will return an error.

Examples:
    update_task("read", "in_progress")
    update_task("read", "done")
"""

    async def _run(self, task_id: str, status: str) -> TaskUpdateOutput:
        rendered = tasks._tasks.update(task_id, status)
        return TaskUpdateOutput(task_id=task_id, status=status, rendered=rendered)

    def render_received(self, args: dict) -> str:
        return f"{args.get('task_id', '')} → {args.get('status', '')}"

    def render_complete(self, args: dict, output: ToolOutput | None) -> str:
        return self.render_received(args)


class RunSkillTool(MiniTool):
    name = "run_skill"
    description = "Execute a skill"
    prompt = """\
Use when: the user's request matches a skill listed in the system prompt.
Don't use for: tasks not covered by available skills; don't guess skill names.

Examples:
    run_skill("review", "review the current PR")
    run_skill("init", "initialize CLAUDE.md for this project")
"""

    async def _run(self, name: str, request: str, context: str = "") -> RunSkillOutput:
        body = skills._skill_manager.body(name)
        if body is None:
            available = ", ".join(skills._skill_manager.names()) or "(none)"
            return RunSkillOutput(
                skill_name=name,
                result=f"Skill '{name}' not found. Available: {available}",
            )

        # Lazy imports: query_engine imports tools indirectly via tool lists
        from mini_cc.engine.query_engine import get_engine
        from mini_cc.engine.store import _triggering_asst_id

        parent_id = _triggering_asst_id.get()
        if parent_id is None:
            return RunSkillOutput(
                skill_name=name,
                result="Error: run_skill called outside a tool-dispatch context",
            )

        user_content = f"{request}\n\n## Context\n{context}" if context else request
        result = await get_engine().run_sidechain(
            parent_id=parent_id,
            system_prompt=body,
            user_prompt=user_content,
            label=f"skill:{name}",
        )
        return RunSkillOutput(skill_name=name, result=result)

    def render_received(self, args: dict) -> str:
        skill = args.get("name", "")
        req = args.get("request", "")
        short_req = (req[:40] + "…") if len(req) > 40 else req
        return f"{skill}: {short_req}" if short_req else skill

    def render_complete(self, args: dict, output: ToolOutput | None) -> str:
        return f"{args.get('name', '')} · done"


class TaskTool(MiniTool):
    name = "task"
    description = "Delegate a subtask"
    prompt = """\
Use when: the subtask is independent and doesn't need the current conversation history.
Don't use for: simple single-step actions you can do directly.

Examples:
    task("Read src/main.py and summarize the architecture")
    task("Write unit tests for the parse_args function in utils.py")
"""

    async def _run(self, description: str) -> SubTaskOutput:
        # Lazy imports: same circular-dep rationale as run_skill
        from mini_cc.engine.query_engine import get_engine
        from mini_cc.engine.store import _triggering_asst_id

        parent_id = _triggering_asst_id.get()
        if parent_id is None:
            return SubTaskOutput(result="Error: task called outside a tool-dispatch context")

        result = await get_engine().run_sidechain(
            parent_id=parent_id,
            system_prompt=prompts.SUB_SYSTEM_PROMPT,
            user_prompt=description,
            label="task",
        )
        return SubTaskOutput(result=result)

    def render_received(self, args: dict) -> str:
        desc = args.get("description", "")
        return (desc[:57] + "…") if len(desc) > 60 else desc

    def render_complete(self, args: dict, output: ToolOutput | None) -> str:
        return "done"


# ---------------------------------------------------------------------------
# Registration — side-effect on import
# ---------------------------------------------------------------------------

register(ExecuteCommandTool())
register(WriteFileTool())
register(EditFileTool())
register(PlanTodosTool())
register(UpdateTodoTool())
register(PlanTasksTool())
register(UpdateTaskTool())
register(RunSkillTool())
register(TaskTool())
