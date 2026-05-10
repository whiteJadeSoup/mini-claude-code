"""Shared utilities for tool implementations."""
from contextlib import contextmanager

from mini_cc.state import file_read_state, tasks, todos, usage
from mini_cc.state.file_read_state import FileReadState
from mini_cc.state.tasks import TaskManager
from mini_cc.state.todos import TodoManager
from mini_cc.state.usage import UsageTracker


@contextmanager
def _sub_agent_scope(label: str):
    """Isolate a sub-agent by swapping global singletons; merges usage on exit."""
    saved_todos = todos._todos
    saved_tasks = tasks._tasks
    saved_tracker = usage._tracker
    saved_state = file_read_state._state
    todos._todos = TodoManager()
    tasks._tasks = TaskManager()  # no persist_path → in-memory only, no DAG state leak
    usage._tracker = UsageTracker()
    file_read_state._state = FileReadState()
    try:
        yield
    finally:
        sub_tracker = usage._tracker
        todos._todos = saved_todos
        tasks._tasks = saved_tasks
        usage._tracker = saved_tracker
        file_read_state._state = saved_state
        usage._tracker.merge_sub(label, sub_tracker)
