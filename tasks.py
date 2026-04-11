"""Task dependency graph for the agent.

Complements todos.py (simple flat checklists) with DAG-aware task tracking:
explicit depends_on relationships, ready/blocked visibility, and persistence
across compact and process restart.

Access the singleton as `tasks._tasks` (module attribute), never via
`from tasks import _tasks` — it is reassigned for sub-agent isolation.
"""
from __future__ import annotations

import json
import os
import tempfile
from collections import defaultdict, deque
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

import config


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"


class Task(BaseModel):
    id: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    depends_on: list[str] = Field(default_factory=list)


class TaskGraph(BaseModel):
    tasks: list[Task] = Field(default_factory=list)


_ICONS = {
    TaskStatus.DONE: "✓",
    TaskStatus.IN_PROGRESS: "◉",
}
_READY_ICON = "▶"
_BLOCKED_ICON = "⊘"


class TaskManager:
    def __init__(self, persist_path: Optional[Path] = None):
        self._graph = TaskGraph()
        self._persist_path = persist_path
        if persist_path and persist_path.exists():
            self._load()

    # --- Public API ---

    def plan(self, raw_tasks: list[dict]) -> str:
        """Replace current task graph with a new DAG. Validates before committing."""
        tasks = []
        seen_ids: set[str] = set()
        for raw in raw_tasks:
            t_id = str(raw.get("id", "")).strip()
            if not t_id:
                return "Error: every task must have a non-empty 'id'"
            if t_id in seen_ids:
                return f"Error: duplicate task id '{t_id}'"
            seen_ids.add(t_id)
            tasks.append(Task(
                id=t_id,
                description=str(raw.get("description", "")),
                depends_on=[str(d) for d in raw.get("depends_on", [])],
            ))

        # Validate dependency references
        for t in tasks:
            for dep in t.depends_on:
                if dep not in seen_ids:
                    return f"Error: task '{t.id}' depends_on unknown id '{dep}'"

        err = self._validate_dag(tasks)
        if err:
            return err

        self._graph = TaskGraph(tasks=tasks)
        self._persist()
        return self._render()

    def update(self, task_id: str, status: str) -> str:
        """Update a task's status. Blocks in_progress if dependencies aren't done."""
        try:
            new_status = TaskStatus(status)
        except ValueError:
            valid = [s.value for s in TaskStatus]
            return f"Error: status must be one of {valid}"

        task = self._find(task_id)
        if task is None:
            return f"Error: task '{task_id}' not found. Call plan_tasks first."

        if new_status == TaskStatus.IN_PROGRESS:
            blocked_by = [dep for dep in task.depends_on
                          if self._find(dep).status != TaskStatus.DONE]  # type: ignore[union-attr]
            if blocked_by:
                return (f"Error: cannot start '{task_id}' — "
                        f"dependencies not done: {blocked_by}")

        task.status = new_status
        self._persist()
        return self._render()

    def render(self) -> str:
        return self._render()

    def has_incomplete(self) -> bool:
        return any(t.status != TaskStatus.DONE for t in self._graph.tasks)

    def state_summary(self) -> Optional[str]:
        """Returns a compact task state string for compact-recovery injection, or None."""
        if not self._graph.tasks:
            return None
        return f"[Task plan restored after compact]\n\n{self._render()}"

    def clear(self) -> str:
        self._graph = TaskGraph()
        self._persist()
        return "Tasks cleared."

    # --- Internals ---

    def _find(self, task_id: str) -> Optional[Task]:
        return next((t for t in self._graph.tasks if t.id == task_id), None)

    def _is_ready(self, task: Task) -> bool:
        """A PENDING task is ready when all its dependencies are DONE."""
        return (task.status == TaskStatus.PENDING and
                all(self._find(dep).status == TaskStatus.DONE  # type: ignore[union-attr]
                    for dep in task.depends_on))

    def _validate_dag(self, tasks: list[Task]) -> Optional[str]:
        """Kahn's algorithm — returns error string if a cycle is found, else None."""
        id_set = {t.id for t in tasks}
        in_degree: dict[str, int] = defaultdict(int)
        for t in tasks:
            if t.id not in in_degree:
                in_degree[t.id] = 0
            for dep in t.depends_on:
                in_degree[t.id] += 1

        queue: deque[str] = deque(tid for tid in id_set if in_degree[tid] == 0)
        # Build reverse adjacency: dep → dependents
        dependents: dict[str, list[str]] = defaultdict(list)
        for t in tasks:
            for dep in t.depends_on:
                dependents[dep].append(t.id)

        visited = 0
        while queue:
            node = queue.popleft()
            visited += 1
            for dependent in dependents[node]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        if visited != len(tasks):
            return "Error: depends_on contains a cycle"
        return None

    def _render(self) -> str:
        if not self._graph.tasks:
            return "Tasks: (empty)"
        lines = ["Tasks:"]
        for t in self._graph.tasks:
            if t.status in _ICONS:
                icon = _ICONS[t.status]
            elif self._is_ready(t):
                icon = _READY_ICON
            else:
                # blocked: find which deps are not done
                icon = _BLOCKED_ICON

            line = f"  {icon} [{t.id}] {t.description}"
            if t.status == TaskStatus.PENDING and not self._is_ready(t):
                not_done = [dep for dep in t.depends_on
                            if self._find(dep).status != TaskStatus.DONE]  # type: ignore[union-attr]
                line += f"  (blocked by: {', '.join(not_done)})"
            lines.append(line)
        return "\n".join(lines)

    def _persist(self) -> None:
        if self._persist_path is None:
            return
        data = self._graph.model_dump()
        # Atomic write: write to temp then rename to avoid partial reads
        dir_ = self._persist_path.parent
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=dir_, delete=False, suffix=".tmp"
        ) as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            tmp_path = f.name
        os.replace(tmp_path, self._persist_path)

    def _load(self) -> None:
        try:
            with open(self._persist_path, encoding="utf-8") as f:  # type: ignore[arg-type]
                data = json.load(f)
            self._graph = TaskGraph.model_validate(data)
        except Exception:
            # Corrupt or unreadable file — start fresh rather than crashing
            self._graph = TaskGraph()


_tasks = TaskManager(persist_path=Path(config.CWD) / ".tasks.json")
