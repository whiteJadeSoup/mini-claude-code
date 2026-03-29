"""Unit tests for tasks.py — TaskManager, DAG validation, status transitions, persistence."""
import pytest
from pathlib import Path
from tasks import TaskManager, TaskStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_manager(tmp_path: Path | None = None) -> TaskManager:
    """Create a TaskManager backed by a temp file (or in-memory if None)."""
    path = tmp_path / ".tasks.json" if tmp_path else None
    return TaskManager(persist_path=path)


def plan(mgr: TaskManager, *specs) -> str:
    """Shortcut: plan_tasks from (id, desc, deps?) tuples."""
    tasks = []
    for spec in specs:
        t: dict = {"id": spec[0], "description": spec[1]}
        if len(spec) > 2:
            t["depends_on"] = spec[2]
        tasks.append(t)
    return mgr.plan(tasks)


# ---------------------------------------------------------------------------
# DAG validation (8 cases)
# ---------------------------------------------------------------------------

class TestDAGValidation:
    def test_empty_plan_succeeds(self):
        mgr = make_manager()
        result = mgr.plan([])
        assert "Tasks: (empty)" in result

    def test_duplicate_ids_rejected(self):
        mgr = make_manager()
        result = plan(mgr, ("a", "step A"), ("a", "step A again"))
        assert "duplicate" in result.lower()

    def test_unknown_dependency_rejected(self):
        mgr = make_manager()
        result = plan(mgr, ("a", "step A", ["nonexistent"]))
        assert "unknown id" in result.lower()

    def test_self_loop_rejected(self):
        mgr = make_manager()
        result = plan(mgr, ("a", "step A", ["a"]))
        assert "cycle" in result.lower()

    def test_two_node_cycle_rejected(self):
        mgr = make_manager()
        result = plan(mgr, ("a", "step A", ["b"]), ("b", "step B", ["a"]))
        assert "cycle" in result.lower()

    def test_three_node_cycle_rejected(self):
        mgr = make_manager()
        result = plan(mgr,
                      ("a", "step A", ["c"]),
                      ("b", "step B", ["a"]),
                      ("c", "step C", ["b"]))
        assert "cycle" in result.lower()

    def test_valid_chain_accepted(self):
        mgr = make_manager()
        result = plan(mgr,
                      ("a", "step A"),
                      ("b", "step B", ["a"]),
                      ("c", "step C", ["b"]))
        assert "Error" not in result
        assert "[a]" in result

    def test_diamond_dag_accepted(self):
        mgr = make_manager()
        result = plan(mgr,
                      ("root", "start"),
                      ("left", "left branch", ["root"]),
                      ("right", "right branch", ["root"]),
                      ("end", "finish", ["left", "right"]))
        assert "Error" not in result
        assert "[root]" in result


# ---------------------------------------------------------------------------
# Status transitions (7 cases)
# ---------------------------------------------------------------------------

class TestStatusTransitions:
    def test_update_unknown_id(self):
        mgr = make_manager()
        result = mgr.update("ghost", "done")
        assert "not found" in result.lower()

    def test_update_invalid_status(self):
        mgr = make_manager()
        plan(mgr, ("a", "step A"))
        result = mgr.update("a", "flying")
        assert "Error" in result

    def test_no_deps_task_is_ready(self):
        mgr = make_manager()
        result = plan(mgr, ("a", "step A"))
        assert "▶" in result  # ready icon

    def test_start_task_with_unmet_deps_blocked(self):
        mgr = make_manager()
        plan(mgr, ("a", "step A"), ("b", "step B", ["a"]))
        result = mgr.update("b", "in_progress")
        assert "blocked" in result.lower() or "dependencies not done" in result.lower()

    def test_complete_dep_unblocks_dependent(self):
        mgr = make_manager()
        plan(mgr, ("a", "step A"), ("b", "step B", ["a"]))
        mgr.update("a", "in_progress")
        mgr.update("a", "done")
        result = mgr.render()
        assert "▶" in result  # b should now be ready

    def test_in_progress_shows_correct_icon(self):
        mgr = make_manager()
        plan(mgr, ("a", "step A"))
        result = mgr.update("a", "in_progress")
        assert "◉" in result

    def test_done_shows_correct_icon(self):
        mgr = make_manager()
        plan(mgr, ("a", "step A"))
        mgr.update("a", "in_progress")
        result = mgr.update("a", "done")
        assert "✓" in result


# ---------------------------------------------------------------------------
# Persistence (5 cases)
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_state_survives_reload(self, tmp_path):
        mgr = make_manager(tmp_path)
        plan(mgr, ("a", "step A"), ("b", "step B", ["a"]))
        mgr.update("a", "in_progress")
        mgr.update("a", "done")

        mgr2 = make_manager(tmp_path)
        result = mgr2.render()
        assert "✓" in result  # a is done
        assert "▶" in result  # b is now ready

    def test_no_persist_path_does_not_crash(self):
        mgr = make_manager()  # no persist path
        plan(mgr, ("a", "step A"))
        mgr.update("a", "done")  # should not raise

    def test_file_missing_starts_fresh(self, tmp_path):
        path = tmp_path / ".tasks.json"
        # File does not exist — should load as empty
        mgr = TaskManager(persist_path=path)
        assert not mgr.has_incomplete()

    def test_corrupt_file_starts_fresh(self, tmp_path):
        path = tmp_path / ".tasks.json"
        path.write_text("not valid json", encoding="utf-8")
        mgr = TaskManager(persist_path=path)
        assert not mgr.has_incomplete()

    def test_clear_removes_file_content(self, tmp_path):
        mgr = make_manager(tmp_path)
        plan(mgr, ("a", "step A"))
        mgr.clear()

        mgr2 = make_manager(tmp_path)
        assert not mgr2.has_incomplete()


# ---------------------------------------------------------------------------
# Compact state_summary (2 cases)
# ---------------------------------------------------------------------------

class TestCompactIntegration:
    def test_state_summary_none_when_empty(self):
        mgr = make_manager()
        assert mgr.state_summary() is None

    def test_state_summary_returned_when_tasks_exist(self):
        mgr = make_manager()
        plan(mgr, ("a", "step A"))
        summary = mgr.state_summary()
        assert summary is not None
        assert "[a]" in summary
        assert "Task plan restored" in summary
