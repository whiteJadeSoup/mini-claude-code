"""Tests for memory_prompt.build_memory_prompt — the # auto memory block.

The prompt text is mostly verbatim from CC and is content-validated by CC's
own evals; here we test the *structural* contract:

  - memdir is inlined as the absolute path the caller passes
  - all four <type> blocks are present (closed-set discipline reflected in prompt)
  - all six top-level headers are present and in the documented order
  - frontmatter example matches what mini_cc.memdir.scan actually parses
    (top-level `type:`, not `metadata.type`)
  - tool names referenced in "Before recommending" are the mini-cc names,
    not CC's (file_read/file_edit/grep/execute_command vs Read/Edit/Grep/Bash)
  - no team-memory leftovers (would confuse the LLM since no team dir exists)
"""
from pathlib import Path

import pytest

from mini_cc.memory_prompt import build_memory_prompt


@pytest.fixture
def sample_memdir(tmp_path):
    d = tmp_path / "memory"
    d.mkdir()
    return d


class TestMemdirInterpolation:
    def test_absolute_path_inlined(self, sample_memdir):
        out = build_memory_prompt(sample_memdir)
        assert str(sample_memdir) in out

    def test_already_exists_claim_present(self, sample_memdir):
        out = build_memory_prompt(sample_memdir)
        # This claim must be backed by ensureMemoryDirExists-style mkdir at
        # call sites; the prompt text is the LLM-facing half of that contract.
        assert "already exists" in out

    def test_file_write_named_as_the_write_tool(self, sample_memdir):
        out = build_memory_prompt(sample_memdir)
        # CC says "the Write tool"; we must say "file_write" — verify the
        # migration didn't leave CC's tool name behind.
        assert "`file_write`" in out
        assert "the Write tool" not in out


class TestTypeBlocksPresent:
    @pytest.mark.parametrize("name", ["user", "feedback", "project", "reference"])
    def test_all_four_types_have_name_tag(self, sample_memdir, name):
        out = build_memory_prompt(sample_memdir)
        assert f"<name>{name}</name>" in out

    def test_types_wrapper_present(self, sample_memdir):
        out = build_memory_prompt(sample_memdir)
        assert "<types>" in out and "</types>" in out


class TestSectionHeaders:
    REQUIRED_HEADERS = [
        "# auto memory",
        "## Types of memory",
        "## What NOT to save in memory",
        "## How to save memories",
        "## When to access memories",
        "## Before recommending from memory",
        "## Memory and other forms of persistence",
    ]

    def test_all_required_headers_present(self, sample_memdir):
        out = build_memory_prompt(sample_memdir)
        for h in self.REQUIRED_HEADERS:
            assert h in out, f"missing header: {h}"

    def test_headers_appear_in_documented_order(self, sample_memdir):
        out = build_memory_prompt(sample_memdir)
        positions = [out.index(h) for h in self.REQUIRED_HEADERS]
        assert positions == sorted(positions), (
            "headers out of order; order matters because CC evals showed "
            "'Before recommending' must come AFTER 'When to access' to fire correctly "
            "(memoryTypes.ts:240-245 eval rationale)"
        )


class TestFrontmatterShape:
    def test_top_level_type_field(self, sample_memdir):
        """memdir.scan reads meta.get('type') — frontmatter must put type at
        top level, not under metadata.type (CC's shape). If we drift to CC's
        shape, every memory the LLM writes will scan with type=None."""
        out = build_memory_prompt(sample_memdir)
        assert "type: {{user | feedback | project | reference}}" in out
        assert "metadata.type" not in out

    def test_frontmatter_lists_all_four_types(self, sample_memdir):
        out = build_memory_prompt(sample_memdir)
        for t in ("user", "feedback", "project", "reference"):
            assert t in out


class TestVerifyToolsAreMiniCcNames:
    @pytest.mark.parametrize("tool", ["file_read", "glob", "grep", "execute_command"])
    def test_referenced_tool_uses_mini_cc_name(self, sample_memdir, tool):
        out = build_memory_prompt(sample_memdir)
        assert tool in out

    def test_no_cc_tool_name_leakage(self, sample_memdir):
        out = build_memory_prompt(sample_memdir)
        # CC's tool names that we should have replaced. "Read"/"Edit" alone
        # would false-positive against prose like "Read the code"; check the
        # backtick-quoted forms only.
        forbidden_quoted = ["`Read`", "`Edit`", "`Grep`", "`Glob`", "`Bash`"]
        for f in forbidden_quoted:
            assert f not in out, f"leftover CC tool name: {f}"


class TestPersistenceSection:
    def test_mentions_plan_tasks_and_plan_todos(self, sample_memdir):
        out = build_memory_prompt(sample_memdir)
        # mini-cc rewrite of CC's memory-vs-persistence section: only
        # tasks/todos, no Plan Mode (we don't have it).
        assert "plan_tasks" in out
        assert "plan_todos" in out

    def test_no_plan_mode_leakage(self, sample_memdir):
        out = build_memory_prompt(sample_memdir)
        # CC's phrasing was "When to use or update a plan" — we removed that
        # bullet because mini-cc has no Plan Mode. If it comes back, the
        # rewrite drifted.
        assert "When to use or update a plan" not in out


class TestNoCcLeftovers:
    def test_no_team_memory(self, sample_memdir):
        out = build_memory_prompt(sample_memdir)
        # No team scope, no team examples — we ship individual-only.
        assert "<scope>" not in out
        assert "team memory" not in out.lower()
        assert "team feedback memory" not in out
        assert "team project memory" not in out

    def test_no_searching_past_context_section(self, sample_memdir):
        out = build_memory_prompt(sample_memdir)
        # Deliberately omitted (CC's grep-template guidance) — mini-cc's grep
        # tool semantics are slightly different; we let the model figure it
        # out rather than ship potentially wrong examples.
        assert "## Searching past context" not in out

    def test_no_kairos_daily_log(self, sample_memdir):
        out = build_memory_prompt(sample_memdir)
        # KAIROS mode is CC's append-only daily-log paradigm. We don't have
        # it, so prompt must not hint at it.
        assert "logs/YYYY/MM/" not in out
        assert "daily log" not in out.lower()
