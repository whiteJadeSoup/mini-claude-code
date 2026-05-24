"""Tests for config.safe_path — sandbox + memdir whitelist.

Three zones, three behaviors:
  - CWD subtree    → resolved (fast path; common case)
  - memdir subtree → resolved (whitelist; enables file_write to write memory)
  - anywhere else  → ValueError

The whitelist exists because the system prompt instructs the LLM to write
memory files into ~/.minicc/.../memory/ using file_write, which goes
through safe_path. Without the whitelist, every memory write would raise.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from mini_cc import config
from mini_cc.memdir import paths as memdir_paths


@pytest.fixture
def stub_home(tmp_path: Path, monkeypatch):
    """Reroute Path.home() to tmp_path so the memdir lives in a tmp tree,
    and clear the @cache so the next get_auto_mem_path() call re-derives
    against the stubbed home. Also stub CWD to a separate tmp subdir so
    "inside CWD" and "inside memdir" are disjoint zones.
    """
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    # Use a subdir to keep CWD distinct from memdir (~/.minicc/...).
    cwd = tmp_path / "project"
    cwd.mkdir()
    monkeypatch.setattr(config, "CWD", str(cwd))
    memdir_paths.get_auto_mem_path.cache_clear()
    yield tmp_path
    memdir_paths.get_auto_mem_path.cache_clear()


class TestSafePathCwdZone:
    def test_relative_path_inside_cwd_resolves(self, stub_home):
        result = config.safe_path("foo.py")
        assert result == os.path.realpath(os.path.join(config.CWD, "foo.py"))

    def test_nested_relative_path_resolves(self, stub_home):
        result = config.safe_path("a/b/c.txt")
        assert result.startswith(os.path.realpath(config.CWD))

    def test_cwd_itself_resolves(self, stub_home):
        # safe_path(".") → CWD; not a traversal.
        result = config.safe_path(".")
        assert os.path.normcase(result) == os.path.normcase(
            os.path.realpath(config.CWD)
        )


class TestSafePathMemdirZone:
    def test_absolute_path_inside_memdir_resolves(self, stub_home):
        memdir = memdir_paths.get_auto_mem_path()
        target = str(memdir / "user_role.md")
        result = config.safe_path(target)
        assert os.path.normcase(result) == os.path.normcase(target)

    def test_nested_path_inside_memdir_resolves(self, stub_home):
        memdir = memdir_paths.get_auto_mem_path()
        target = str(memdir / "logs" / "2026" / "05" / "today.md")
        result = config.safe_path(target)
        assert os.path.normcase(result) == os.path.normcase(target)

    def test_memdir_root_itself_resolves(self, stub_home):
        memdir = memdir_paths.get_auto_mem_path()
        result = config.safe_path(str(memdir))
        assert os.path.normcase(result) == os.path.normcase(str(memdir))


class TestSafePathRejection:
    def test_traversal_out_of_cwd_rejected(self, stub_home):
        with pytest.raises(ValueError, match="outside working directory or memdir"):
            config.safe_path("../escape.txt")

    def test_absolute_outside_both_zones_rejected(self, stub_home, tmp_path):
        # Create a sibling dir of CWD that is NOT inside memdir.
        elsewhere = tmp_path / "not_in_any_zone"
        elsewhere.mkdir()
        with pytest.raises(ValueError, match="outside working directory or memdir"):
            config.safe_path(str(elsewhere / "x.md"))

    def test_path_resembling_memdir_prefix_rejected(self, stub_home):
        # Defends against accidental "starts with memdir name" matches that
        # aren't actual subpaths — e.g. ~/.minicc/projects/<slug>/memory_evil/
        memdir = memdir_paths.get_auto_mem_path()
        sibling = memdir.parent / (memdir.name + "_evil")
        # Don't create it — safe_path doesn't check existence, only path shape.
        with pytest.raises(ValueError, match="outside working directory or memdir"):
            config.safe_path(str(sibling / "foo.md"))


class TestSafePathLazyImport:
    def test_cwd_path_does_not_trigger_memdir_resolution(
        self, stub_home, monkeypatch
    ):
        """Performance + isolation: the fast path through CWD must not call
        get_auto_mem_path. If it did, every file_write inside the project
        would pay the (cached but nonzero) memdir-derivation cost and would
        couple sandbox checks to git rev-parse subprocess behavior.
        """
        calls = {"n": 0}
        real = memdir_paths.get_auto_mem_path

        def counting(*a, **kw):
            calls["n"] += 1
            return real(*a, **kw)

        # The lazy import inside safe_path resolves the name from
        # mini_cc.memdir (the package), which re-exports get_auto_mem_path
        # from mini_cc.memdir.paths. Monkeypatching the package attr is
        # what the import binds to.
        import mini_cc.memdir as memdir_pkg
        monkeypatch.setattr(memdir_pkg, "get_auto_mem_path", counting)

        config.safe_path("foo.py")
        assert calls["n"] == 0
