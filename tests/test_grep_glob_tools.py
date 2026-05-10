"""Tests for grep / glob tools.

Goal coverage (per grep-glob-tools.md plan §D7):
  G1 grep tool         — TG01-TG12 happy/edge/error
  G2 glob tool         — TG19-TG24
  G3 VCS exclude       — TG13 (real rg)
  G4 path relativize   — TG14 (real rg)
  G7 errors+did-you-mean — TG10, TG15, TG16, TG23
  G8 type param        — TG17, TG18

Uses subprocess.run monkeypatching for unit tests (TG01-TG12 etc) so the
suite runs even on systems without ripgrep installed. Tests that genuinely
need rg (TG13/14/17/19-22) are marked with `requires_rg` and skipped on
no-rg systems.
"""
import asyncio
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from mini_cc import config
from mini_cc.tools.base import GlobOutput, GrepOutput, ToolErrorOutput
from mini_cc.tools.glob import GlobTool
from mini_cc.tools.grep import (
    GrepTool,
    _apply_head_limit,
    _common_prefix_len,
    _split_glob_patterns,
    _suggest_path,
)


HAS_RG = shutil.which("rg") is not None
requires_rg = pytest.mark.skipif(not HAS_RG, reason="ripgrep not installed on PATH")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def workdir(tmp_path: Path, monkeypatch):
    """Sandbox CWD = tmp_path."""
    monkeypatch.setattr(config, "CWD", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def grep():
    return GrepTool()


@pytest.fixture
def glob_tool():
    return GlobTool()


def _arun(coro):
    return asyncio.run(coro)


def _mock_rg(monkeypatch, *, stdout="", stderr="", returncode=0):
    """Patch subprocess.run in BOTH grep and glob modules. Captures the
    args list the tool tried to pass to rg.
    """
    captured = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return SimpleNamespace(stdout=stdout, stderr=stderr, returncode=returncode)

    monkeypatch.setattr("mini_cc.tools.grep.subprocess.run", fake_run)
    monkeypatch.setattr("mini_cc.tools.glob.subprocess.run", fake_run)
    return captured


# ---------------------------------------------------------------------------
# Helpers (TH00-prefix tests)
# ---------------------------------------------------------------------------

def test_split_glob_patterns_keeps_braces():
    assert _split_glob_patterns("*.{ts,tsx}") == ["*.{ts,tsx}"]


def test_split_glob_patterns_splits_on_comma_and_space():
    assert _split_glob_patterns("*.py, *.pyx") == ["*.py", "*.pyx"]
    assert _split_glob_patterns("*.py *.pyx") == ["*.py", "*.pyx"]


def test_apply_head_limit_truncates():
    items = list(range(20))
    sliced, applied = _apply_head_limit([str(i) for i in items], limit=5, offset=0)
    assert len(sliced) == 5
    assert applied == 5


def test_apply_head_limit_zero_means_unlimited():
    items = [str(i) for i in range(20)]
    sliced, applied = _apply_head_limit(items, limit=0, offset=0)
    assert len(sliced) == 20
    assert applied is None


def test_apply_head_limit_with_offset():
    items = [str(i) for i in range(20)]
    sliced, applied = _apply_head_limit(items, limit=5, offset=10)
    assert sliced == ["10", "11", "12", "13", "14"]
    assert applied == 5


def test_common_prefix_len():
    assert _common_prefix_len("srrc", "src") == 2
    assert _common_prefix_len("abc", "xyz") == 0
    assert _common_prefix_len("hello", "hello") == 5


# ---------------------------------------------------------------------------
# G1: grep happy path
# ---------------------------------------------------------------------------

def test_TG01_grep_files_with_matches_default(workdir, grep, monkeypatch):
    (workdir / "a.py").write_text("def foo():\n    pass\n", encoding="utf-8")
    (workdir / "b.py").write_text("def bar():\n    pass\n", encoding="utf-8")
    captured = _mock_rg(
        monkeypatch,
        stdout=str(workdir / "a.py") + "\n" + str(workdir / "b.py") + "\n",
        returncode=0,
    )
    out = _arun(grep.execute(pattern="def"))
    assert isinstance(out, GrepOutput)
    assert out.mode == "files_with_matches"
    assert out.num_files == 2
    # files relativized to workdir
    assert "a.py" in out.filenames
    # `-l` flag passed
    assert "-l" in captured["argv"]


def test_TG02_grep_content_mode_with_line_numbers(workdir, grep, monkeypatch):
    captured = _mock_rg(
        monkeypatch,
        stdout=f"{workdir / 'a.py'}:1:def foo():\n{workdir / 'b.py'}:3:def bar():\n",
        returncode=0,
    )
    out = _arun(grep.execute(pattern="def", output_mode="content"))
    assert isinstance(out, GrepOutput)
    assert out.mode == "content"
    assert "def foo()" in out.content
    assert "-n" in captured["argv"]
    assert "-l" not in captured["argv"]


def test_TG03_grep_count_mode(workdir, grep, monkeypatch):
    _mock_rg(
        monkeypatch,
        stdout=f"{workdir / 'a.py'}:5\n{workdir / 'b.py'}:3\n",
        returncode=0,
    )
    out = _arun(grep.execute(pattern="def", output_mode="count"))
    assert isinstance(out, GrepOutput)
    assert out.mode == "count"
    assert out.num_matches == 8
    assert out.num_files == 2


# ---------------------------------------------------------------------------
# G1 edge: head_limit, offset, dash-leading pattern
# ---------------------------------------------------------------------------

def test_TG04_grep_head_limit_truncates(workdir, grep, monkeypatch):
    files = [str(workdir / f"f{i}.py") for i in range(50)]
    for p in files:
        Path(p).write_text("def x(): pass\n")
    _mock_rg(monkeypatch, stdout="\n".join(files) + "\n", returncode=0)
    out = _arun(grep.execute(pattern="def", head_limit=2))
    assert out.applied_limit == 2
    assert len(out.filenames) == 2


def test_TG05_grep_offset_skips(workdir, grep, monkeypatch):
    files = [str(workdir / f"f{i}.py") for i in range(10)]
    for p in files:
        Path(p).write_text("def x(): pass\n")
    _mock_rg(monkeypatch, stdout="\n".join(files) + "\n", returncode=0)
    out = _arun(grep.execute(pattern="def", offset=5, head_limit=3))
    assert out.applied_offset == 5
    assert len(out.filenames) == 3


def test_TG06_grep_head_limit_zero_unlimited(workdir, grep, monkeypatch):
    files = [str(workdir / f"f{i}.py") for i in range(50)]
    for p in files:
        Path(p).write_text("def x(): pass\n")
    _mock_rg(monkeypatch, stdout="\n".join(files) + "\n", returncode=0)
    out = _arun(grep.execute(pattern="def", head_limit=0))
    assert out.applied_limit is None
    assert len(out.filenames) == 50


def test_TG07_grep_dash_leading_pattern_uses_e(workdir, grep, monkeypatch):
    captured = _mock_rg(monkeypatch, stdout="", returncode=1)
    _arun(grep.execute(pattern="-foo"))
    # `-e` must precede the pattern so rg doesn't parse it as a flag.
    argv = captured["argv"]
    e_idx = argv.index("-e")
    assert argv[e_idx + 1] == "-foo"


def test_TG08_grep_case_insensitive_flag(workdir, grep, monkeypatch):
    captured = _mock_rg(monkeypatch, stdout="", returncode=1)
    _arun(grep.execute(pattern="todo", case_insensitive=True))
    assert "-i" in captured["argv"]


def test_TG09_grep_context_flag_in_content_mode(workdir, grep, monkeypatch):
    captured = _mock_rg(monkeypatch, stdout="", returncode=1)
    _arun(grep.execute(pattern="def", output_mode="content", context=2))
    argv = captured["argv"]
    c_idx = argv.index("-C")
    assert argv[c_idx + 1] == "2"


# ---------------------------------------------------------------------------
# G1 error: sandbox, regex, timeout
# ---------------------------------------------------------------------------

def test_TG10_grep_sandbox_escape_rejected(workdir, grep):
    out = _arun(grep.execute(pattern="x", path="../../etc"))
    assert isinstance(out, ToolErrorOutput)
    assert "outside the working directory" in out.message
    # Next-step requirement (G7): error must include actionable recovery
    assert "execute_command" in out.message


def test_TG11_grep_invalid_regex_message(workdir, grep, monkeypatch):
    _mock_rg(
        monkeypatch,
        stderr="regex parse error: unclosed group at position 1",
        returncode=2,
    )
    out = _arun(grep.execute(pattern="("))
    assert isinstance(out, ToolErrorOutput)
    assert "Invalid regex pattern" in out.message
    assert "Rust regex" in out.message
    assert "execute_command" in out.message  # next-step


def test_TG12_grep_timeout_returns_error(workdir, grep, monkeypatch):
    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="rg", timeout=60)
    monkeypatch.setattr("mini_cc.tools.grep.subprocess.run", fake_run)
    out = _arun(grep.execute(pattern="x"))
    assert isinstance(out, ToolErrorOutput)
    assert "timed out after 60s" in out.message
    assert "head_limit" in out.message  # next-step


# ---------------------------------------------------------------------------
# G3: VCS auto-exclude (real rg)
# ---------------------------------------------------------------------------

@requires_rg
def test_TG13_grep_excludes_git_dir(workdir, grep):
    # Create a fake .git directory with a file that matches the pattern,
    # plus a normal file.
    (workdir / ".git").mkdir()
    (workdir / ".git" / "config").write_text("token=secret\n", encoding="utf-8")
    (workdir / "normal.txt").write_text("token=public\n", encoding="utf-8")
    out = _arun(grep.execute(pattern="token"))
    assert isinstance(out, GrepOutput)
    # Only normal.txt should appear; .git/config excluded
    paths = [Path(f).name for f in out.filenames]
    assert "normal.txt" in paths
    assert "config" not in paths


# ---------------------------------------------------------------------------
# G4: relativize paths (real rg)
# ---------------------------------------------------------------------------

@requires_rg
def test_TG14_grep_relativizes_paths(workdir, grep):
    sub = workdir / "src"
    sub.mkdir()
    (sub / "a.py").write_text("def foo():\n    pass\n", encoding="utf-8")
    out = _arun(grep.execute(pattern="def"))
    # Should NOT be absolute; should start with src/ or just the file
    for f in out.filenames:
        assert not os.path.isabs(f)


# ---------------------------------------------------------------------------
# G7: did-you-mean for path typos
# ---------------------------------------------------------------------------

def test_TG15_grep_path_typo_suggests_similar(workdir, grep):
    (workdir / "src").mkdir()
    (workdir / "src" / "a.py").write_text("def foo(): pass", encoding="utf-8")
    out = _arun(grep.execute(pattern="def", path="srrc"))
    assert isinstance(out, ToolErrorOutput)
    assert 'Did you mean "src"' in out.message
    assert "execute_command" in out.message  # next-step


def test_TG16_grep_path_typo_no_similar_omits_hint(workdir, grep):
    out = _arun(grep.execute(pattern="def", path="zzzzz"))
    assert isinstance(out, ToolErrorOutput)
    assert "Did you mean" not in out.message
    # Even without suggestion, must still include actionable next-step
    assert "execute_command" in out.message


# ---------------------------------------------------------------------------
# G8: --type param
# ---------------------------------------------------------------------------

@requires_rg
def test_TG17_grep_type_filter(workdir, grep):
    (workdir / "a.py").write_text("import os\n", encoding="utf-8")
    (workdir / "b.txt").write_text("import os\n", encoding="utf-8")
    out = _arun(grep.execute(pattern="import", type="py"))
    assert isinstance(out, GrepOutput)
    paths = [Path(f).name for f in out.filenames]
    assert "a.py" in paths
    assert "b.txt" not in paths


def test_TG18_grep_unknown_type_actionable_error(workdir, grep, monkeypatch):
    _mock_rg(
        monkeypatch,
        stderr="error: unrecognized file type: nonexistent",
        returncode=2,
    )
    out = _arun(grep.execute(pattern="x", type="nonexistent"))
    assert isinstance(out, ToolErrorOutput)
    assert "Unknown file type 'nonexistent'" in out.message
    assert "rg --type-list" in out.message  # next-step
    assert "glob" in out.message  # alternative


# ---------------------------------------------------------------------------
# G2: glob happy + edge
# ---------------------------------------------------------------------------

@requires_rg
def test_TG19_glob_pattern_filter(workdir, glob_tool):
    (workdir / "a.py").write_text("", encoding="utf-8")
    (workdir / "b.py").write_text("", encoding="utf-8")
    (workdir / "c.txt").write_text("", encoding="utf-8")
    out = _arun(glob_tool.execute(pattern="**/*.py"))
    assert isinstance(out, GlobOutput)
    names = sorted(Path(f).name for f in out.filenames)
    assert names == ["a.py", "b.py"]


def test_TG20_glob_default_path_is_cwd(workdir, glob_tool, monkeypatch):
    captured = _mock_rg(monkeypatch, stdout="", returncode=0)
    _arun(glob_tool.execute(pattern="*.py"))
    # Last positional arg of rg should be CWD when no path passed
    assert captured["argv"][-1] == config.CWD


def test_TG21_glob_truncated_when_over_cap(workdir, glob_tool, monkeypatch):
    files = [str(workdir / f"f{i}.py") for i in range(150)]
    for p in files:
        Path(p).write_text("")
    _mock_rg(monkeypatch, stdout="\n".join(files) + "\n", returncode=0)
    out = _arun(glob_tool.execute(pattern="**/*.py"))
    assert out.truncated is True
    assert len(out.filenames) == 100


def test_TG22_glob_mtime_descending(workdir, glob_tool, monkeypatch):
    older = workdir / "old.py"
    newer = workdir / "new.py"
    older.write_text("")
    newer.write_text("")
    # Force older < newer mtime regardless of write order
    os.utime(older, (1, 1))
    os.utime(newer, (1_000_000, 1_000_000))
    _mock_rg(monkeypatch, stdout=f"{newer}\n{older}\n", returncode=0)
    out = _arun(glob_tool.execute(pattern="*.py"))
    assert out.filenames[0].endswith("new.py")
    assert out.filenames[1].endswith("old.py")


# ---------------------------------------------------------------------------
# G2 errors
# ---------------------------------------------------------------------------

def test_TG23_glob_path_is_file_rejected(workdir, glob_tool):
    (workdir / "foo.py").write_text("", encoding="utf-8")
    out = _arun(glob_tool.execute(pattern="*.py", path="foo.py"))
    assert isinstance(out, ToolErrorOutput)
    assert "is a file, not a directory" in out.message
    # Next-step recovery
    assert "file_read" in out.message or "parent directory" in out.message


def test_TG24_glob_sandbox_escape_rejected(workdir, glob_tool):
    out = _arun(glob_tool.execute(pattern="*", path="../../etc"))
    assert isinstance(out, ToolErrorOutput)
    assert "outside the working directory" in out.message
    assert "execute_command" in out.message  # next-step


# ---------------------------------------------------------------------------
# Helper: _suggest_path
# ---------------------------------------------------------------------------

def test_suggest_path_finds_same_prefix(workdir):
    (workdir / "src").mkdir()
    out = _suggest_path(str(workdir / "srrc"), kind="any")
    assert "src" in out


def test_suggest_path_dir_kind_filters_files(workdir):
    (workdir / "src.py").write_text("")  # file, not dir
    # "src" base, only src.py exists → no dir candidate, suggestion empty
    out = _suggest_path(str(workdir / "src"), kind="dir")
    # src.py is a file, not a directory; with kind="dir" it should not match
    assert "src.py" not in out


def test_suggest_path_no_candidate_returns_empty(workdir):
    (workdir / "totally_different").mkdir()
    out = _suggest_path(str(workdir / "zzzzz"), kind="any")
    assert out == ""
