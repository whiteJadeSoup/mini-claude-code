"""P1-1D: exit criteria for the 可存 (storable) attribute.

These tests verify the strict-write / lenient-read contract by
constructing fixture .md files and asserting:
  1. Valid files appear in scan results, mtime-desc sorted.
  2. Files with malformed frontmatter / unknown types are INCLUDED
     (with degraded fields) rather than dropped — 严进宽出.
  3. MEMORY.md (the index file) is excluded.
  4. Multi-project isolation: different cwds → different memdir keys.
  5. KAIROS daily-log files under ``logs/**`` are also picked up
     (the rglob hook works as documented).

We do not test ``get_auto_mem_path`` against the real ``~/.minicc/``
location — that depends on whether the test runner happens to be inside
a git repo, which is unstable. Instead, scan tests construct temp
memdirs and call ``scan_memory_files(tmpdir)`` directly. ``get_auto_mem_path``
gets its own isolation test that mocks cwd.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from mini_cc.memdir import (
    MemoryType,
    format_memory_manifest,
    get_auto_mem_daily_log_path,
    get_auto_mem_path,
    parse_memory_type,
    scan_memory_files,
    validate_memory_path,
)
from mini_cc.memdir import paths as paths_module


# ---------------------------------------------------------------------------
# types.py
# ---------------------------------------------------------------------------


def test_memory_types_are_a_closed_set_of_four():
    assert sorted(t.value for t in MemoryType) == [
        "feedback", "project", "reference", "user",
    ]


@pytest.mark.parametrize("raw, expected", [
    ("user", MemoryType.USER),
    ("feedback", MemoryType.FEEDBACK),
    ("project", MemoryType.PROJECT),
    ("reference", MemoryType.REFERENCE),
    ("unknown", None),         # not in closed set
    ("USER", None),            # case-sensitive
    ("", None),
    (None, None),              # non-string
    (42, None),
    ([], None),
])
def test_parse_memory_type_tolerant(raw, expected):
    """parse_memory_type must NEVER raise; unknown/non-str → None.

    This is load-bearing for scan's lenient-read contract — if parse
    raised here, an unknown-type file would be silently DROPPED via the
    outer try/except in scan, instead of surfacing with type=None.
    """
    assert parse_memory_type(raw) is expected


# ---------------------------------------------------------------------------
# paths.py: validate_memory_path safety
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw", [
    None,
    "",
    "   ",
    "relative/path",        # not absolute
    "/",                    # too short
    "C:",                   # Windows drive root
    "/a",                   # length < 3 after rstrip (2 chars)
    "/a\x00b",              # null byte
    "~",                    # bare tilde → would expand to $HOME
    "~/",                   # would expand to $HOME
    "~/.",                  # would normalize to $HOME
    "~/..",                 # would normalize to $HOME parent
    "\\\\server\\share",    # UNC
])
def test_validate_memory_path_rejects_dangerous_inputs(raw):
    """Rejecting bare ``~`` is the most safety-critical case. If memdir
    expanded to $HOME, the future MemoryDirSandbox (P3) would allow
    writes to anywhere under $HOME, including ~/.ssh/authorized_keys.
    """
    assert validate_memory_path(raw) is None


def test_validate_memory_path_accepts_well_formed_absolute(tmp_path):
    p = validate_memory_path(str(tmp_path))
    assert p is not None
    assert p.is_absolute()


def test_validate_memory_path_expands_tilde_when_not_trivial():
    expanded = validate_memory_path("~/somesubdir")
    assert expanded is not None
    assert "~" not in str(expanded)
    assert str(expanded).endswith("somesubdir")


# ---------------------------------------------------------------------------
# paths.py: derivation + KAIROS hook
# ---------------------------------------------------------------------------


def test_get_auto_mem_path_is_memoized():
    """Memoization is correctness-critical: changing cwd mid-session
    must NOT relocate the memdir. The memoize on get_auto_mem_path
    pins the first-derivation result for the rest of the session.
    """
    get_auto_mem_path.cache_clear()  # ensure clean state for this test
    first = get_auto_mem_path()
    second = get_auto_mem_path()
    assert first is second  # cache returns the SAME object


def test_daily_log_path_shape():
    """KAIROS hook: shape must be <memdir>/logs/YYYY/MM/YYYY-MM-DD.md.
    Two-level bucketing keeps any single directory bounded to ~31 files
    even after years of writes.
    """
    from datetime import datetime
    p = get_auto_mem_daily_log_path(datetime(2026, 5, 17))
    parts = p.parts
    # last 4 components: 'logs', 'YYYY', 'MM', 'YYYY-MM-DD.md'
    assert parts[-4] == "logs"
    assert parts[-3] == "2026"
    assert parts[-2] == "05"
    assert parts[-1] == "2026-05-17.md"


def test_multi_project_isolation_via_canonical_root(tmp_path, monkeypatch):
    """Two different cwd parents → two different sanitized slugs →
    two different memdirs. This is the "memdir per logical project"
    invariant — the foundation that lets a user have separate memory
    streams for separate codebases.

    We can't easily simulate "actual canonical git root" in a unit test
    without creating real git repos, so we exercise the fallback path
    (non-git cwd) and verify the slug derivation differentiates.
    """
    # Two distinct subdirs of tmp_path simulate two distinct project cwds.
    cwd_a = tmp_path / "project-a"
    cwd_b = tmp_path / "project-b"
    cwd_a.mkdir()
    cwd_b.mkdir()

    # Force the "not a git repo" branch so the test is deterministic.
    monkeypatch.setattr(paths_module, "_git_canonical_root", lambda _: None)

    get_auto_mem_path.cache_clear()
    monkeypatch.chdir(cwd_a)
    memdir_a = get_auto_mem_path()

    get_auto_mem_path.cache_clear()
    monkeypatch.chdir(cwd_b)
    memdir_b = get_auto_mem_path()

    assert memdir_a != memdir_b
    # And neither should be a substring/prefix of the other (independent slugs)
    assert "project-a" in str(memdir_a)
    assert "project-b" in str(memdir_b)
    assert "project-a" not in str(memdir_b)
    assert "project-b" not in str(memdir_a)


# ---------------------------------------------------------------------------
# scan.py: the core P1-1D fixture test
# ---------------------------------------------------------------------------


def _write(p: Path, content: str, mtime_offset_s: float = 0.0) -> None:
    """Helper: write a fixture file and optionally backdate its mtime
    so the sort order test is deterministic regardless of execution
    speed."""
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    if mtime_offset_s != 0:
        now = time.time()
        os.utime(p, (now + mtime_offset_s, now + mtime_offset_s))


def test_scan_5_fixture_files(tmp_path, capsys):
    """The headline P1-1D exit criterion. Fixture per the design table:
      - 3 legal files (user/feedback/project), backdated for deterministic
        mtime ordering
      - 1 with no frontmatter at all
      - 1 with frontmatter but ``type: random_emotion`` (not in closed set)
      - MEMORY.md (the index file — MUST be excluded)
    Plus a KAIROS-mode daily log under logs/YYYY/MM/.

    Asserts:
      1. ALL 5 non-MEMORY.md files appear (lenient-read contract)
      2. The KAIROS log file is also picked up (rglob hook)
      3. MEMORY.md is excluded
      4. Sort order is mtime-desc
      5. Degraded fields land as None for the broken/unknown cases
      6. startup log mentions the scan count
    """
    memdir = tmp_path / "memory"
    memdir.mkdir()

    # 3 legal files, backdated so mtime ordering is predictable
    _write(memdir / "user_role.md",
           "---\nname: user_role\ndescription: senior Java engineer\ntype: user\n---\nbody",
           mtime_offset_s=-30)
    _write(memdir / "feedback_uv.md",
           "---\nname: feedback_uv\ndescription: prefer uv over pip\ntype: feedback\n---\nbody",
           mtime_offset_s=-20)
    _write(memdir / "project_butler.md",
           "---\nname: project_butler\ndescription: personal butler MVP\ntype: project\n---\nbody",
           mtime_offset_s=-10)

    # 1 file with no frontmatter
    _write(memdir / "no_frontmatter.md", "just plain markdown body, no frontmatter\n",
           mtime_offset_s=-5)

    # 1 file with frontmatter but unknown type
    _write(memdir / "unknown_type.md",
           "---\nname: weirdo\ndescription: a strange one\ntype: random_emotion\n---\nbody",
           mtime_offset_s=-3)

    # MEMORY.md — must be excluded
    _write(memdir / "MEMORY.md", "- [user] user_role.md: ...\n", mtime_offset_s=-1)

    # KAIROS-mode daily log (deeper path)
    _write(memdir / "logs" / "2026" / "05" / "2026-05-17.md",
           "---\nname: daily\ndescription: today log\ntype: feedback\n---\nbody",
           mtime_offset_s=0)

    headers = scan_memory_files(memdir)
    captured = capsys.readouterr()

    filenames = [h.filename for h in headers]

    # 1. All 5 non-MEMORY.md files + 1 KAIROS log = 6 results
    assert len(headers) == 6, f"Expected 6 headers, got {len(headers)}: {filenames}"

    # 3. MEMORY.md excluded
    assert "MEMORY.md" not in filenames

    # 2. KAIROS log present (rglob hook works)
    assert any("logs/2026/05/2026-05-17.md" in f for f in filenames), \
        f"KAIROS daily log not picked up; found: {filenames}"

    # 4. mtime-desc sort: most recent (offset=0, KAIROS log) first,
    #    oldest (offset=-30, user_role.md) last
    assert "2026-05-17.md" in headers[0].filename
    assert headers[-1].filename == "user_role.md"

    # 5. Lenient-read field degradation
    by_name = {h.filename: h for h in headers}
    # Legal files: type and description present
    assert by_name["user_role.md"].type is MemoryType.USER
    assert by_name["user_role.md"].description == "senior Java engineer"
    assert by_name["feedback_uv.md"].type is MemoryType.FEEDBACK
    assert by_name["project_butler.md"].type is MemoryType.PROJECT
    # No-frontmatter file: both fields degrade to None, but file is INCLUDED
    assert by_name["no_frontmatter.md"].type is None
    assert by_name["no_frontmatter.md"].description is None
    # Unknown-type file: type degrades to None, description preserved
    assert by_name["unknown_type.md"].type is None
    assert by_name["unknown_type.md"].description == "a strange one"

    # 6. startup log mentions the scan
    assert "[memdir] scan: found" in captured.err
    assert "[memdir] scan: returning 6 headers" in captured.err


def test_scan_handles_missing_memdir(tmp_path):
    """A non-existent memdir returns [] gracefully, never raises.
    This is the boot-time invariant: on first session, memdir may not
    exist yet (P1's get_auto_mem_path will mkdir it, but other callers
    can scan arbitrary dirs)."""
    missing = tmp_path / "definitely-does-not-exist"
    assert scan_memory_files(missing) == []


def test_scan_handles_broken_yaml(tmp_path):
    """File with structurally-broken YAML still appears in results;
    fields degrade to None. This is the 'one corrupt file must not
    poison the rest' contract."""
    memdir = tmp_path / "memory"
    memdir.mkdir()
    _write(memdir / "broken.md",
           "---\nname c\nbroken yaml line with no colon\n---\nbody")
    _write(memdir / "good.md",
           "---\nname: good\ndescription: fine\ntype: feedback\n---\nbody")

    headers = scan_memory_files(memdir)
    by_name = {h.filename: h for h in headers}
    assert len(headers) == 2
    assert by_name["broken.md"].type is None
    assert by_name["broken.md"].description is None
    assert by_name["good.md"].type is MemoryType.FEEDBACK


def test_scan_handles_description_with_colon(tmp_path):
    """Memory descriptions like 'Why: prevents X' must NOT silently
    null out due to YAML's unquoted-colon parse failure. This is the
    quoteProblematicValues retry path."""
    memdir = tmp_path / "memory"
    memdir.mkdir()
    _write(memdir / "colon_desc.md",
           "---\nname: x\ndescription: Why: prevents silent data loss\ntype: feedback\n---\nbody")
    headers = scan_memory_files(memdir)
    assert len(headers) == 1
    assert headers[0].description is not None
    assert "prevents silent data loss" in headers[0].description


def test_scan_caps_at_max_files(tmp_path):
    """Implicit-LRU via mtime-desc + slice(MAX_MEMORY_FILES). Older
    files past the cap drop out of the manifest."""
    from mini_cc.memdir.scan import MAX_MEMORY_FILES

    memdir = tmp_path / "memory"
    memdir.mkdir()
    for i in range(MAX_MEMORY_FILES + 5):
        _write(memdir / f"f{i:03d}.md",
               f"---\nname: f{i}\ndescription: file {i}\ntype: user\n---\nbody",
               mtime_offset_s=-i)  # f000 newest, f204 oldest

    headers = scan_memory_files(memdir)
    assert len(headers) == MAX_MEMORY_FILES
    # Newest survive; oldest dropped
    assert headers[0].filename == "f000.md"
    assert headers[-1].filename == f"f{MAX_MEMORY_FILES - 1:03d}.md"


# ---------------------------------------------------------------------------
# format_memory_manifest — the 4 row shapes
# ---------------------------------------------------------------------------


def test_format_manifest_four_row_shapes(tmp_path):
    """Manifest format per design §2.3 / scan.py docstring.
    All 4 combinations of (type present/absent) × (description present/absent)
    must produce a valid bullet line."""
    memdir = tmp_path / "memory"
    memdir.mkdir()

    _write(memdir / "a_full.md",
           "---\nname: a\ndescription: with desc\ntype: user\n---\n",
           mtime_offset_s=-1)
    _write(memdir / "b_type_only.md",
           "---\nname: b\ntype: feedback\n---\n",
           mtime_offset_s=-2)
    _write(memdir / "c_desc_only.md",
           "---\nname: c\ndescription: only desc\n---\n",
           mtime_offset_s=-3)
    _write(memdir / "d_neither.md", "just a body\n", mtime_offset_s=-4)

    headers = scan_memory_files(memdir)
    text = format_memory_manifest(headers)
    lines = text.split("\n")
    assert len(lines) == 4
    # newest first
    assert "[user]" in lines[0] and ": with desc" in lines[0]
    assert "[feedback]" in lines[1] and "b_type_only.md" in lines[1] and ": " not in lines[1].split(")")[-1]
    assert "[" not in lines[2].split("c_desc_only.md")[0]  # no tag bracket
    assert ": only desc" in lines[2]
    assert "[" not in lines[3].split("d_neither.md")[0]
    assert ": " not in lines[3].split(")")[-1]  # no description either


def test_format_manifest_iso_timestamp_not_relative_age(tmp_path):
    """P4 待回答 question (挂账 task #15): manifest uses ISO timestamp,
    NOT 'N days ago' phrasing. Reason (to be discussed in P4): the
    manifest's consumer is the Sonnet *selector* — it needs a precise
    numeric time signal. Relative-age phrasing belongs in P4's
    read-time freshness hint where the final answering LLM gets nudged
    to question old memories.
    """
    memdir = tmp_path / "memory"
    memdir.mkdir()
    _write(memdir / "a.md",
           "---\nname: a\ndescription: x\ntype: user\n---\n")
    headers = scan_memory_files(memdir)
    text = format_memory_manifest(headers)
    # ISO-8601-ish presence: contains a T and a Z (or +00:00)
    assert "T" in text
    assert ("Z" in text) or ("+00:00" in text)
    # And NOT in relative form
    assert "days ago" not in text
    assert "hours ago" not in text
