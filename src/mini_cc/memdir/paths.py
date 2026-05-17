"""Memory directory path derivation + safety validation.

Derives the per-project memory directory at:
    ``~/.minicc/projects/<sanitized-canonical-root>/memory/``

Where ``<canonical-root>`` is the **canonical git root** (so all worktrees
of the same repo share one memdir), falling back to ``os.getcwd()`` when
not in a git repo.

OPTIMIZATION NOTE (`_git_canonical_root`)
-----------------------------------------
We use ``subprocess`` to invoke ``git rev-parse --git-common-dir``. The
spawn cost is ~30-80ms on Linux, 80-200ms on Windows. This is acceptable
because the entire derivation chain is memoized via ``functools.cache``
on ``get_auto_mem_path``: the subprocess fires **at most once per
session**. If mini-cc ever becomes a TUI hot-path (memdir derivation
called per UI render), switch to CC's pure-FS approach: read ``.git``
file → parse ``gitdir:`` prefix → read ``commondir`` → resolve back to
the main repo's working dir, with worktree-structure validation against
attacker-controlled repos. See
``claude-code/src/utils/git.ts:123-183`` (``resolveCanonicalRoot``).

KAIROS hook (`get_auto_mem_daily_log_path`)
-------------------------------------------
Returns ``<memdir>/logs/YYYY/MM/YYYY-MM-DD.md``. Implemented but **not
called** by P1-P3. Reserved for a future ``MemoryMode.assistant`` mode
where long-lived sessions append to dated logs instead of editing
``MEMORY.md`` directly (CC's ``feature('KAIROS')`` design). Scan picks
up these files automatically because it uses ``rglob`` (see scan.py),
so no scan changes needed when KAIROS is enabled later.

The ``logs/YYYY/MM/`` two-level bucketing keeps any single directory
bounded to ~31 files even after years of daily logs — readdir stays
fast.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from datetime import datetime
from functools import cache
from pathlib import Path


# CC parity: matches ``AUTO_MEM_DIRNAME`` and ``AUTO_MEM_ENTRYPOINT_NAME``
# in ``memdir/paths.ts``. Renaming either would break compatibility with
# memory directories created by other tools sharing this layout.
_AUTO_MEM_DIRNAME = "memory"
_AUTO_MEM_ENTRYPOINT_NAME = "MEMORY.md"

# Where the per-project memdir lives. We keep this under ``~/.minicc/``
# (not ``~/.claude/``) so mini-cc and Claude Code don't fight over the
# same files; an integration layer can symlink them later if desired.
_MEMORY_BASE = Path.home() / ".minicc" / "projects"


def _git_canonical_root(start: Path) -> Path | None:
    """Resolve the canonical git root from ``start``.

    Returns the main repository's working directory (so all worktrees of
    one repo map to the same identity), or ``None`` if ``start`` is not
    inside a git repo, git is unavailable, or the subprocess times out.

    Algorithm:
      ``git rev-parse --git-common-dir`` returns the *common* .git
      directory (shared by all worktrees). ``.parent`` is the main
      repo's working dir — exactly the canonical root we want.

    Failure modes are all silenced to ``None`` because the caller will
    fall back to cwd — a degraded but functional memdir is far better
    than a startup error.
    """
    try:
        result = subprocess.run(
            ["git", "-C", str(start), "rev-parse", "--git-common-dir"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=2.0,
        )
    except FileNotFoundError:
        print("[memdir] git not in PATH; falling back to cwd",
              file=sys.stderr, flush=True)
        return None
    except subprocess.TimeoutExpired:
        print(f"[memdir] git rev-parse timed out on {start}; falling back to cwd",
              file=sys.stderr, flush=True)
        return None

    if result.returncode != 0:
        # Not a git repo. This is the common case for non-project cwds —
        # not an error, no log needed.
        return None

    raw = result.stdout.strip()
    if not raw:
        return None

    common_git_dir = Path(raw)
    if not common_git_dir.is_absolute():
        # ``--git-common-dir`` returns a path relative to ``start`` when
        # it can. Resolve relative to ``start``, not cwd (which may
        # differ if the caller passes a non-cwd ``start``).
        common_git_dir = (start / common_git_dir).resolve()

    return common_git_dir.parent


def _sanitize_for_path(raw: str) -> str:
    """Turn a filesystem path into a single-segment slug usable as a
    directory name. Same sanitization rule as
    ``consumers/persistence.py:_cwd_slug`` (replaces ``:``, ``\\``, ``/``,
    whitespace with ``-``, trims edges)."""
    return re.sub(r"[:\\/\s]", "-", raw).strip("-")


def validate_memory_path(raw: str | None) -> Path | None:
    """Validate a user-supplied memory directory path.

    Rejects (returns ``None``) for any of:
      - ``None`` / empty / whitespace-only
      - relative path (would be interpreted relative to cwd → unstable)
      - length < 3 (catches ``/``, ``C:``, etc.)
      - Windows drive root pattern like ``C:`` / ``D:`` (matches before
        normalization could append a separator)
      - UNC paths (``\\\\server\\share``) — opaque trust boundary
      - contains a null byte (truncates in syscalls, NEVER trust)
      - bare ``~`` / ``~/`` / ``~/.`` / ``~/..`` (would expand to $HOME
        or its parent → catastrophic if used as a write-allowlist root,
        e.g. would let memory writes hit ``~/.ssh/authorized_keys``)

    Otherwise returns the normalized absolute ``Path`` (with ``~``
    expanded).

    P1 callers: none. This function exists for the future case where
    ``settings.json`` adds a ``memory_dir`` field. We pre-build the
    validator so the security posture is baked in — the moment someone
    adds the setting, this validator is the gate they must go through.
    """
    if raw is None:
        return None
    candidate = raw.strip()
    if not candidate:
        return None

    # Tilde expansion: reject trivial remainders that would resolve to
    # $HOME or its parent. ``~`` alone, ``~/``, ``~/.``, ``~/..`` are all
    # banned — they would make the entire home directory a memory write
    # zone (RCE-level risk via authorized_keys / id_rsa overwrite).
    if candidate.startswith("~/") or candidate.startswith("~\\") or candidate == "~":
        rest = candidate[1:].lstrip("/\\")
        if not rest or os.path.normpath(rest) in (".", ".."):
            return None
        candidate = os.path.expanduser(candidate)

    if "\x00" in candidate:
        return None
    if candidate.startswith("\\\\") or candidate.startswith("//"):
        return None  # UNC
    if not os.path.isabs(candidate):
        return None
    # Strip any trailing separators before length check
    normalized = candidate.rstrip("/\\")
    if len(normalized) < 3:
        return None
    if re.match(r"^[A-Za-z]:$", normalized):
        return None  # Windows drive root after strip ("C:")

    return Path(normalized)


@cache
def get_auto_mem_path() -> Path:
    """Return (and create on first call) the memdir for this project.

    Resolution order:
      1. ``_git_canonical_root(cwd)`` if in a git repo
      2. Otherwise fall back to cwd

    The result is sanitized into a single-segment slug and joined under
    ``~/.minicc/projects/<slug>/memory/``. The directory (and its
    parents) is created if missing.

    Memoized for the lifetime of the process. Cache invalidates when
    cwd changes are irrelevant — once mini-cc has decided where memory
    lives for this session, switching cwd should NOT relocate memory
    (that would mean "my memory disappears when I cd around", a UX bug).
    """
    cwd = Path(os.getcwd())
    canonical = _git_canonical_root(cwd)
    if canonical is not None:
        key_path = canonical
        source = "git"
    else:
        key_path = cwd
        source = "cwd"

    slug = _sanitize_for_path(str(key_path))
    memdir = _MEMORY_BASE / slug / _AUTO_MEM_DIRNAME

    # mkdir -p; idempotent. Done at first derivation rather than first
    # write so downstream readers (scan) never see a missing dir on
    # initial session.
    memdir.mkdir(parents=True, exist_ok=True)
    print(f"[memdir] resolved memdir={memdir} (source={source})",
          file=sys.stderr, flush=True)
    return memdir


def get_auto_mem_entrypoint() -> Path:
    """Path to the index file (``MEMORY.md``) inside the memdir. P1 does
    not read this file; P2 will inject its content as the
    ``messages[0]`` user-context payload."""
    return get_auto_mem_path() / _AUTO_MEM_ENTRYPOINT_NAME


def get_auto_mem_daily_log_path(date: datetime | None = None) -> Path:
    """KAIROS hook: path to the daily log file for ``date`` (default today).

    Shape: ``<memdir>/logs/YYYY/MM/YYYY-MM-DD.md``

    NOT called anywhere in P1-P4 — reserved for a future ``assistant``
    mode where long-lived sessions append to dated logs. See module
    docstring for the full design intent.

    The two-level ``YYYY/MM/`` bucketing keeps directory file counts
    bounded — even after years of daily writes, no single directory
    exceeds ~31 entries.
    """
    if date is None:
        date = datetime.now()
    return (
        get_auto_mem_path()
        / "logs"
        / f"{date.year:04d}"
        / f"{date.month:02d}"
        / f"{date.year:04d}-{date.month:02d}-{date.day:02d}.md"
    )
