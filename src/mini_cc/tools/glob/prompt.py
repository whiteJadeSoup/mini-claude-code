PROMPT = """\
Find files by name pattern (powered by ripgrep).

Usage:
- ALWAYS use glob for filename search. NEVER invoke `find`, `ls -R`, `git ls-files`, or `rg --files` via execute_command — glob handles VCS exclusion, mtime sorting, and result capping that those raw commands lack.
- Pattern uses shell glob syntax (e.g. `**/*.py`, `src/**/*.{ts,tsx}`, `tests/**/test_*.py`).
- Optional `path` narrows the search to a directory (must be a directory; for a single file, use file_read).
- Results are sorted by mtime descending (most recently modified first), capped at 100 files. When capped, `truncated=true` is set — narrow the pattern or path.
- VCS dirs (.git/.svn/.hg/.bzr/.jj/.sl) are excluded automatically.
- For content matching (not filename), use grep with a regex pattern.

Examples:
    glob("**/*.py")
    glob("src/**/*.{ts,tsx}")
    glob("test_*.py", path="tests")
"""
