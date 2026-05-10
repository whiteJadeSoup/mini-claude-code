PROMPT = """\
Find files by name pattern (powered by ripgrep).

REQUIRED arguments:
- pattern (str): shell glob — `**/*.py`, `src/**/*.{ts,tsx}`, `tests/**/test_*.py`, etc.

OPTIONAL arguments (path is OPTIONAL — defaults to project root):
- path (str, default cwd): directory to enumerate; must be a directory (not a file).

Usage:
- ALWAYS use glob for filename search. NEVER invoke `find`, `ls -R`, `git ls-files`, or `rg --files` via execute_command — glob handles VCS exclusion, mtime sorting, and result capping that those raw commands lack.
- Results are sorted by mtime descending (most recently modified first), capped at 100 files. When capped, `truncated=true` is set — narrow the pattern or path.
- VCS dirs (.git/.svn/.hg/.bzr/.jj/.sl) are excluded automatically.
- For content matching (not filename), use grep with a regex pattern.
- For a single file you already know, use file_read with that path directly.

Examples (always pass arguments by keyword):
    glob(pattern="**/*.py")
    glob(pattern="src/**/*.{ts,tsx}")
    glob(pattern="test_*.py", path="tests")
"""
