PROMPT = """\
A powerful search tool built on ripgrep (rg).

REQUIRED arguments:
- pattern (str): Rust regex (NOT PCRE/Python). Literal braces need escaping (`interface\\{\\}` to find `interface{}`).

OPTIONAL arguments (path is OPTIONAL — defaults to project root):
- path (str, default cwd): directory or file to search.
- glob (str): file-name filter, e.g. "*.js", "**/*.{ts,tsx}".
- type (str): language bundle, e.g. "py", "js", "rust", "go", "java" — run `execute_command("rg --type-list")` for the full set.
- output_mode (str, default "files_with_matches"): one of `files_with_matches` (mtime-sorted paths), `content` (matching lines), `count` (per-file totals).
- show_line_numbers (bool, default true): only meaningful with output_mode="content".
- context (int): -C value, only meaningful with output_mode="content".
- case_insensitive (bool, default false).
- head_limit (int, default 100): cap on results; pass 0 only when you genuinely need every match.
- offset (int, default 0): skip the first N results before applying head_limit.

Usage:
- ALWAYS use grep for content search. NEVER invoke `grep` or `rg` as a Bash command via execute_command — execute_command lacks the head_limit budget, structured output, and VCS-exclude defaults that this tool provides.
- For cross-line patterns or PCRE features, fall back to execute_command("rg -U ...").
- VCS dirs (.git/.svn/.hg/.bzr/.jj/.sl) are excluded automatically; lines wider than 500 cols are skipped (defeats minified/base64 noise).
- Errors include actionable next-steps (e.g. did-you-mean for path typos, install hint for missing types).

Examples (always pass arguments by keyword):
    grep(pattern="TODO", path="src", type="py")
    grep(pattern="function\\s+handle", output_mode="content", context=2)
    grep(pattern="import", output_mode="count", glob="*.ts")
"""
