PROMPT = """\
A powerful search tool built on ripgrep (rg).

Usage:
- ALWAYS use grep for content search. NEVER invoke `grep` or `rg` as a Bash command via execute_command — execute_command lacks the head_limit budget, structured output, and VCS-exclude defaults that this tool provides.
- Supports full Rust regex syntax (e.g. `log.*Error`, `function\\s+\\w+`). NOT PCRE/Python — for cross-line patterns or PCRE features, fall back to execute_command("rg -U ...").
- Pattern syntax: literal braces need escaping (`interface\\{\\}` to find `interface{}` in Go code).
- Filter files with `glob` (e.g. "*.js", "**/*.{ts,tsx}") or `type` (e.g. "py", "js", "rust", "go", "java", etc — run `execute_command("rg --type-list")` for the full set).
- Output modes:
    - "files_with_matches" (default): list paths sorted by mtime descending
    - "content": matching lines (supports `context` for -C, `show_line_numbers` for -n)
    - "count": match counts per file
- `head_limit` defaults to 100; pass 0 for unlimited (use sparingly — can flood context).
- VCS dirs (.git/.svn/.hg/.bzr/.jj/.sl) are excluded automatically; lines wider than 500 cols are skipped (defeats minified/base64 noise).
- Errors include actionable next-steps (e.g. did-you-mean for path typos, install hint for missing types).

Examples:
    grep("TODO", path="src", type="py")
    grep("function\\s+handle", output_mode="content", context=2)
    grep("import", output_mode="count", glob="*.ts")
"""
