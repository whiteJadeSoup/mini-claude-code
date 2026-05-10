PROMPT = """\
Read a file from the filesystem with line numbers (cat -n format).

Usage:
- The path can be relative to the working directory or absolute (within the project sandbox).
- By default, reads up to 2000 lines from the start. For larger files, use offset (1-based start line) and limit (max lines) to read in chunks.
- Output format per line: `<line_number><tab><line_content>`. The line number prefix is for navigation only — NEVER include any part of the prefix when constructing old_string for file_edit.
- Single lines longer than 2000 characters are truncated; the truncation marker `...[line truncated]` is appended.
- Only UTF-8 text files are supported. For binary files (images, PDFs, archives), use execute_command instead.
- For directory listings or content search, use execute_command with `ls`, `rg`, etc.

Examples:
    file_read("src/main.py")
    file_read("logs/big.log", offset=2001, limit=500)
"""
