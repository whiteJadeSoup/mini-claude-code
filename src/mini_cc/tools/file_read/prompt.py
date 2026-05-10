PROMPT = """\
Read a file from the filesystem with line numbers (cat -n format).

REQUIRED arguments:
- path (str): file to read; relative to the project root or absolute (within the sandbox).

OPTIONAL arguments:
- offset (int, default 1): 1-based starting line for chunked continuation.
- limit (int, default 2000): max lines to return; pair with offset for files larger than 2000 lines.

Usage:
- ALWAYS use file_read to read text files. NEVER invoke `cat`, `head`, `tail`, or `type` via execute_command — file_read enforces the read-gate, dedup, and token-budget protections that those raw commands lack.
- Output format per line: `<line_number><tab><line_content>`. The line number prefix is for navigation only — NEVER include any part of the prefix when constructing old_string for file_edit.
- Single lines longer than 2000 characters are truncated; the truncation marker `...[line truncated]` is appended.
- Only UTF-8 text files are supported. For binary files (images, PDFs, archives), use execute_command instead.

Examples (always pass arguments by keyword):
    file_read(path="src/main.py")
    file_read(path="logs/big.log", offset=2001, limit=500)
"""
