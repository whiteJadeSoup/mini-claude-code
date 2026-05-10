PROMPT = """\
Make a targeted string replacement in an existing file.

REQUIRED arguments:
- path (str): file to edit; must exist and have been read by file_read.
- old_string (str): exact bytes to replace; must occur exactly once in the file unless replace_all=true. NEVER include the line-number prefix from file_read output.
- new_string (str): bytes to substitute in.

OPTIONAL arguments:
- replace_all (bool, default false): replace every occurrence instead of requiring uniqueness.

Usage:
- ALWAYS use file_edit for string replacements. NEVER invoke `sed -i`, `awk -i inplace`, or shell-redirected rewrites via execute_command — file_edit enforces the read-gate, uniqueness check, and CRLF-aware staleness detection that those raw commands lack.
- You MUST call file_read on this file before file_edit (the read-gate enforces this). file_edit will reject if there's no read evidence or the file changed since your last read.
- old_string must match the bytes on disk exactly, including whitespace/indentation.
- old_string must occur exactly once in the file (uniqueness gate). If multiple matches exist, either (1) extend old_string with surrounding context to make it unique, or (2) set replace_all=true to replace all occurrences.
- old_string and new_string must differ; an identical pair is rejected as no-op.
- Don't use this for creating new files or rewriting entire content — use file_write.

Examples (always pass arguments by keyword):
    file_edit(path="app.py", old_string="def foo():", new_string="def bar():")
    file_edit(path="config.json", old_string='"debug": false', new_string='"debug": true')
    # Rename a variable across the file:
    file_edit(path="util.py", old_string="old_name", new_string="new_name", replace_all=True)
"""
