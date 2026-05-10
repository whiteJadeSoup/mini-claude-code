PROMPT = """\
Make a targeted string replacement in an existing file.

Usage:
- You MUST call file_read on this file before file_edit (the read-gate enforces this). file_edit will reject if there's no read evidence or the file changed since your last read.
- old_string must match the bytes on disk exactly, including whitespace/indentation. NEVER include the line-number prefix (`<num><tab>`) from file_read output in old_string — that prefix is rendering only, not part of the file.
- old_string must occur exactly once in the file (uniqueness gate). If multiple matches exist, either (1) extend old_string with surrounding context to make it unique, or (2) set replace_all=true to replace all occurrences.
- old_string and new_string must differ; an identical pair is rejected as no-op.
- Don't use this for creating new files or rewriting entire content — use file_write.

Examples:
    file_edit("app.py", "def foo():", "def bar():")
    file_edit("config.json", '"debug": false', '"debug": true')
    # Rename a variable across the file:
    file_edit("util.py", "old_name", "new_name", replace_all=True)
"""
