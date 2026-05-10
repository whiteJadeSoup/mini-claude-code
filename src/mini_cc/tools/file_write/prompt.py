PROMPT = """\
Create a new file or fully overwrite an existing one.

REQUIRED arguments:
- path (str): destination path; relative to the project root or absolute (within the sandbox).
- content (str): exact bytes to write — no line-number prefixes, no diff markers, no extra framing.

Usage:
- ALWAYS use file_write for create/overwrite. NEVER invoke `echo > file`, `cat <<EOF`, `tee`, or `Set-Content` via execute_command — file_write enforces the read-gate (preventing silent overwrite of unseen content) that those raw commands skip.
- For NEW files: just provide path and content. Parent directories are created automatically.
- For EXISTING files: you MUST call file_read first (the read-gate enforces this). Overwriting an unread file risks silently losing changes you can't see.
- Don't use this for small targeted edits — use file_edit instead (cheaper, less risky).

Examples (always pass arguments by keyword):
    file_write(path="hello.py", content="print('hello')")
    file_write(path="config/settings.json", content='{"debug": true}')
"""
