PROMPT = """\
Create a new file or fully overwrite an existing one.

Usage:
- For NEW files: just provide path and content. Parent directories are created automatically.
- For EXISTING files: you MUST call file_read first (the read-gate enforces this). Overwriting an unread file risks silently losing changes you can't see.
- Don't use this for small targeted edits — use file_edit instead (cheaper, less risky).
- Do not include line numbers or any prefix in `content` — write what you want the file to literally contain.

Examples:
    file_write("hello.py", "print('hello')")
    file_write("config/settings.json", '{"debug": true}')
"""
