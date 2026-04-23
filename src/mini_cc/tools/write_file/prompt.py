PROMPT = """\
Use when: creating a new file, or replacing an entire file's content.
Don't use for: small targeted edits to existing files — use edit_file instead.

Examples:
    write_file("hello.py", "print('hello')")
    write_file("config/settings.json", '{"debug": true}')
"""
