import os, platform, shutil

CWD = os.getcwd()
PLATFORM = platform.system()        # "Windows" / "Linux" / "Darwin"
BASH_PATH = shutil.which("bash")    # Git Bash on Windows, /bin/bash on Unix, or None


def safe_path(path: str) -> str:
    """Resolve path relative to CWD and reject escapes.

    Prevents the LLM from reading/writing outside the project via
    path traversal (e.g. ../../etc/passwd).
    """
    resolved = os.path.realpath(os.path.join(CWD, path))
    # normcase for Windows case-insensitive comparison (no-op on Unix)
    resolved_norm = os.path.normcase(resolved)
    cwd_norm = os.path.normcase(CWD)
    if not resolved_norm.startswith(cwd_norm + os.sep) and resolved_norm != cwd_norm:
        raise ValueError(f"Path {path} is outside working directory")
    return resolved
