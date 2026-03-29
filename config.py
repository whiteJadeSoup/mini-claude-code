import os, platform, shutil

CWD = os.getcwd()
PLATFORM = platform.system()        # "Windows" / "Linux" / "Darwin"
def _find_bash() -> str | None:
    """Find a working bash. On Windows, prefer Git Bash over WSL."""
    if PLATFORM != "Windows":
        return shutil.which("bash")
    # Derive Git Bash from git.exe location: git is at <git_root>/cmd/git.exe
    # or <git_root>/mingw64/bin/git.exe → walk up to find <git_root>/usr/bin/bash.exe
    git_path = shutil.which("git")
    if git_path:
        d = os.path.dirname(os.path.realpath(git_path))
        for _ in range(4):
            bash = os.path.join(d, "usr", "bin", "bash.exe")
            if os.path.isfile(bash):
                return bash
            d = os.path.dirname(d)
    return None

BASH_PATH = _find_bash()


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
