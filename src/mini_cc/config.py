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

# ripgrep gate for grep/glob tools. None means rg is not on PATH; the tools
# are not registered in that case and the LLM must fall back to
# execute_command("rg ..."). No env override (one-line detection by design).
RG_PATH = shutil.which("rg")


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


def relativize(path: str) -> str:
    """Turn an absolute path into a CWD-relative form for output.

    grep/glob emit paths to the LLM; relative paths cost fewer tokens and
    match how the user thinks about the project. Falls back to the input
    unchanged for paths outside CWD (e.g. symlinks resolved elsewhere).
    """
    try:
        rel = os.path.relpath(path, CWD)
    except ValueError:
        # Different drives on Windows — relpath raises; return path as-is.
        return path
    return rel if not rel.startswith("..") else path
