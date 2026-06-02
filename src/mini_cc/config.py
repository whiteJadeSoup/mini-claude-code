import os, platform, shutil
from pathlib import Path

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

# Bundled ripgrep — placed by hatch_build.py at install time. RG_PATH=None
# means the build hook didn't land a binary (network failure, unsupported
# platform, or the package is being run from an unbuilt source tree); in that
# case grep/glob don't register and the LLM is steered to reinstall.
# No `shutil.which("rg")` system fallback: bundling is supposed to be the
# always-present invariant, mixing in system rg would let version mismatches
# leak in.
_VENDOR_RG = Path(__file__).parent / "_vendor" / ("rg.exe" if PLATFORM == "Windows" else "rg")
RG_PATH: str | None = str(_VENDOR_RG) if _VENDOR_RG.is_file() else None


def safe_path(path: str) -> str:
    """Resolve path relative to CWD and reject escapes, with a memdir whitelist.

    Two zones are writable: the project CWD subtree (sandbox), and the
    auto-memory directory subtree (~/.minicc/projects/<slug>/memory/, derived
    by mini_cc.memdir.get_auto_mem_path). The memdir whitelist exists because
    the system prompt instructs the LLM to write memory files there with
    file_write — without this carve-out the sandbox would reject every memory
    write and the prompt would be a lie.
    """
    resolved = os.path.realpath(os.path.join(CWD, path))
    # normcase for Windows case-insensitive comparison (no-op on Unix)
    resolved_norm = os.path.normcase(resolved)
    cwd_norm = os.path.normcase(CWD)
    if resolved_norm == cwd_norm or resolved_norm.startswith(cwd_norm + os.sep):
        return resolved

    # Lazy: avoid import-time git rev-parse subprocess. is_memory_path does the
    # normcase memdir comparison (get_auto_mem_path is @cache'd: ~0ms after first).
    from mini_cc.memdir import is_memory_path
    if is_memory_path(resolved):
        return resolved

    raise ValueError(f"Path {path} is outside working directory or memdir")


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
