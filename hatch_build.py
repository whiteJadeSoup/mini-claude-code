"""Hatchling build hook: download platform-specific ripgrep into _vendor/.

Runs at `pip install` / `uv pip install -e .` time, mirroring CC's npm
postinstall pattern. Result lands at `src/mini_cc/_vendor/rg(.exe)` and is
included in the wheel as package data.

Idempotent: if the binary already exists at the pinned version, skip download.
Graceful degradation: network failure logs a warning and leaves _vendor/ empty
— the package still installs, but config.RG_PATH ends up None and grep/glob
don't register (same fallback path as before bundling).
"""
from __future__ import annotations

import io
import os
import platform
import stat
import sys
import tarfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

# Pin a known-good ripgrep release. Bump this when upgrading; binary cache
# keys off the version sentinel file so a version change forces re-download.
RG_VERSION = "14.1.1"

# (system, machine) → GitHub release asset name (relative to release tag URL).
# The asset list comes from https://github.com/BurntSushi/ripgrep/releases.
# Linux uses musl static builds for max portability across distros.
_ASSETS: dict[tuple[str, str], str] = {
    ("Windows", "AMD64"):    f"ripgrep-{RG_VERSION}-x86_64-pc-windows-msvc.zip",
    ("Windows", "x86_64"):   f"ripgrep-{RG_VERSION}-x86_64-pc-windows-msvc.zip",
    ("Windows", "ARM64"):    f"ripgrep-{RG_VERSION}-aarch64-pc-windows-msvc.zip",
    ("Linux",   "x86_64"):   f"ripgrep-{RG_VERSION}-x86_64-unknown-linux-musl.tar.gz",
    ("Linux",   "aarch64"):  f"ripgrep-{RG_VERSION}-aarch64-unknown-linux-gnu.tar.gz",
    ("Linux",   "armv7l"):   f"ripgrep-{RG_VERSION}-armv7-unknown-linux-gnueabihf.tar.gz",
    ("Darwin",  "x86_64"):   f"ripgrep-{RG_VERSION}-x86_64-apple-darwin.tar.gz",
    ("Darwin",  "arm64"):    f"ripgrep-{RG_VERSION}-aarch64-apple-darwin.tar.gz",
}

_RELEASE_URL = (
    "https://github.com/BurntSushi/ripgrep/releases/download/"
    "{version}/{asset}"
)


def _vendor_dir() -> Path:
    return Path(__file__).parent / "src" / "mini_cc" / "_vendor"


def _binary_name() -> str:
    return "rg.exe" if platform.system() == "Windows" else "rg"


def _version_sentinel(vendor: Path) -> Path:
    # Plain text file recording the version of the cached binary; lets the
    # build hook know to re-download when RG_VERSION bumps.
    return vendor / ".rg_version"


def _resolve_asset() -> str | None:
    key = (platform.system(), platform.machine())
    asset = _ASSETS.get(key)
    if asset is None:
        # Try a normalized lookup — Windows reports "AMD64", but some setups
        # report it as "x86_64". The dict carries both, so this is a no-op
        # for currently-known platforms; kept as a safety net.
        return None
    return asset


def _download_and_extract(asset: str, dest: Path) -> None:
    url = _RELEASE_URL.format(version=RG_VERSION, asset=asset)
    print(f"[mini-cc] fetching ripgrep {RG_VERSION} for "
          f"{platform.system()}/{platform.machine()} → {dest}", flush=True)
    print(f"[mini-cc]   {url}", flush=True)

    with urllib.request.urlopen(url, timeout=60) as resp:
        data = resp.read()

    bin_name = _binary_name()
    if asset.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            # The archive contains `<stem>/rg.exe` plus docs. Pull just rg.
            members = [m for m in zf.namelist() if m.endswith(f"/{bin_name}")]
            if not members:
                raise RuntimeError(
                    f"rg binary not found inside {asset} (members={zf.namelist()[:5]}…)"
                )
            with zf.open(members[0]) as src, open(dest, "wb") as out:
                out.write(src.read())
    else:
        # tar.gz
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
            members = [m for m in tf.getmembers()
                       if m.name.endswith(f"/{bin_name}") and m.isfile()]
            if not members:
                raise RuntimeError(
                    f"rg binary not found inside {asset}"
                )
            extracted = tf.extractfile(members[0])
            if extracted is None:
                raise RuntimeError(f"failed to extract rg from {asset}")
            with open(dest, "wb") as out:
                out.write(extracted.read())

    # Unix needs +x; Windows ignores chmod.
    mode = os.stat(dest).st_mode
    os.chmod(dest, mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _ensure_vendored_rg() -> bool:
    """Returns True if a usable rg ended up in _vendor/, False on failure."""
    vendor = _vendor_dir()
    vendor.mkdir(parents=True, exist_ok=True)
    # Marker file so editable rebuilds don't re-download.
    init_py = vendor / "__init__.py"
    if not init_py.exists():
        init_py.write_text(
            "# Vendored binaries — not Python.\n"
            "# Populated by hatch_build.py at install time.\n",
            encoding="utf-8",
        )

    binary = vendor / _binary_name()
    sentinel = _version_sentinel(vendor)
    if (
        binary.is_file()
        and sentinel.is_file()
        and sentinel.read_text(encoding="utf-8").strip() == RG_VERSION
    ):
        # Already cached with the right version — short-circuit.
        return True

    asset = _resolve_asset()
    if asset is None:
        print(
            f"[mini-cc] WARN: no prebuilt ripgrep asset known for "
            f"{platform.system()}/{platform.machine()}. "
            f"grep/glob tools will not register; install rg manually if needed.",
            file=sys.stderr,
            flush=True,
        )
        return False

    try:
        _download_and_extract(asset, binary)
    except (urllib.error.URLError, OSError, RuntimeError) as e:
        print(
            f"[mini-cc] WARN: failed to fetch ripgrep ({type(e).__name__}: {e}). "
            f"grep/glob tools will not register. mini-cc still works for "
            f"everything else; reinstall when network is available.",
            file=sys.stderr,
            flush=True,
        )
        # Best effort: leave _vendor/ empty so config.RG_PATH ends up None.
        if binary.exists():
            binary.unlink(missing_ok=True)
        return False

    sentinel.write_text(RG_VERSION + "\n", encoding="utf-8")
    return True


class CustomBuildHook(BuildHookInterface):
    PLUGIN_NAME = "custom"

    def initialize(self, version: str, build_data: dict) -> None:
        # Run for both wheel and editable targets; sdist builds skip vendor
        # (no point baking a host-platform binary into a source distribution
        # that may be installed on a different platform).
        target = self.target_name
        if target == "sdist":
            return
        _ensure_vendored_rg()
