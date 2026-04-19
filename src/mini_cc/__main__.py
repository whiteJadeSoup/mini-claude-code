"""Entry point for mini-cc."""
from __future__ import annotations


def _suppress_asyncio_cleanup_noise() -> None:
    """Filter 'I/O operation on closed pipe' unraisable exceptions on Windows.

    When asyncio subprocess pipe transports are GC'd after the event loop
    closes, their __del__ calls fileno() on a dead socket and raises ValueError.
    These are cosmetic — the process already exited cleanly.
    """
    import sys

    _orig = sys.unraisablehook

    def _hook(args: "sys.UnraisableHookArgs") -> None:
        if isinstance(args.exc_value, ValueError) and "closed pipe" in str(args.exc_value):
            return
        _orig(args)

    sys.unraisablehook = _hook


def main() -> None:
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    _suppress_asyncio_cleanup_noise()
    from dotenv import load_dotenv
    load_dotenv()
    # Pre-warm heavy imports before Textual starts so on_mount doesn't block
    # the already-visible UI waiting for langchain / tool modules to load.
    import mini_cc.llm                    # noqa: F401 — LLM client + tool lists
    import mini_cc.commands               # noqa: F401 — command registry
    import mini_cc.consumers.persistence  # noqa: F401 — transcript writer
    import mini_cc.engine.query_engine    # noqa: F401 — engine + agent loop
    from mini_cc.consumers.tui.app import MiniCCApp
    MiniCCApp().run()


if __name__ == "__main__":
    main()
