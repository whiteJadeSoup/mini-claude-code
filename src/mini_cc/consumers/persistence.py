"""Append-only JSONL transcript per session.

Writes Layer 1 (API) and Layer 2 (UI) messages to
~/.minicc/projects/<cwd-slug>/<session-uuid>.jsonl as one JSON object per line.
I/O failures degrade silently — a missing transcript line is always better
than a crashed agent.

Wire via: engine.subscribe(PersistenceConsumer())
"""
import json
import re
import sys
import uuid
from pathlib import Path

from mini_cc import config
from mini_cc.consumers.base import QueuedConsumer

SESSION_ID: str = str(uuid.uuid4())


def _cwd_slug() -> str:
    return re.sub(r"[:\\/\s]", "-", config.CWD).strip("-")


def transcript_path() -> Path:
    path = Path.home() / ".minicc" / "projects" / _cwd_slug() / f"{SESSION_ID}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _on_append(msg) -> None:
    """Sync JSONL writer. Also used by PersistenceConsumer (async wrapper).

    Kept as a module-level function so tests can call it synchronously
    without spinning up an event loop — the I/O itself is sync anyway.
    """
    from mini_cc.engine.messages import LAYER_1_TYPES, LAYER_2_TYPES
    if not isinstance(msg, LAYER_1_TYPES + LAYER_2_TYPES):
        return
    try:
        record = msg.model_dump(mode="json")
        with transcript_path().open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except OSError as e:
        # Degrade silently — transcript loss is acceptable, crashing is not.
        print(f"[persistence: {e}]", file=sys.stderr)


class PersistenceConsumer(QueuedConsumer):
    """Engine consumer that appends Layer 1 + Layer 2 messages to JSONL."""

    async def _handle(self, msg) -> None:
        _on_append(msg)
