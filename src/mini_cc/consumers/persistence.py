"""Append-only JSONL transcript per session.

Writes each received Message to
``~/.minicc/projects/<cwd-slug>/<session-uuid>.jsonl`` as one JSON object
per line. I/O failures degrade silently — a missing transcript line is
always better than a crashed agent.

Layer filtering is declared at the subscription site (via
``predicates.is_persisted_layer``), not here. This consumer assumes every
message it receives should be written.

Wire via::

    engine.subscribe(
        PersistenceConsumer(),
        name="persistence",
        filter=is_persisted_layer,
        policy="async",
    )
"""
import json
import re
import sys
import uuid
from pathlib import Path

from mini_cc import config
from mini_cc.engine.messages import Message

SESSION_ID: str = str(uuid.uuid4())


def _cwd_slug() -> str:
    return re.sub(r"[:\\/\s]", "-", config.CWD).strip("-")


def transcript_path() -> Path:
    path = Path.home() / ".minicc" / "projects" / _cwd_slug() / f"{SESSION_ID}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def tool_result_path(tool_call_id: str) -> Path:
    """Landing path for the full content of an oversized tool result.

    Co-located with the session's transcript so an offline reader can pair
    a truncated ToolResultMessage with its full payload. The model can
    also re-read this path via the Read tool if the in-context preview
    isn't enough to act on.
    """
    return transcript_path().parent / "tool_results" / f"{tool_call_id}.txt"


def _on_append(msg: Message) -> None:
    """Sync JSONL writer. Module-level so tests can call it without an event loop."""
    try:
        record = msg.model_dump(mode="json")
        with transcript_path().open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except OSError as e:
        print(f"[persistence: {e}]", file=sys.stderr)


class PersistenceConsumer:
    async def on_message(self, msg: Message) -> None:
        _on_append(msg)
