class TodoManager:
    _ICONS = {"pending": "○", "in_progress": "◉", "done": "✓"}
    _VALID = frozenset(_ICONS)

    def __init__(self):
        self._todos: list[dict] = []

    def plan(self, items: list[str]) -> str:
        self._todos = [{"item": i, "status": "pending"} for i in items]
        return self._render()

    def update(self, item: str, status: str) -> str:
        if status not in self._VALID:
            return f"Error: status must be one of {sorted(self._VALID)}"
        if status == "in_progress":
            busy = next((t for t in self._todos if t["status"] == "in_progress"), None)
            if busy:
                return f"Error: '{busy['item']}' is already in_progress. Mark it done first."
        todo = next((t for t in self._todos if t["item"] == item), None)
        if todo is None:
            return f"Error: '{item}' not found. Call plan_todos first."
        todo["status"] = status
        return self._render()

    def _render(self) -> str:
        if not self._todos:
            return "TODOs: (empty)"
        lines = [f"  {self._ICONS.get(t['status'], '?')} {t['item']}" for t in self._todos]
        return "TODOs:\n" + "\n".join(lines)


# NOTE: _todos is reassigned (not just mutated) by tools.py's task tool
# for sub-agent isolation. `from todos import _todos` would capture a stale
# reference — always access as todos._todos (module attribute lookup).
_todos = TodoManager()
