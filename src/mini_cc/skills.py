import re
import yaml
from pathlib import Path

from mini_cc.config import CWD


class SkillManager:
    """Manages skills in CWD/skills/. Lazy-loads body content on demand."""

    def __init__(self, skills_dir: Path):
        self._dir = skills_dir
        self._meta: dict[str, dict] = {}   # name -> {description, _path}
        self._body: dict[str, str] = {}    # name -> body (cached on first access)
        self._fingerprints: dict[str, float] = {}  # name -> SKILL.md mtime
        self.rescan()

    @staticmethod
    def _parse_body(content: str) -> str:
        """Strip YAML frontmatter (---...---) and return the markdown body."""
        m = re.match(r'^---\s*\n.*?\n---\s*\n?', content, re.DOTALL)
        return content[m.end():] if m else content

    def rescan(self):
        """Rescan skills dir. Detects added/removed/changed skills."""
        if not self._dir.exists():
            self._meta.clear()
            self._fingerprints.clear()
            self._body.clear()
            return
        current_meta = {}
        current_fp = {}
        for skill_dir in self._dir.iterdir():
            if not skill_dir.is_dir():
                continue
            skill_md = skill_dir / "SKILL.md"
            if not skill_md.exists():
                continue
            try:
                content = skill_md.read_text(encoding="utf-8")
                m = re.match(r'^---\s*\n(.*?)\n---', content, re.DOTALL)
                if not m:
                    continue
                meta = yaml.safe_load(m.group(1)) or {}
                if "name" not in meta:
                    continue
            except Exception as e:
                print(f"  [warning: skipping {skill_dir.name}/SKILL.md — {e}]")
                continue
            name = meta["name"]
            current_meta[name] = {"description": meta.get("description", ""),
                                  "_path": skill_md}
            current_fp[name] = skill_md.stat().st_mtime

        # Clear body cache for removed + changed skills
        for name in set(self._meta) - set(current_meta):
            self._body.pop(name, None)
        for name in set(current_meta) & set(self._meta):
            if current_fp.get(name) != self._fingerprints.get(name):
                self._body.pop(name, None)

        self._meta = current_meta
        self._fingerprints = current_fp

    def names(self) -> list[str]:
        return list(self._meta)

    def body(self, name: str) -> str | None:
        """Return skill body, lazily loaded. None if skill not found or deleted."""
        if name not in self._meta:
            return None
        if name not in self._body:
            try:
                content = self._meta[name]["_path"].read_text(encoding="utf-8")
            except (FileNotFoundError, OSError):
                return None
            self._body[name] = self._parse_body(content)
        return self._body[name]

    def description(self, name: str) -> str:
        """Return skill description, or empty string if not found."""
        return self._meta.get(name, {}).get("description", "")

    def unload(self, name: str):
        """Release cached body to free memory."""
        self._body.pop(name, None)

    def prompt_section(self) -> str:
        """Build system prompt section listing available skills."""
        if not self._meta:
            return ""
        lines = ["\n\n## Available Skills",
                 "Call run_skill(name, request) when the user's request matches a skill, "
                 "or the user invokes /skill-name:"]
        for name, meta in self._meta.items():
            lines.append(f"- {name}: {meta['description']}")
        return "\n".join(lines)


_skill_manager = SkillManager(Path(CWD) / "skills")
