from __future__ import annotations
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_NAME_RE = re.compile(r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?$")


MAX_AUTO_SKILLS = 2
MAX_SKILL_CHARS = 5000


@dataclass
class SkillMeta:
    name: str
    description: str
    path: Path
    disable_model_invocation: bool = False
    allowed_tools: list[str] = field(default_factory=list)
    triggers: list[str] = field(default_factory=list)


def _parse_frontmatter(text: str) -> dict[str, Any]:
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {}
    block = match.group(1)
    try:
        data = yaml.safe_load(block)
    except yaml.YAMLError:
        return {}
    if not isinstance(data, dict):
        return {}
    return data


def _extract_body(text: str) -> str:
    match = _FRONTMATTER_RE.match(text)
    if match:
        return text[match.end():]
    return text


def _parse_skill_file(path: Path) -> SkillMeta | None:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        logger.warning("Failed to read skill file %s: %s", path, e)
        return None

    fm = _parse_frontmatter(text)
    name = str(fm.get("name", "")).strip()
    description = str(fm.get("description", "")).strip()

    if not name or not description:
        logger.warning("Skill %s missing required name or description", path)
        return None

    if not _NAME_RE.match(name):
        logger.warning("Skill name '%s' does not match naming convention", name)
        return None

    disable_raw = fm.get("disable-model-invocation", False)
    disable_model = bool(disable_raw) if isinstance(disable_raw, bool) else str(disable_raw).lower() in ("true", "yes", "1")
    tools_str = str(fm.get("allowed-tools", ""))
    allowed_tools = tools_str.split() if tools_str else []

    triggers_raw = fm.get("triggers", [])
    if isinstance(triggers_raw, list):
        triggers = [str(t).strip().lower() for t in triggers_raw if str(t).strip()]
    else:
        triggers = []

    return SkillMeta(
        name=name,
        description=description,
        path=path,
        disable_model_invocation=disable_model,
        allowed_tools=allowed_tools,
        triggers=triggers,
    )


class SkillRegistry:
    def __init__(
        self,
        project_dir: Path | None = None,
        personal_dir: Path | None = None,
        extra_dirs: list[str | Path] | None = None,
    ):
        self._skills: dict[str, SkillMeta] = {}
        self._scan(project_dir, personal_dir, extra_dirs)

    def _scan(
        self,
        project_dir: Path | None,
        personal_dir: Path | None,
        extra_dirs: list[str | Path] | None,
    ):
        if personal_dir is None:
            personal_dir = Path.home() / ".loongcli" / "skills"

        dirs = []
        if project_dir:
            dirs.append(project_dir / ".loongcli" / "skills")
        dirs.append(personal_dir)
        for d in (extra_dirs or []):
            dirs.append(Path(d))

        for base in dirs:
            if not base.is_dir():
                continue
            self._scan_dir(base)

        logger.info("Loaded %d skills", len(self._skills))

    def _scan_dir(self, base: Path):
        for entry in sorted(base.iterdir()):
            if not entry.is_dir():
                continue
            skill_md = entry / "SKILL.md"
            if not skill_md.is_file():
                skill_md = entry / "skill.md"
            if skill_md.is_file():
                meta = _parse_skill_file(skill_md)
                if meta and meta.name not in self._skills:
                    self._skills[meta.name] = meta
            else:
                self._scan_nested_skills(entry)

    def _scan_nested_skills(self, project_dir: Path):
        for prefix in (".claude/skills", ".loongcli/skills"):
            skills_dir = project_dir / prefix
            if skills_dir.is_dir():
                self._scan_dir(skills_dir)

    def get(self, name: str) -> SkillMeta | None:
        return self._skills.get(name)

    def list_skills(self, include_model_disabled: bool = True) -> list[SkillMeta]:
        skills = list(self._skills.values())
        if not include_model_disabled:
            skills = [s for s in skills if not s.disable_model_invocation]
        return sorted(skills, key=lambda s: s.name)

    def load_content(self, name: str) -> str | None:
        meta = self._skills.get(name)
        if not meta:
            return None
        try:
            text = meta.path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return None
        return _extract_body(text)

    def match(self, user_input: str) -> list[SkillMeta]:
        lower = user_input.lower()
        matched: list[SkillMeta] = []
        for meta in self._skills.values():
            if not meta.triggers:
                continue
            if any(t in lower for t in meta.triggers):
                matched.append(meta)
                if len(matched) >= MAX_AUTO_SKILLS:
                    break
        return matched

    def enrich_prompt(self, user_input: str) -> str:
        matched = self.match(user_input)
        if not matched:
            return user_input
        parts = [user_input]
        for meta in matched:
            content = self.load_content(meta.name)
            if not content:
                continue
            if len(content) > MAX_SKILL_CHARS:
                content = content[:MAX_SKILL_CHARS] + "\n...(截断)"
            parts.append(f"\n---\n[自动加载 skill: {meta.name}]\n{content}")
        return "".join(parts)

    def build_listing(self, include_model_disabled: bool = True, verbose: bool = False) -> str:
        skills = self.list_skills(include_model_disabled)
        if not skills:
            return ""
        if verbose:
            return "\n".join(f"- {s.name}: {s.description}" for s in skills)
        return ", ".join(s.name for s in skills)
