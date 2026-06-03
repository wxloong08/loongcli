from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

MEMORY_TYPES = ("user", "feedback", "project", "reference")
_INDEX_FILENAME = "MEMORY.md"
_INDEX_MAX_LINES = 200


def _sanitize_name(name: str) -> str:
    """Replace any character that is not alphanumeric, dash, or underscore with a dash.

    Consecutive dashes are collapsed and trailing dashes are stripped.
    """
    result = re.sub(r"[^a-zA-Z0-9_-]", "-", name)
    result = re.sub(r"-{2,}", "-", result)
    result = result.strip("-")
    if not result:
        raise ValueError("Memory name must contain at least one alphanumeric character")
    return result


def _quote_value(value: str) -> str:
    """Quote a YAML value if it contains colons or leading/trailing spaces."""
    if ":" in value or value != value.strip() or value.startswith('"'):
        return f'"{value}"'
    return value


def _unquote_value(value: str) -> str:
    """Remove surrounding quotes from a YAML value."""
    stripped = value.strip()
    if len(stripped) >= 2 and stripped[0] == '"' and stripped[-1] == '"':
        return stripped[1:-1]
    return stripped


def _render_frontmatter(meta: dict[str, str]) -> str:
    """Render a dict as YAML frontmatter block."""
    lines = ["---"]
    for key, val in meta.items():
        lines.append(f"{key}: {_quote_value(val)}")
    lines.append("---")
    return "\n".join(lines)


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Parse YAML frontmatter and body from a markdown file.

    Returns (meta_dict, body_string). If no valid frontmatter found,
    returns ({}, full_text).
    """
    if not text.startswith("---"):
        return {}, text

    # Find the closing --- (skip the opening one)
    # Split only on the first two --- markers
    rest = text[3:]  # skip opening ---
    if rest.startswith("\n"):
        rest = rest[1:]

    closing_idx = rest.find("\n---")
    if closing_idx == -1:
        return {}, text

    fm_block = rest[:closing_idx]
    body = rest[closing_idx + 4:]  # skip \n---
    if body.startswith("\n"):
        body = body[1:]

    meta: dict[str, str] = {}
    for line in fm_block.split("\n"):
        line = line.strip()
        if not line:
            continue
        colon_idx = line.find(":")
        if colon_idx == -1:
            continue
        key = line[:colon_idx].strip()
        val = line[colon_idx + 1:].strip()
        meta[key] = _unquote_value(val)

    return meta, body


class MarkdownMemoryStore:
    """Markdown-file-based memory store with YAML frontmatter.

    Each memory is stored as a separate `.md` file under `base_dir`.
    An index file (MEMORY.md) is automatically maintained.
    """

    def __init__(self, base_dir: Path | None = None):
        self.base_dir = base_dir or (Path.home() / ".loongcli" / "memory")
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _file_path(self, name: str) -> Path:
        return self.base_dir / f"{name}.md"

    def save(self, name: str, description: str, type: str, content: str) -> str:
        """Create or update a memory file. Returns the sanitized name."""
        name = _sanitize_name(name)

        if type not in MEMORY_TYPES:
            type = "project"

        now = datetime.now(timezone.utc).isoformat()
        created_at = now

        # Preserve created_at on update
        path = self._file_path(name)
        if path.exists():
            existing_text = path.read_text(encoding="utf-8")
            existing_meta, _ = _parse_frontmatter(existing_text)
            if "created_at" in existing_meta:
                created_at = existing_meta["created_at"]

        meta = {
            "name": name,
            "description": description,
            "type": type,
            "created_at": created_at,
            "updated_at": now,
        }

        file_content = _render_frontmatter(meta) + "\n\n" + content
        path.write_text(file_content, encoding="utf-8")

        self._rebuild_index()
        return name

    def load(self, name: str) -> dict | None:
        """Load a memory by name. Returns None if not found."""
        path = self._file_path(name)
        if not path.exists():
            return None

        text = path.read_text(encoding="utf-8")
        meta, body = _parse_frontmatter(text)

        return {
            "name": meta.get("name", name),
            "description": meta.get("description", ""),
            "type": meta.get("type", "project"),
            "created_at": meta.get("created_at", ""),
            "updated_at": meta.get("updated_at", ""),
            "content": body.strip(),
        }

    def delete(self, name: str) -> bool:
        """Delete a memory file. Returns True if deleted, False if not found."""
        path = self._file_path(name)
        if not path.exists():
            return False

        path.unlink()
        self._rebuild_index()
        return True

    def list_all(self, type_filter: str | None = None) -> list[dict]:
        """List all memories, optionally filtered by type.

        Excludes MEMORY.md from results.
        """
        results: list[dict] = []
        for md_file in sorted(self.base_dir.glob("*.md")):
            if md_file.name == _INDEX_FILENAME:
                continue

            text = md_file.read_text(encoding="utf-8")
            meta, _ = _parse_frontmatter(text)

            entry_type = meta.get("type", "project")
            if type_filter is not None and entry_type != type_filter:
                continue

            results.append({
                "name": meta.get("name", md_file.stem),
                "description": meta.get("description", ""),
                "type": entry_type,
                "created_at": meta.get("created_at", ""),
                "updated_at": meta.get("updated_at", ""),
            })

        return results

    def _rebuild_index(self) -> None:
        """Rebuild MEMORY.md index from all .md files in base_dir."""
        entries = self.list_all()
        if not entries:
            # Remove stale index if no memories remain
            index_path = self.base_dir / _INDEX_FILENAME
            if index_path.exists():
                index_path.unlink()
            return

        lines = ["# Memory Index", ""]
        for entry in entries[:_INDEX_MAX_LINES - 2]:  # reserve 2 lines for header
            lines.append(f"- [{entry['name']}]({entry['name']}.md) — {entry['description']}")

        index_path = self.base_dir / _INDEX_FILENAME
        index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def get_index(self, max_chars: int = 6000) -> str:
        """Read MEMORY.md content, truncate if needed. Returns '' if empty/missing."""
        index_path = self.base_dir / _INDEX_FILENAME
        if not index_path.exists():
            return ""

        text = index_path.read_text(encoding="utf-8")
        if not text.strip():
            return ""

        if len(text) > max_chars:
            cut = text[:max_chars].rfind("\n")
            text = text[:cut] if cut > 0 else text[:max_chars]

        return text
