from __future__ import annotations

import re
import threading
from datetime import datetime, timezone
from pathlib import Path

MEMORY_TYPES = ("user", "feedback", "project", "reference")
_INDEX_FILENAME = "MEMORY.md"
_INDEX_MAX_LINES = 200
_MAX_INDEX_LINE_LENGTH = 150
_DEDUP_SIMILARITY = 0.7
_STALE_INDEX_DAYS = 90
_KEEP_FOREVER = ("user", "feedback")


def _tokenize(text: str) -> set[str]:
    """Tokenizer supporting English (word-level) and CJK (char + bigram)."""
    tokens = set()
    text_lower = text.lower()
    for word in re.findall(r'[a-z0-9_]+', text_lower):
        tokens.add(word)
    cjk = re.findall(r'[一-鿿]', text_lower)
    tokens.update(cjk)
    for i in range(len(cjk) - 1):
        tokens.add(cjk[i] + cjk[i + 1])
    return tokens


def _similarity(a: str, b: str) -> float:
    """Jaccard similarity between two strings based on word overlap."""
    wa, wb = _tokenize(a), _tokenize(b)
    union = wa | wb
    if not union:
        return 0.0
    return len(wa & wb) / len(union)


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
        # web UI 跑在独立线程、agent 的 auto_extract 在事件循环线程——两者可能同时写
        # 同一批记忆文件 + 重建索引，无锁会交错致丢写/索引损坏。写操作串行化。
        self._write_lock = threading.Lock()

    def _file_path(self, name: str) -> Path:
        return self.base_dir / f"{name}.md"

    def save(self, name: str, description: str, type: str, content: str, source: str = "") -> str:
        """Create or update a memory file. Returns the sanitized name.

        source 非空时写入 frontmatter 的 source_session（溯源：这条记忆由哪个
        写入者/会话产生，毒记忆事故后加——出问题五秒定位来源会话）。更新时覆写
        为最新写入者。
        """
        with self._write_lock:
            return self._save_locked(name, description, type, content, source)

    def _save_locked(self, name: str, description: str, type: str, content: str, source: str) -> str:
        name = _sanitize_name(name)

        if type not in MEMORY_TYPES:
            type = "project"

        # Dedup: only merge into an existing memory of the SAME type whose
        # description is very similar. The type guard is a data-safety measure —
        # a `user` fact and a `project` fact are never the same memory no matter
        # how similar their descriptions read. Without it, a lexical collision on
        # short descriptions would silently redirect this write and overwrite an
        # unrelated entry's content. Semantic dedup is handled upstream by the
        # auto-extract prompt; this loop is just a conservative deterministic backstop.
        for existing in self.list_all():
            if existing["name"] == name:
                continue
            if existing["type"] != type:
                continue
            if _similarity(description, existing["description"]) >= _DEDUP_SIMILARITY:
                name = existing["name"]

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
        if source:
            meta["source_session"] = source

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
        with self._write_lock:
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

    def _is_stale_for_index(self, entry: dict) -> bool:
        """Check if an entry is too old to appear in the index.

        user and feedback memories are kept forever.
        project and reference memories older than _STALE_INDEX_DAYS are trimmed
        from the index (but remain on disk for recall/listing).
        """
        if entry["type"] in _KEEP_FOREVER:
            return False
        updated = entry.get("updated_at", "")
        if not updated:
            return False
        try:
            dt = datetime.fromisoformat(updated)
            age = datetime.now(timezone.utc) - dt
            return age.days > _STALE_INDEX_DAYS
        except (ValueError, TypeError):
            return False

    def _rebuild_index(self) -> None:
        """Rebuild MEMORY.md index from all .md files in base_dir.

        Stale project/reference entries (older than _STALE_INDEX_DAYS) are
        excluded from the index but remain on disk for recall/listing.
        """
        entries = self.list_all()
        # Filter out stale project/reference memories
        entries = [e for e in entries if not self._is_stale_for_index(e)]
        if not entries:
            # Remove stale index if no memories remain
            index_path = self.base_dir / _INDEX_FILENAME
            if index_path.exists():
                index_path.unlink()
            return

        lines = ["# Memory Index", ""]
        for entry in entries[:_INDEX_MAX_LINES - 2]:
            prefix = f"- [{entry['name']}]({entry['name']}.md) — "
            desc = entry["description"]
            max_desc_len = _MAX_INDEX_LINE_LENGTH - len(prefix)
            if max_desc_len > 3 and len(desc) > max_desc_len:
                desc = desc[:max_desc_len - 3] + "..."
            lines.append(f"{prefix}{desc}")

        index_path = self.base_dir / _INDEX_FILENAME
        index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def index_is_complete(self, max_bytes: int = 25_000) -> bool:
        """True when MEMORY.md lists every memory — nothing hidden from the model.

        The index drops entries three ways: stale-trim (old project/reference),
        the _INDEX_MAX_LINES cap, and byte-truncation on load. If any applies,
        some memories are invisible in the always-loaded index, so automatic
        semantic recall earns its keep. If none applies, the model already sees
        every description and automatic recall is pure redundant cost.
        """
        entries = self.list_all()
        if not entries:
            return True
        non_stale = [e for e in entries if not self._is_stale_for_index(e)]
        if len(non_stale) < len(entries):
            return False  # stale-trimmed entries missing from index
        if len(non_stale) > _INDEX_MAX_LINES - 2:
            return False  # line-capped
        index_path = self.base_dir / _INDEX_FILENAME
        if not index_path.exists() or index_path.stat().st_size > max_bytes:
            return False  # no index shown, or byte-truncated on load
        return True

    def get_index(self, max_bytes: int = 25_000) -> str:
        """Read MEMORY.md content, truncate if needed. Returns '' if empty/missing.

        Truncation is byte-aware: caps at max_bytes to defend against
        long-line index bombs that stay under the 200-line limit but
        still carry hundreds of KB (e.g. verbose Chinese descriptions).
        """
        index_path = self.base_dir / _INDEX_FILENAME
        if not index_path.exists():
            return ""

        raw = index_path.read_bytes()
        if len(raw) == 0:
            return ""

        if len(raw) > max_bytes:
            # Truncate at last complete UTF-8 byte boundary before max_bytes
            cut = max_bytes
            while cut > 0 and (raw[cut - 1] & 0xC0) == 0x80:
                cut -= 1
            truncated = raw[:cut]
            # Find last newline to avoid cutting mid-line
            last_nl = truncated.rfind(b"\n")
            if last_nl > 0:
                truncated = truncated[:last_nl]
            text = truncated.decode("utf-8")
            text += f"\n\n> ⚠ 索引已截断（{len(raw)} 字节中只加载了前 {len(truncated)} 字节），部分记忆未显示。使用 recall 工具检索遗漏的条目。\n"
            return text

        return raw.decode("utf-8")
