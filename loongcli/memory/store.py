from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MEMORY_TYPES = ("user", "feedback", "project", "reference")
PRIORITY_ORDER = ["feedback", "user", "project", "reference"]


class MemoryEntry:
    __slots__ = ("value", "type", "created_at", "updated_at")

    def __init__(
        self,
        value: str,
        type: str = "project",
        created_at: str | None = None,
        updated_at: str | None = None,
    ):
        self.value = value
        self.type = type if type in MEMORY_TYPES else "project"
        now = datetime.now(timezone.utc).isoformat()
        self.created_at = created_at or now
        self.updated_at = updated_at or now

    def to_dict(self) -> dict[str, str]:
        return {
            "value": self.value,
            "type": self.type,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_raw(cls, raw: Any) -> MemoryEntry:
        if isinstance(raw, dict) and "value" in raw:
            return cls(
                value=raw["value"],
                type=raw.get("type", "project"),
                created_at=raw.get("created_at"),
                updated_at=raw.get("updated_at"),
            )
        return cls(value=str(raw))


class MemoryStore:
    """JSON-backed KV store with typed metadata."""

    def __init__(self, path: Path | None = None):
        self.path = path or Path.home() / ".loongcli" / "memory" / "kv.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, dict[str, MemoryEntry]] = {}
        self._load()

    def _load(self):
        if not self.path.exists():
            return
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        for cat, kvs in raw.items():
            self._data[cat] = {}
            for key, val in kvs.items():
                self._data[cat][key] = MemoryEntry.from_raw(val)

    def _save(self):
        out: dict[str, dict[str, Any]] = {}
        for cat, kvs in self._data.items():
            out[cat] = {k: e.to_dict() for k, e in kvs.items()}
        self.path.write_text(
            json.dumps(out, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def set(self, category: str, key: str, value: str, type: str = "project"):
        existing = self._data.get(category, {}).get(key)
        if existing:
            existing.value = value
            existing.type = type if type in MEMORY_TYPES else existing.type
            existing.updated_at = datetime.now(timezone.utc).isoformat()
        else:
            self._data.setdefault(category, {})[key] = MemoryEntry(
                value=value, type=type,
            )
        self._save()

    def get(self, category: str, key: str) -> str | None:
        entry = self._data.get(category, {}).get(key)
        return entry.value if entry else None

    def get_entry(self, category: str, key: str) -> MemoryEntry | None:
        return self._data.get(category, {}).get(key)

    def delete(self, category: str, key: str | None = None) -> bool:
        if category not in self._data:
            return False
        if key is None:
            del self._data[category]
            self._save()
            return True
        if key in self._data[category]:
            del self._data[category][key]
            if not self._data[category]:
                del self._data[category]
            self._save()
            return True
        return False

    def get_all(self, category: str) -> dict[str, str]:
        return {k: e.value for k, e in self._data.get(category, {}).items()}

    def list_categories(self) -> dict[str, int]:
        return {cat: len(kvs) for cat, kvs in self._data.items()}

    def all_entries(self) -> list[tuple[str, str, MemoryEntry]]:
        result = []
        for cat, kvs in self._data.items():
            for key, entry in kvs.items():
                result.append((cat, key, entry))
        return result

    def summary(self) -> str:
        cats = self.list_categories()
        if not cats:
            return ""
        return " | ".join(f"{cat}({count})" for cat, count in cats.items())

    def format_all(self) -> str:
        if not self._data:
            return "（暂无记忆）"
        lines: list[str] = []
        for cat, kvs in self._data.items():
            lines.append(f"[{cat}]")
            for k, e in kvs.items():
                lines.append(f"  {k}: {e.value}")
        return "\n".join(lines)

    def format_category(self, category: str) -> str:
        kvs = self._data.get(category)
        if not kvs:
            return f"分类 '{category}' 不存在或为空"
        lines = [f"[{category}]"]
        for k, e in kvs.items():
            lines.append(f"  {k}: {e.value}")
        return "\n".join(lines)

    def format_for_prompt(self, max_chars: int = 4000) -> str:
        entries = self.all_entries()
        if not entries:
            return ""

        by_type: dict[str, list[tuple[str, str, MemoryEntry]]] = {}
        for cat, key, entry in entries:
            by_type.setdefault(entry.type, []).append((cat, key, entry))

        lines: list[str] = []
        total = 0
        for ptype in PRIORITY_ORDER:
            group = by_type.get(ptype, [])
            if not group:
                continue
            for cat, key, entry in group:
                line = f"- [{ptype}] {cat}/{key}: {entry.value}"
                if total + len(line) > max_chars:
                    lines.append(f"... (已截断，共 {len(entries)} 条记忆)")
                    return "\n".join(lines)
                lines.append(line)
                total += len(line) + 1

        return "\n".join(lines)
