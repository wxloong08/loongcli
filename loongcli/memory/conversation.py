from __future__ import annotations
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ConversationStore:
    """Persists full conversation history per session."""

    def __init__(self, base_dir: Path | None = None):
        self.base_dir = base_dir or Path.home() / ".loongcli" / "sessions"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = uuid.uuid4().hex[:12]
        self._meta: dict[str, Any] = {
            "session_id": self.session_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "title": "",
        }

    def _session_path(self, session_id: str | None = None) -> Path:
        return self.base_dir / f"{session_id or self.session_id}.json"

    def save(self, messages: list[dict]):
        if not self._meta["title"] and messages:
            for m in messages:
                if m.get("role") == "user":
                    text = m.get("content", "")
                    self._meta["title"] = text.replace("\n", " ").strip()[:60]
                    break

        self._meta["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._meta["turn_count"] = sum(1 for m in messages if m.get("role") == "user")

        data = {"meta": self._meta, "messages": messages}
        self._session_path().write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load(self, session_id: str) -> dict | None:
        path = self._session_path(session_id)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def list_sessions(self, limit: int = 20) -> list[dict]:
        sessions: list[dict] = []
        for f in sorted(self.base_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                meta = data["meta"]
                meta["file_size"] = f.stat().st_size
                sessions.append(meta)
            except (json.JSONDecodeError, KeyError):
                continue
            if len(sessions) >= limit:
                break
        return sessions

    def save_compact(self, compact_messages: list[dict]):
        path = self._session_path()
        if not path.exists():
            return
        data = json.loads(path.read_text(encoding="utf-8"))
        data["compact_messages"] = compact_messages
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def resume(self, session_id: str) -> list[dict] | None:
        data = self.load(session_id)
        if not data:
            return None
        self.session_id = session_id
        self._meta = data["meta"]
        return data.get("compact_messages") or data.get("messages", [])
