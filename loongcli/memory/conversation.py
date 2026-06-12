from __future__ import annotations
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _path_to_slug(project_dir: Path) -> str:
    raw = str(project_dir.resolve())
    raw = re.sub(r'[:\\]', '-', raw)
    raw = re.sub(r'-+', '-', raw)
    return raw.strip('-')


def _projects_root() -> Path:
    return Path.home() / ".loongcli" / "projects"


def _project_sessions_dir(project_dir: Path | None = None) -> Path:
    if project_dir is None:
        project_dir = Path.cwd()
    slug = _path_to_slug(project_dir)
    return _projects_root() / slug / "sessions"


def current_project_slug(project_dir: Path | None = None) -> str:
    return _path_to_slug(project_dir or Path.cwd())


def list_all_projects(projects_root: Path | None = None) -> list[dict]:
    """扫描所有项目，返回 [{slug, session_count, last_active}]，按 last_active 倒序。

    忽略没有 sessions 子目录或会话为空的项目。
    """
    root = projects_root or _projects_root()
    if not root.exists():
        return []

    projects: list[dict] = []
    for project_dir in root.iterdir():
        sessions_dir = project_dir / "sessions"
        if not sessions_dir.is_dir():
            continue
        session_files = list(sessions_dir.glob("*.json"))
        if not session_files:
            continue
        last_mtime = max(f.stat().st_mtime for f in session_files)
        projects.append({
            "slug": project_dir.name,
            "session_count": len(session_files),
            "last_active": datetime.fromtimestamp(last_mtime, tz=timezone.utc).isoformat(),
        })

    projects.sort(key=lambda p: p["last_active"], reverse=True)
    return projects


def list_project_sessions(slug: str, limit: int = 50, projects_root: Path | None = None) -> list[dict]:
    """列出指定项目的会话 meta，按文件 mtime 倒序。损坏的 json 跳过。"""
    sessions_dir = (projects_root or _projects_root()) / slug / "sessions"
    if not sessions_dir.is_dir():
        return []

    sessions: list[dict] = []
    for f in sorted(sessions_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            meta = dict(data["meta"])
            meta["file_size"] = f.stat().st_size
            sessions.append(meta)
        except (json.JSONDecodeError, KeyError, OSError):
            continue
        if len(sessions) >= limit:
            break
    return sessions


def load_session(slug: str, session_id: str, projects_root: Path | None = None) -> dict | None:
    """按 slug + session_id 读取单个会话。调用方必须先校验两个参数的格式。"""
    path = (projects_root or _projects_root()) / slug / "sessions" / f"{session_id}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


class ConversationStore:
    """Persists full conversation history per session, isolated by project."""

    def __init__(self, base_dir: Path | None = None, project_dir: Path | None = None):
        self.base_dir = base_dir or _project_sessions_dir(project_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = uuid.uuid4().hex[:12]
        self._meta: dict[str, Any] = {
            "session_id": self.session_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "title": "",
        }

    @property
    def session_path(self) -> Path:
        return self.base_dir / f"{self.session_id}.json"

    def _session_path(self, session_id: str | None = None) -> Path:
        return self.base_dir / f"{session_id or self.session_id}.json"

    def _read_existing(self) -> dict:
        path = self._session_path()
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    def save(self, messages: list[dict]):
        if not self._meta["title"] and messages:
            for m in messages:
                if m.get("role") == "user":
                    text = m.get("content", "")
                    self._meta["title"] = text.replace("\n", " ").strip()[:60]
                    break

        self._meta["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._meta["turn_count"] = sum(1 for m in messages if m.get("role") == "user")

        # 保留归档段等历史字段——messages 字段只代表「当前工作上下文」，
        # 完整历史 = archived_segments + messages（见 archive_segment / full_history）
        data = self._read_existing()
        data["meta"] = self._meta
        data["messages"] = messages
        self._session_path().write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def archive_segment(self, messages: list[dict], reason: str = "compact"):
        """在压缩/清空等「就地改写 messages」的操作前归档原始消息。

        compact 会用摘要替换 agent.messages，随后的 save() 会把压缩版覆写进
        messages 字段——不先归档的话，完整历史在磁盘上也会丢失。
        """
        if not messages:
            return
        data = self._read_existing()
        data.setdefault("meta", self._meta)
        data.setdefault("archived_segments", []).append({
            "archived_at": datetime.now(timezone.utc).isoformat(),
            "reason": reason,
            "message_count": len(messages),
            "messages": messages,
        })
        data.setdefault("messages", messages)
        self._session_path().write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def full_history(self, session_id: str | None = None) -> list[dict]:
        """完整历史 = 所有归档段按序拼接 + 当前 messages。

        相邻段之间存在重叠（压缩保留的最近几轮 + 摘要前缀）；
        检索场景下重叠无害，展示场景由调用方标记边界。
        """
        data = self.load(session_id or self.session_id)
        if not data:
            return []
        history: list[dict] = []
        for seg in data.get("archived_segments", []):
            history.extend(seg.get("messages", []))
        history.extend(data.get("messages", []))
        return history

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

    def save_compact(
        self,
        compact_messages: list[dict],
        structured_state: dict | None = None,
    ):
        path = self._session_path()
        if not path.exists():
            return
        data = json.loads(path.read_text(encoding="utf-8"))
        data["compact_messages"] = compact_messages
        if structured_state is not None:
            data["structured_state"] = structured_state
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

    def resume_structured(self, session_id: str) -> dict | None:
        """Load structured state for smart resume. Returns the dict or None."""
        data = self.load(session_id)
        if not data:
            return None
        structured = data.get("structured_state")
        if not structured:
            return None
        self.session_id = session_id
        self._meta = data["meta"]
        return structured
