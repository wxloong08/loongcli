from __future__ import annotations
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loongcli.core.messages import message_text


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


def input_history_path(project_dir: Path | None = None) -> Path:
    """项目级输入历史文件（prompt_toolkit FileHistory 用）。

    与 sessions 平级、按项目隔离——新会话的方向键历史来自这里，跨会话持久。"""
    if project_dir is None:
        project_dir = Path.cwd()
    return _projects_root() / _path_to_slug(project_dir) / "input_history"


def project_tasks_dir(project_dir: Path | None = None) -> Path:
    """子任务 trace 的落盘目录——与 sessions 平级隔离，
    避免污染会话列表（web UI / --continue 只扫 sessions）。"""
    if project_dir is None:
        project_dir = Path.cwd()
    slug = _path_to_slug(project_dir)
    return _projects_root() / slug / "tasks"


def project_memory_dir(project_dir: Path | None = None) -> Path:
    """项目级记忆库目录——与 sessions 平级，按项目隔离。

    project/feedback/reference 类型记忆经 MemoryRouter 落这里；
    全局库 ~/.loongcli/memory 只留跨项目的 user 记忆。"""
    if project_dir is None:
        project_dir = Path.cwd()
    return _projects_root() / _path_to_slug(project_dir) / "memory"


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

    def __init__(
        self,
        base_dir: Path | None = None,
        project_dir: Path | None = None,
        session_id: str | None = None,
        extra_meta: dict[str, Any] | None = None,
    ):
        self.base_dir = base_dir or _project_sessions_dir(project_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        # session_id 可注入：子任务 trace 以 task_id 为文件名，与 TaskManager 对齐
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self._meta: dict[str, Any] = {
            "session_id": self.session_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "title": "",
        }
        if extra_meta:
            self._meta.update(extra_meta)

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

    def _atomic_write(self, path: Path, text: str) -> None:
        """原子落盘：写同目录临时文件后 os.replace 替换。会话文件是单文件全量重写
        且装着整段历史（当前 messages + 所有归档段），直接 write_text 若写到一半被杀
        会截断毁掉全部不可再生历史；rename 是原子操作，崩溃只会留下无害的 .tmp。"""
        tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.tmp")
        try:
            tmp.write_text(text, encoding="utf-8")
            os.replace(tmp, path)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise

    def save(self, messages: list[dict]):
        if not self._meta["title"] and messages:
            for m in messages:
                if m.get("role") == "user":
                    # content 可能是多模态 list，经 message_text 取纯文本作标题
                    text = message_text(m)
                    self._meta["title"] = text.replace("\n", " ").strip()[:60]
                    break

        self._meta["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._meta["turn_count"] = sum(1 for m in messages if m.get("role") == "user")

        # 保留归档段等历史字段——messages 字段只代表「当前工作上下文」，
        # 完整历史 = archived_segments + messages（见 archive_segment / full_history）
        data = self._read_existing()
        data["meta"] = self._meta
        data["messages"] = messages
        self._atomic_write(
            self._session_path(),
            json.dumps(data, ensure_ascii=False, indent=2),
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
        self._atomic_write(
            self._session_path(),
            json.dumps(data, ensure_ascii=False, indent=2),
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
        # 损坏的会话文件（如上次非原子写崩溃的残留、外部误改）不应让 resume/--continue
        # 硬崩——优雅降级为 None，调用方回退启动新会话。与 _read_existing/list_* 的容错一致。
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

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
        # 胶囊时间戳：resume 时与 meta.updated_at 比新鲜度（见 _capsule_stale）
        data["capsule_saved_at"] = datetime.now(timezone.utc).isoformat()
        if structured_state is not None:
            data["structured_state"] = structured_state
        self._atomic_write(path, json.dumps(data, ensure_ascii=False, indent=2))

    @staticmethod
    def _capsule_stale(data: dict) -> bool:
        """胶囊比 raw messages 旧则过期：干净退出→continue 干活→崩溃退出后，
        messages 已随每轮落盘更新而胶囊还是上次干净退出的产物——按胶囊恢复
        等于上下文时间旅行回旧状态，必须降级走 raw。降级链的优先序按保真类型
        （structured > compact > raw）之前，先按新鲜度裁一刀。
        无时间戳的旧会话文件按不过期处理（升级前的文件无从判断，维持原行为）。"""
        saved_at = data.get("capsule_saved_at")
        updated_at = (data.get("meta") or {}).get("updated_at")
        if not saved_at or not updated_at:
            return False
        # 两者都是 datetime.now(timezone.utc).isoformat() 产物，同构可字典序比较
        return updated_at > saved_at

    def _archive_for_resume(self, data: dict) -> None:
        """胶囊/压缩版恢复会重建上下文，随后第一次 save() 就把磁盘上旧段的
        raw messages 尾巴（上次会话内最后一次 compact 之后的原始轮次）覆盖掉——
        与 compact/clear 前归档同理，恢复前必须先归档，否则 full_history/
        search_history 的「细节可找回」承诺在最近一段上是破的。
        连续无新轮次的 resume 会重复归档同一内容，与最后一个归档段相同时跳过。"""
        messages = data.get("messages") or []
        if not messages:
            return
        segments = data.get("archived_segments") or []
        if segments and segments[-1].get("messages") == messages:
            return
        self.archive_segment(messages, reason="resume")

    def resume(self, session_id: str) -> list[dict] | None:
        data = self.load(session_id)
        if not data:
            return None
        self.session_id = session_id
        self._meta = data["meta"]
        compact = data.get("compact_messages")
        if compact and not self._capsule_stale(data):
            self._archive_for_resume(data)
            return compact
        return data.get("messages", [])

    def resume_structured(self, session_id: str) -> dict | None:
        """Load structured state for smart resume. Returns the dict or None."""
        data = self.load(session_id)
        if not data:
            return None
        structured = data.get("structured_state")
        if not structured:
            return None
        if self._capsule_stale(data):
            return None  # 过期胶囊：返回 None 让调用方降级走 resume() 的 raw 路径
        self.session_id = session_id
        self._meta = data["meta"]
        self._archive_for_resume(data)
        return structured
