"""Web API 逻辑层。

所有 handler 返回 (status_code, json_dict)，不依赖 HTTP 层，便于直接单测。
会话端点只有读取方法——只读由此层与路由层共同保证。
"""
from __future__ import annotations

import re
from pathlib import Path

from loongcli.memory.conversation import (
    current_project_slug,
    list_all_projects,
    list_project_sessions,
    load_session,
)
from loongcli.memory.markdown_store import MEMORY_TYPES, MarkdownMemoryStore

_SESSION_ID_RE = re.compile(r"^[0-9a-f]{12}$")
_SLUG_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_MEMORY_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")

_MAX_DESCRIPTION_LEN = 500
_MAX_CONTENT_LEN = 100_000


def _err(status: int, message: str) -> tuple[int, dict]:
    return status, {"error": message}


class WebAPI:
    def __init__(
        self,
        memory_store: MarkdownMemoryStore | None = None,
        projects_root: Path | None = None,
        current_dir: Path | None = None,
    ):
        self.memory = memory_store or MarkdownMemoryStore()
        self.projects_root = projects_root
        self.current_dir = current_dir

    # ── 会话（只读） ──────────────────────────────────────────

    def get_projects(self) -> tuple[int, dict]:
        projects = list_all_projects(self.projects_root)
        return 200, {
            "projects": projects,
            "current": current_project_slug(self.current_dir),
        }

    def _check_slug(self, slug: str) -> tuple[int, dict] | None:
        if not _SLUG_RE.match(slug) or ".." in slug:
            return _err(400, "非法的项目标识")
        return None

    def get_sessions(self, slug: str, limit: int = 50) -> tuple[int, dict]:
        if bad := self._check_slug(slug):
            return bad
        limit = max(1, min(limit, 200))
        sessions = list_project_sessions(slug, limit=limit, projects_root=self.projects_root)
        return 200, {"sessions": sessions}

    def get_session(self, slug: str, session_id: str) -> tuple[int, dict]:
        if bad := self._check_slug(slug):
            return bad
        if not _SESSION_ID_RE.match(session_id):
            return _err(400, "非法的会话 ID")
        data = load_session(slug, session_id, projects_root=self.projects_root)
        if data is None:
            return _err(404, "会话不存在")
        return 200, data

    # ── 记忆（CRUD） ──────────────────────────────────────────

    def _check_memory_name(self, name: str) -> tuple[int, dict] | None:
        if not _MEMORY_NAME_RE.match(name):
            return _err(400, "非法的记忆名称")
        return None

    def list_memories(self, type_filter: str | None = None) -> tuple[int, dict]:
        if type_filter is not None and type_filter not in MEMORY_TYPES:
            return _err(400, f"type 必须是 {'/'.join(MEMORY_TYPES)} 之一")
        return 200, {"memories": self.memory.list_all(type_filter)}

    def get_memory(self, name: str) -> tuple[int, dict]:
        if bad := self._check_memory_name(name):
            return bad
        entry = self.memory.load(name)
        if entry is None:
            return _err(404, "记忆不存在")
        return 200, entry

    def _validate_memory_body(self, body: dict) -> tuple[int, dict] | None:
        if not isinstance(body, dict):
            return _err(400, "请求体必须是 JSON 对象")
        description = body.get("description", "")
        content = body.get("content", "")
        if not isinstance(description, str) or not description.strip():
            return _err(400, "description 不能为空")
        if not isinstance(content, str) or not content.strip():
            return _err(400, "content 不能为空")
        if len(description) > _MAX_DESCRIPTION_LEN:
            return _err(400, f"description 超过 {_MAX_DESCRIPTION_LEN} 字符上限")
        if len(content) > _MAX_CONTENT_LEN:
            return _err(400, f"content 超过 {_MAX_CONTENT_LEN} 字符上限")
        mem_type = body.get("type", "project")
        if mem_type not in MEMORY_TYPES:
            return _err(400, f"type 必须是 {'/'.join(MEMORY_TYPES)} 之一")
        return None

    def create_memory(self, body: dict) -> tuple[int, dict]:
        if bad := self._validate_memory_body(body):
            return bad
        name = body.get("name", "")
        if not isinstance(name, str) or (bad := self._check_memory_name(name)):
            return _err(400, "非法的记忆名称")
        # save() 的去重可能合并到已有条目，必须以返回值为准
        saved_name = self.memory.save(
            name=name,
            description=body["description"].strip(),
            type=body.get("type", "project"),
            content=body["content"],
        )
        return 201, {"name": saved_name, "merged": saved_name != name}

    def update_memory(self, name: str, body: dict) -> tuple[int, dict]:
        if bad := self._check_memory_name(name):
            return bad
        if self.memory.load(name) is None:
            return _err(404, "记忆不存在")
        if bad := self._validate_memory_body(body):
            return bad
        saved_name = self.memory.save(
            name=name,
            description=body["description"].strip(),
            type=body.get("type", "project"),
            content=body["content"],
        )
        return 200, {"name": saved_name, "merged": saved_name != name}

    def delete_memory(self, name: str) -> tuple[int, dict]:
        if bad := self._check_memory_name(name):
            return bad
        if not self.memory.delete(name):
            return _err(404, "记忆不存在")
        return 200, {"deleted": name}
