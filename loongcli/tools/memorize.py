from __future__ import annotations

from loongcli.tools.base import Tool
from loongcli.memory.markdown_store import MarkdownMemoryStore, MEMORY_TYPES


class MemorizeTool(Tool):
    name = "memorize"
    description = (
        "Save, update, or delete a persistent memory. "
        "Memories are stored as individual files and persist across sessions. "
        "Always recall first to check if a similar memory exists before saving."
    )
    parameters = {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["save", "delete"],
                "description": "save: create/update a memory. delete: remove it.",
            },
            "name": {
                "type": "string",
                "description": "Short kebab-case identifier (e.g. 'user-role', 'no-comments')",
            },
            "description": {
                "type": "string",
                "description": "One-line summary used for relevance matching in future sessions",
            },
            "type": {
                "type": "string",
                "enum": list(MEMORY_TYPES),
                "description": "Memory type: user/feedback/project/reference",
            },
            "content": {
                "type": "string",
                "description": "Full memory content (markdown supported)",
            },
        },
        "required": ["operation", "name"],
    }

    def __init__(self, memory: MarkdownMemoryStore, session_provider=None):
        self.memory = memory
        # 溯源：与 auto-extract 同一套 source_session 机制，模型主动写入同样要留案底
        self._session_provider = session_provider

    def _source(self) -> str:
        sid = ""
        if self._session_provider:
            try:
                sid = str(self._session_provider() or "")
            except Exception:
                sid = ""
        return f"memorize:{sid}" if sid else "memorize"

    async def execute(
        self,
        operation: str,
        name: str,
        description: str | None = None,
        type: str = "project",
        content: str | None = None,
    ) -> str:
        if operation == "save":
            if not content:
                return "错误：save 操作需要 content 参数"
            if not description:
                return "错误：save 操作需要 description 参数"
            saved_name = self.memory.save(
                name=name, description=description,
                type=type, content=content,
                source=self._source(),
            )
            return f"已保存 [{type}] {saved_name}"

        if operation == "delete":
            if self.memory.delete(name):
                return f"已删除 {name}"
            return f"未找到 {name}"

        return f"未知操作: {operation}"
