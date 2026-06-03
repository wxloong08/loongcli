from __future__ import annotations

from loongcli.tools.base import Tool
from loongcli.memory.markdown_store import MarkdownMemoryStore, MEMORY_TYPES


class RecallTool(Tool):
    name = "recall"
    description = (
        "Retrieve saved memories. "
        "No args: list all memories with descriptions. "
        "name: get full content of a specific memory. "
        "type: filter by memory type."
    )
    parameters = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Name of a specific memory to retrieve",
            },
            "type": {
                "type": "string",
                "enum": list(MEMORY_TYPES),
                "description": "Filter by memory type",
            },
        },
        "required": [],
    }

    def __init__(self, memory: MarkdownMemoryStore):
        self.memory = memory

    async def execute(
        self,
        name: str | None = None,
        type: str | None = None,
    ) -> str:
        if name:
            mem = self.memory.load(name)
            if mem is None:
                return f"未找到记忆: {name}"
            return (
                f"[{mem['type']}] {mem['name']}\n"
                f"描述: {mem['description']}\n"
                f"创建: {mem['created_at']}\n"
                f"更新: {mem['updated_at']}\n"
                f"---\n{mem['content']}"
            )

        entries = self.memory.list_all(type_filter=type)
        if not entries:
            return "（暂无记忆）" if not type else f"没有 {type} 类型的记忆"

        lines = []
        for e in entries:
            lines.append(f"- [{e['type']}] {e['name']} — {e['description']}")
        return "\n".join(lines)
