from __future__ import annotations
from loongcli.tools.base import Tool
from loongcli.memory.store import MemoryStore


class RecallTool(Tool):
    name = "recall"
    description = (
        "Retrieve saved memories. "
        "No args: list all categories with entry counts. "
        "category only: show all entries in that category. "
        "category + key: get a specific entry with metadata."
    )
    parameters = {
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "description": "Memory category to query",
            },
            "key": {
                "type": "string",
                "description": "Specific key within the category",
            },
        },
        "required": [],
    }

    def __init__(self, memory: MemoryStore):
        self.memory = memory

    async def execute(self, category: str | None = None, key: str | None = None) -> str:
        if category and key:
            entry = self.memory.get_entry(category, key)
            if entry is None:
                return f"未找到记忆: [{category}] {key}"
            return (
                f"[{category}] {key}\n"
                f"  值: {entry.value}\n"
                f"  类型: {entry.type}\n"
                f"  创建: {entry.created_at}\n"
                f"  更新: {entry.updated_at}"
            )

        if category:
            return self.memory.format_category(category)

        return self.memory.format_all()
