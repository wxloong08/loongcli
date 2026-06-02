from __future__ import annotations

from loongcli.tools.base import Tool
from loongcli.memory.store import MemoryStore, MEMORY_TYPES


class MemorizeTool(Tool):
    name = "memorize"
    description = (
        "Save, update, or delete a memory entry for future sessions. "
        "Use this when you learn something worth remembering: "
        "user preferences, corrections, project decisions, or resource locations. "
        "Always recall first to check if the memory already exists before saving."
    )
    parameters = {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["save", "delete"],
                "description": "save: create or update a memory. delete: remove a memory.",
            },
            "category": {
                "type": "string",
                "description": "Category/namespace for the memory (e.g. 'preferences', 'project-decisions')",
            },
            "key": {
                "type": "string",
                "description": "Unique key within the category",
            },
            "value": {
                "type": "string",
                "description": "The information to remember (required for save)",
            },
            "type": {
                "type": "string",
                "enum": list(MEMORY_TYPES),
                "description": "Memory type: user (preferences/role), feedback (corrections), project (decisions/context), reference (external resources)",
            },
        },
        "required": ["operation", "category", "key"],
    }

    def __init__(self, memory: MemoryStore):
        self.memory = memory

    async def execute(
        self,
        operation: str,
        category: str,
        key: str,
        value: str | None = None,
        type: str = "project",
    ) -> str:
        if operation == "save":
            if not value:
                return "错误：save 操作需要 value 参数"
            self.memory.set(category, key, value, type=type)
            existing = self.memory.get_entry(category, key)
            return f"已保存 [{type}] {category}/{key}"

        if operation == "delete":
            if self.memory.delete(category, key):
                return f"已删除 {category}/{key}"
            return f"未找到 {category}/{key}"

        return f"未知操作: {operation}"
