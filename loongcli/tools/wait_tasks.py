from __future__ import annotations

import json
from loongcli.tools.base import Tool
from loongcli.core.task import TaskManager


class WaitTasksTool(Tool):
    name = "wait_tasks"
    description = (
        "Wait for one or more background tasks to complete. "
        "Returns all results as JSON. Tasks that exceed timeout are NOT cancelled."
    )
    parameters = {
        "type": "object",
        "properties": {
            "task_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Task IDs to wait for",
            },
            "timeout": {
                "type": "number",
                "description": "Max seconds to wait (default 300)",
            },
        },
        "required": ["task_ids"],
    }

    def __init__(self, task_manager: TaskManager):
        self._task_manager = task_manager

    async def execute(self, task_ids: list[str], timeout: float = 300) -> str:
        if not task_ids:
            return json.dumps({"error": "task_ids 为空"}, ensure_ascii=False)

        result = await self._task_manager.wait(task_ids, timeout=timeout)
        return json.dumps(result, ensure_ascii=False, indent=2)
