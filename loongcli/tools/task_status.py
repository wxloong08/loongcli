from __future__ import annotations
import json

from loongcli.tools.base import Tool
from loongcli.core.task import TaskManager


class TaskStatusTool(Tool):
    name = "task_status"
    description = (
        "Check the status and result of a SubAgent task. "
        "Returns task_id, status (running/completed/failed), and result."
    )
    parameters = {
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": "The task ID to check",
            },
        },
        "required": ["task_id"],
    }

    def __init__(self, task_manager: TaskManager):
        self._task_manager = task_manager

    async def execute(self, task_id: str) -> str:
        info = self._task_manager.get_status(task_id)
        return json.dumps(info, ensure_ascii=False, indent=2)
