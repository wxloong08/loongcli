from __future__ import annotations

from loongcli.tools.base import Tool
from loongcli.core.task import TaskManager


class StopTaskTool(Tool):
    name = "stop_task"
    description = "Stop a running background task to save tokens."
    parameters = {
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": "Task ID to stop",
            },
            "reason": {
                "type": "string",
                "description": "Why this task is being stopped",
            },
        },
        "required": ["task_id"],
    }

    def __init__(self, task_manager: TaskManager):
        self._task_manager = task_manager

    async def execute(self, task_id: str, reason: str = "") -> str:
        return self._task_manager.stop(task_id, reason or "手动停止")
