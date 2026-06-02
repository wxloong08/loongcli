from __future__ import annotations

from loongcli.tools.base import Tool
from loongcli.core.task import TaskManager


class SendMessageTool(Tool):
    name = "send_message"
    description = (
        "Send a message to a running or completed SubAgent. "
        "If the SubAgent has completed, it will be woken up to continue."
    )
    parameters = {
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": "The task ID of the SubAgent",
            },
            "message": {
                "type": "string",
                "description": "Message to send to the SubAgent",
            },
        },
        "required": ["task_id", "message"],
    }

    def __init__(self, task_manager: TaskManager):
        self._task_manager = task_manager

    async def execute(self, task_id: str, message: str) -> str:
        return self._task_manager.send_message(task_id, message)
