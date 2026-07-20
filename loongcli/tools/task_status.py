from __future__ import annotations
import json

from loongcli.tools.base import Tool
from loongcli.core.task import (
    SINGLE_TASK_RESULT_CAP,
    TaskManager,
    cap_task_result,
    trace_path_of,
)


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
        # 自管预算（ToolResultManager 对本工具豁免）：入口一刀切会让重调返回同样的
        # 截断结果，"重新调用工具"的提示在这里是死循环——截断须附 trace 指针。
        result = info.get("result")
        if isinstance(result, str) and result:
            info["result"] = cap_task_result(
                result, SINGLE_TASK_RESULT_CAP, trace_path_of(self._task_manager.get(task_id))
            )
        return json.dumps(info, ensure_ascii=False, indent=2)
