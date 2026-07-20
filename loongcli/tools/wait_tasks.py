from __future__ import annotations

import json
from loongcli.tools.base import Tool
from loongcli.core.task import (
    BATCH_RESULT_BUDGET,
    MIN_TASK_RESULT_CAP,
    TaskManager,
    cap_task_result,
    trace_path_of,
)


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

        data = await self._task_manager.wait(task_ids, timeout=timeout)
        # 自管预算（ToolResultManager 对本工具豁免）：与 batch_delegate 同口径，
        # 总额按任务数均摊、截断附 trace 指针。
        entries = data.get("results", [])
        per_cap = max(MIN_TASK_RESULT_CAP, BATCH_RESULT_BUDGET // max(len(entries), 1))
        for entry in entries:
            result = entry.get("result")
            if isinstance(result, str) and result:
                task = self._task_manager.get(entry.get("task_id", ""))
                entry["result"] = cap_task_result(result, per_cap, trace_path_of(task))
        return json.dumps(data, ensure_ascii=False, indent=2)
