from __future__ import annotations
import asyncio
import json
import logging
from typing import TYPE_CHECKING, Callable

from loongcli.tools.base import Tool, ToolRegistry
from loongcli.tools.routing import AgentRole, filter_tools
from loongcli.tools.agent_tool import SUBAGENT_SYSTEM_PROMPT
from loongcli.core.task import TaskManager, TaskStatus
from loongcli.core.agent import AgentLoop
from loongcli.core.compact import Compactor
from loongcli.core.events import BatchProgress
from loongcli.security.permissions import PermissionChecker

if TYPE_CHECKING:
    from loongcli.core.llm import LLMClient

logger = logging.getLogger(__name__)


class BatchDelegateTool(Tool):
    name = "batch_delegate"
    description = (
        "Dispatch multiple tasks to SubAgents in parallel, wait for all to complete. "
        "Use for fan-out/fan-in: parallel research, multi-file analysis, comparison tasks. "
        "Returns all results as structured JSON for synthesis."
    )
    parameters = {
        "type": "object",
        "properties": {
            "tasks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "Task description for the SubAgent",
                        },
                        "tools": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional: tool names the SubAgent can use",
                        },
                    },
                    "required": ["prompt"],
                },
                "description": "List of subtasks to run in parallel",
            },
            "timeout": {
                "type": "number",
                "description": "Max seconds to wait (default 300)",
            },
        },
        "required": ["tasks"],
    }
    supports_progress = True

    def __init__(
        self,
        task_manager: TaskManager,
        llm: LLMClient,
        parent_registry: ToolRegistry,
        security: PermissionChecker,
        depth: int = 0,
        sub_llm: LLMClient | None = None,
    ):
        self._task_manager = task_manager
        self._llm = llm
        self._sub_llm = sub_llm or llm
        self._parent_registry = parent_registry
        self._permission_checker = security
        self._depth = depth
        self._parent_task_id: str | None = None  # 委派树父任务（与 AgentTool 同语义）
        self._progress_callback: Callable[[BatchProgress], None] | None = None

    async def execute(self, tasks: list[dict], timeout: float = 300) -> str:
        if self._depth + 1 >= TaskManager.MAX_DEPTH:
            return json.dumps({"error": "已达到 SubAgent 最大嵌套深度"}, ensure_ascii=False)

        if not tasks:
            return json.dumps({"error": "任务列表为空"}, ensure_ascii=False)

        total = len(tasks)
        completed_count = 0

        task_objs: list[tuple[int, object, str]] = []
        for i, t in enumerate(tasks):
            sub_registry = self._build_sub_registry(t.get("tools"))
            compactor = Compactor(llm=self._sub_llm, threshold=16000)

            # 继承父级 checker（而非一律 SKIP），并以非交互方式运行，
            # 使 CONFIRM 自动拒绝——详见 AgentTool.execute。
            sub_agent = AgentLoop(
                llm=self._sub_llm,
                tool_registry=sub_registry,
                permission_checker=self._permission_checker,
                system_prompt=SUBAGENT_SYSTEM_PROMPT,
                max_iterations=30,
                compactor=compactor,
                role=AgentRole.SUBAGENT,
                interactive=False,
            )

            task_obj = await self._task_manager.create_and_run(
                prompt=t["prompt"],
                agent_loop=sub_agent,
                depth=self._depth + 1,
                parent_id=self._parent_task_id,
            )
            task_objs.append((i, task_obj, t["prompt"]))

            if self._progress_callback:
                def _on_done(future, *, idx=i, prompt=t["prompt"], tobj=task_obj):
                    nonlocal completed_count
                    completed_count += 1
                    self._progress_callback(BatchProgress(
                        completed=completed_count,
                        total=total,
                        task_index=idx,
                        task_prompt=prompt,
                        status=tobj.status.value,
                    ))
                task_obj._async_task.add_done_callback(_on_done)

        async_tasks = [tobj._async_task for _, tobj, _ in task_objs]
        _, pending = await asyncio.wait(async_tasks, timeout=timeout)

        timed_out_ids: set[str] = set()
        if pending:
            for at in pending:
                at.cancel()
                for _, tobj, _ in task_objs:
                    if tobj._async_task is at:
                        timed_out_ids.add(tobj.id)
            await asyncio.gather(*pending, return_exceptions=True)
            for _, tobj, _ in task_objs:
                if tobj.id in timed_out_ids:
                    tobj.status = TaskStatus.FAILED
                    tobj.result = "超时取消"

        batch_ids = {tobj.id for _, tobj, _ in task_objs}
        self._task_manager._notifications = [
            n for n in self._task_manager._notifications
            if n["task_id"] not in batch_ids
        ]

        results = []
        n_completed = 0
        n_failed = 0
        for i, tobj, prompt in task_objs:
            if tobj.status == TaskStatus.COMPLETED:
                n_completed += 1
            else:
                n_failed += 1
            results.append({
                "index": i,
                "prompt": prompt,
                "status": tobj.status.value,
                "result": tobj.result,
            })

        return json.dumps({
            "completed": n_completed,
            "failed": n_failed,
            "results": results,
        }, ensure_ascii=False, indent=2)

    def _build_sub_registry(self, allowed_tools: list[str] | None) -> ToolRegistry:
        visible = filter_tools(self._parent_registry._tools, AgentRole.SUBAGENT)
        sub_registry = ToolRegistry()
        for name, tool in visible.items():
            if allowed_tools and name not in allowed_tools:
                continue
            sub_registry.register(tool)
        return sub_registry
