from __future__ import annotations
from typing import TYPE_CHECKING

from loongcli.tools.base import Tool, ToolRegistry
from loongcli.tools.routing import AgentRole, filter_tools
from loongcli.core.task import TaskManager, TaskStatus as TS
from loongcli.core.agent import AgentLoop
from loongcli.core.compact import Compactor
from loongcli.security.permissions import PermissionChecker, PermissionMode

if TYPE_CHECKING:
    from loongcli.core.llm import LLMClient

SUBAGENT_SYSTEM_PROMPT = """\
你是一个 SubAgent，负责执行主 Agent 分配给你的具体任务。
专注于完成任务，输出简洁的结果摘要。不要问用户问题。\
"""


class AgentTool(Tool):
    name = "delegate"
    description = (
        "Delegate a task to a SubAgent that runs independently in the background. "
        "Returns immediately with a task_id. Use task_status to check results."
    )
    parameters = {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "Task description for the SubAgent",
            },
            "tools": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional: tool names the SubAgent can use. Omit for all available tools.",
            },
        },
        "required": ["prompt"],
    }

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

    async def execute(self, prompt: str, tools: list[str] | None = None) -> str:
        if self._depth + 1 >= TaskManager.MAX_DEPTH:
            return "错误：已达到 SubAgent 最大嵌套深度，无法再派发"

        sub_registry = self._build_sub_registry(tools)

        compactor = Compactor(llm=self._sub_llm, threshold=16000)

        sub_checker = PermissionChecker(PermissionMode.SKIP)

        sub_agent = AgentLoop(
            llm=self._sub_llm,
            tool_registry=sub_registry,
            permission_checker=sub_checker,
            system_prompt=SUBAGENT_SYSTEM_PROMPT,
            max_iterations=30,
            compactor=compactor,
            role=AgentRole.SUBAGENT,
        )

        task = await self._task_manager.create_and_run(
            prompt=prompt,
            agent_loop=sub_agent,
            depth=self._depth + 1,
        )

        return f"任务已创建并在后台运行。task_id: {task.id}"

    def _build_sub_registry(self, allowed_tools: list[str] | None) -> ToolRegistry:
        visible = filter_tools(self._parent_registry._tools, AgentRole.SUBAGENT)

        sub_registry = ToolRegistry()
        for name, tool in visible.items():
            if allowed_tools and name not in allowed_tools:
                continue
            sub_registry.register(tool)

        return sub_registry
