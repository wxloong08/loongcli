from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

from loongcli.tools.base import Tool, ToolRegistry
from loongcli.tools.routing import AgentRole, filter_tools, COORDINATOR_ALLOWED
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

COORDINATOR_SYSTEM_PROMPT = """\
你是一个协调者(Coordinator)。你的职责是理解目标、拆分任务、指挥 worker 执行、合成结果。

## 核心原则

1. **你不直接执行**：不写文件、不改代码、不跑命令。这些交给 worker。
2. **你必须理解**：收到 worker 的调研结果后，自己读懂、思考、合成，写出具体的实现规格再派 worker 去做。绝不转发。
3. **并行是你的超能力**：能并行的任务同时派出，不要串行等待。

## 四阶段流水线

### 阶段 1：调研（Workers 并行）
- 拆分为独立的调研子任务，同时派多个 worker
- 每个 worker 的 prompt 要具体：指明要看哪些文件、要回答什么问题

### 阶段 2：合成（你亲自做）
- 等调研 worker 完成（wait_tasks）
- 自己阅读所有结果，理解问题全貌
- 写出具体的实现规格（改哪些文件、怎么改、接口契约）
- 这一步不能委派！

### 阶段 3：实现（Workers）
- 按合成的规格派 worker 执行具体修改
- 每个 worker 的 prompt 包含完整规格，不依赖"之前的发现"

### 阶段 4：验证（新 Worker）
- 派一个新的 worker 做验证，不让实现者自验
- 验证 worker 跑测试、检查改动是否正确

## Continue vs Spawn 决策

- 后续任务与 worker 现有上下文高度相关 → send_message 续命，省 token
- 无关任务 / worker 跑偏 → 派新 worker
- 验证 → 永远派新 worker（需要新鲜视角）
- worker 明显跑偏 → stop_task 立即停止，节省 token

## 输出

完成所有阶段后，合成最终结果。包含：做了什么改动（概要）、验证结果、需要注意的事项。\
"""

AUTO_TIMEOUT = 30
SYNC_TIMEOUT = 300


class AgentTool(Tool):
    name = "delegate"
    description = (
        "Delegate a task to a SubAgent. Modes: auto (wait 30s then background), "
        "background (immediate), sync (block until done). "
        "Set coordinator=true for complex multi-step tasks requiring parallel workers."
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
            "mode": {
                "type": "string",
                "enum": ["auto", "background", "sync"],
                "description": "auto: wait 30s then background; background: immediate; sync: block until done",
            },
            "timeout": {
                "type": "number",
                "description": "Seconds for sync/auto modes (default 30 for auto, 300 for sync)",
            },
            "coordinator": {
                "type": "boolean",
                "description": "If true, spawn a Coordinator that manages parallel workers",
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

    async def execute(
        self,
        prompt: str,
        tools: list[str] | None = None,
        mode: str = "auto",
        timeout: float | None = None,
        coordinator: bool = False,
    ) -> str:
        if self._depth + 1 >= TaskManager.MAX_DEPTH:
            return "错误：已达到 SubAgent 最大嵌套深度，无法再派发"

        if coordinator:
            sub_registry = self._build_coordinator_registry()
            system_prompt = COORDINATOR_SYSTEM_PROMPT
            role = AgentRole.COORDINATOR
            max_iter = 80
        else:
            sub_registry = self._build_sub_registry(tools)
            system_prompt = SUBAGENT_SYSTEM_PROMPT
            role = AgentRole.SUBAGENT
            max_iter = 30

        compactor = Compactor(llm=self._sub_llm, threshold=16000)
        sub_checker = PermissionChecker(PermissionMode.SKIP)

        sub_agent = AgentLoop(
            llm=self._sub_llm,
            tool_registry=sub_registry,
            permission_checker=sub_checker,
            system_prompt=system_prompt,
            max_iterations=max_iter,
            compactor=compactor,
            task_manager=self._task_manager if coordinator else None,
            role=role,
        )

        task = await self._task_manager.create_and_run(
            prompt=prompt,
            agent_loop=sub_agent,
            depth=self._depth + 1,
        )

        if mode == "background":
            return f"任务已创建并在后台运行。task_id: {task.id}"

        wait_seconds = timeout or (AUTO_TIMEOUT if mode == "auto" else SYNC_TIMEOUT)

        try:
            await asyncio.wait_for(
                asyncio.shield(task._async_task),
                timeout=wait_seconds,
            )
        except asyncio.TimeoutError:
            if mode == "sync":
                return json.dumps({
                    "status": "timeout",
                    "task_id": task.id,
                    "message": f"任务未在 {wait_seconds}s 内完成",
                }, ensure_ascii=False)
            # auto: convert to background
            return (
                f"任务仍在运行，已自动转后台。task_id: {task.id}\n"
                f"用 wait_tasks 或 task_status 获取结果。"
            )

        return json.dumps({
            "status": task.status.value,
            "task_id": task.id,
            "result": task.result,
        }, ensure_ascii=False)

    def _build_sub_registry(self, allowed_tools: list[str] | None) -> ToolRegistry:
        visible = filter_tools(self._parent_registry._tools, AgentRole.SUBAGENT)
        sub_registry = ToolRegistry()
        for name, tool in visible.items():
            if allowed_tools and name not in allowed_tools:
                continue
            sub_registry.register(tool)
        return sub_registry

    def _build_coordinator_registry(self) -> ToolRegistry:
        visible = filter_tools(self._parent_registry._tools, AgentRole.COORDINATOR)
        sub_registry = ToolRegistry()
        for name, tool in visible.items():
            sub_registry.register(tool)
        return sub_registry
