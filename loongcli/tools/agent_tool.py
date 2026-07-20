from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

from loongcli.tools.base import Tool, ToolRegistry
from loongcli.tools.routing import AgentRole, filter_tools, COORDINATOR_ALLOWED
from loongcli.core.task import TaskManager, TaskStatus as TS
from loongcli.core.agent import AgentLoop, AgentServices
from loongcli.core.compact import Compactor
from loongcli.security.permissions import PermissionChecker

if TYPE_CHECKING:
    from loongcli.core.llm import LLMClient

SUBAGENT_SYSTEM_PROMPT = """\
你是一个 SubAgent，负责执行主 Agent 分配给你的具体任务。
专注于完成任务，输出简洁的结果摘要。不要问用户问题。
结果里的事实与数字以工具结果为准——查不到就写「未知/未查到」，不要用典型值或印象填补；
不确定的内容明确标注不确定，数据缺口如实报告而不是凑成完整结论。\
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

## 合成纪律

- 结论里的事实与数字必须能指认到某个 worker 的结果；worker 没查到的就是缺口，
  如实写「未知」，不要用你的印象或典型值补齐，更不要用先验推翻 worker 拿回的一手数据。
- 失败/超时的 worker = 对应信息面不完整——宁可缩小结论范围并说明缺口，也不要凑成完整结论。

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
        telemetry=None,
        hook_manager=None,
    ):
        self._task_manager = task_manager
        self._llm = llm
        self._sub_llm = sub_llm or llm
        self._parent_registry = parent_registry
        self._permission_checker = security
        self._depth = depth
        # 主会话的事件流实例（可 None）：子代理共享同一文件，事件靠 role 字段区分主/子
        self._telemetry = telemetry
        # 用户 hook 同样约束子代理的工具调用——hook 当安全闸用时子代理不能是旁路
        self._hook_manager = hook_manager
        # 委派树父任务：主 agent 的 delegate 为 None（派出的是根任务）；
        # 协调者的克隆在协调者任务创建后被回填为该任务 id（见 execute）
        self._parent_task_id: str | None = None

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

        # 继承父级的权限检查器（同 mode + 会话学习到的白名单），而非一律 SKIP：
        # 子代理绝不能绕过主 agent 所受的确认层，否则 delegate/提示注入就成了提权路径。
        # interactive=False 让任何 CONFIRM 确定性地降级为 DENY（子代理内没有人可应答）。
        sub_agent = AgentLoop(
            llm=self._sub_llm,
            tool_registry=sub_registry,
            permission_checker=self._permission_checker,
            system_prompt=system_prompt,
            max_iterations=max_iter,
            role=role,
            interactive=False,
            services=AgentServices(
                compactor=compactor,
                task_manager=self._task_manager if coordinator else None,
                telemetry=self._telemetry,
                hook_manager=self._hook_manager,
            ),
        )

        task = await self._task_manager.create_and_run(
            prompt=prompt,
            agent_loop=sub_agent,
            depth=self._depth + 1,
            parent_id=self._parent_task_id,
        )

        if coordinator:
            # 回填克隆的父任务 id，协调者派发的 worker 由此挂到它名下成链。
            # create_and_run 无内部 await，同一 tick 完成——协调者此刻尚未开跑
            clone = sub_registry._tools.get(self.name)
            if isinstance(clone, AgentTool):
                clone._parent_task_id = task.id

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
            if name == self.name:
                # depth 属于调用上下文而非共享实例：直接复用父级 delegate 的话，
                # _depth 永远停在主装配时的 0，MAX_DEPTH 闸在协调者链上不可达
                # （协调者套协调者可无限嵌套）。为协调者克隆一个携带自身层级的 delegate。
                tool = AgentTool(
                    task_manager=self._task_manager,
                    llm=self._llm,
                    parent_registry=self._parent_registry,
                    security=self._permission_checker,
                    depth=self._depth + 1,
                    sub_llm=self._sub_llm,
                    telemetry=self._telemetry,
                    hook_manager=self._hook_manager,
                )
            sub_registry.register(tool)
        return sub_registry
