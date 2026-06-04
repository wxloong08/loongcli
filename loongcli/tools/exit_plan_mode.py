from __future__ import annotations

import json
from typing import TYPE_CHECKING

from loongcli.tools.base import Tool

if TYPE_CHECKING:
    from loongcli.core.agent import AgentLoop
    from loongcli.plan.store import PlanStore


class ExitPlanModeTool(Tool):
    name = "exit_plan_mode"
    description = (
        "提交计划等待用户审批。必须先用 plan 工具创建计划，再调用此工具。"
        "用户可以批准、修改或取消计划。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "plan_id": {
                "type": "string",
                "description": "要提交审批的计划 ID（plan 工具创建时返回的 ID）",
            },
        },
        "required": ["plan_id"],
    }

    def __init__(self):
        self._agent: AgentLoop | None = None
        self._plan_store: PlanStore | None = None

    def bind_agent(self, agent: AgentLoop):
        self._agent = agent

    def bind_plan_store(self, plan_store: PlanStore):
        self._plan_store = plan_store

    async def execute(self, plan_id: str) -> str:
        if not self._agent:
            return "错误：工具未绑定到 Agent"

        if not self._agent.plan_mode:
            return "错误：当前不在规划模式中。无需调用此工具。"

        if not self._plan_store:
            return "错误：PlanStore 未配置"

        plan = self._plan_store.load(plan_id)
        if not plan:
            return f"错误：未找到计划 {plan_id}"

        if not plan.steps:
            return "错误：计划没有步骤。请先用 plan 工具添加步骤。"

        summary = plan.format_summary()
        return json.dumps({
            "__plan_approval__": True,
            "plan_id": plan_id,
            "plan_summary": summary,
        }, ensure_ascii=False)
