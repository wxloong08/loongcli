from __future__ import annotations

from typing import TYPE_CHECKING

from loongcli.tools.base import Tool

if TYPE_CHECKING:
    from loongcli.core.agent import AgentLoop

ENTER_PLAN_DESCRIPTION = """\
进入规划模式：只保留只读工具，先调研再规划，计划经用户审批后再执行。

非琐碎任务时主动调用此工具。以下情况应进入规划模式：
1. 新功能实现 — 需要确定方案
2. 多种可行方案 — 需要用户选择
3. 架构决策 — 需要用户确认方向
4. 多文件变更 — 影响范围大
5. 需求不明确 — 需要先调研

以下情况不需要：单文件简单修改、用户明确要求立即执行、纯信息查询。\
"""


class EnterPlanModeTool(Tool):
    name = "enter_plan_mode"
    description = ENTER_PLAN_DESCRIPTION
    parameters = {
        "type": "object",
        "properties": {},
    }

    def __init__(self):
        self._agent: AgentLoop | None = None

    def bind_agent(self, agent: AgentLoop):
        self._agent = agent

    async def execute(self) -> str:
        if not self._agent:
            return "错误：工具未绑定到 Agent"

        if self._agent.plan_mode:
            return "已经处于规划模式中。请继续调研和规划。"

        if self._agent.role.value != "main":
            return "错误：SubAgent 不能进入规划模式（需要用户交互）"

        self._agent.enter_plan_mode()
        self._agent._rebuild_system_prompt()

        return (
            "已进入规划模式。写操作已禁用，只能使用只读工具调研。\n\n"
            "工作流程：\n"
            "1. 用 read_file / glob / grep 调研代码\n"
            "2. 用 plan 工具创建结构化计划\n"
            "3. 调用 exit_plan_mode 提交计划等待用户审批"
        )
