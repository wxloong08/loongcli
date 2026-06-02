from __future__ import annotations
import json

from loongcli.tools.base import Tool
from loongcli.plan.store import PlanStore, Plan, PlanStep, STEP_STATUSES


class PlanTool(Tool):
    name = "plan"
    description = (
        "Manage persistent work plans that survive across sessions. "
        "Use for multi-step tasks: create a plan, update step status as you work, "
        "mark completed when done. Active plans are shown at session start."
    )
    parameters = {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["create", "update_step", "get", "complete", "abandon", "list"],
                "description": (
                    "create: new plan with title+steps. "
                    "update_step: set step status/output. "
                    "get: show plan details. "
                    "complete/abandon: close a plan. "
                    "list: show all plans."
                ),
            },
            "plan_id": {
                "type": "string",
                "description": "Plan ID (required for update_step/get/complete/abandon)",
            },
            "title": {
                "type": "string",
                "description": "Plan title (for create)",
            },
            "steps": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Step descriptions (for create)",
            },
            "step_index": {
                "type": "integer",
                "description": "Step number to update (for update_step)",
            },
            "step_status": {
                "type": "string",
                "enum": list(STEP_STATUSES),
                "description": "New step status (for update_step)",
            },
            "step_output": {
                "type": "string",
                "description": "What was produced in this step (for update_step)",
            },
        },
        "required": ["operation"],
    }

    def __init__(self, plan_store: PlanStore):
        self._store = plan_store

    async def execute(
        self,
        operation: str,
        plan_id: str | None = None,
        title: str | None = None,
        steps: list[str] | None = None,
        step_index: int | None = None,
        step_status: str | None = None,
        step_output: str | None = None,
    ) -> str:
        if operation == "create":
            return self._create(title, steps)
        if operation == "update_step":
            return self._update_step(plan_id, step_index, step_status, step_output)
        if operation == "get":
            return self._get(plan_id)
        if operation == "complete":
            return self._close(plan_id, "completed")
        if operation == "abandon":
            return self._close(plan_id, "abandoned")
        if operation == "list":
            return self._list()
        return f"未知操作: {operation}"

    def _create(self, title: str | None, step_descs: list[str] | None) -> str:
        if not title:
            return "错误：需要 title"
        if not step_descs:
            return "错误：需要 steps"

        plan_steps = [PlanStep(index=i, description=d) for i, d in enumerate(step_descs)]
        plan = Plan(title=title, steps=plan_steps)
        self._store.save(plan)
        return f"计划已创建: {plan.id}\n{plan.format_summary()}"

    def _update_step(
        self, plan_id: str | None, step_index: int | None,
        status: str | None, output: str | None,
    ) -> str:
        if not plan_id:
            return "错误：需要 plan_id"
        plan = self._store.load(plan_id)
        if not plan:
            return f"未找到计划: {plan_id}"
        if step_index is None:
            return "错误：需要 step_index"
        if step_index < 0 or step_index >= len(plan.steps):
            return f"错误：step_index {step_index} 超出范围 (0-{len(plan.steps)-1})"

        step = plan.steps[step_index]
        if status:
            step.status = status
        if output:
            step.output = output
        self._store.save(plan)

        done, total = plan.progress()
        return f"步骤 {step_index} 已更新 → {step.status}\n进度: {done}/{total}"

    def _get(self, plan_id: str | None) -> str:
        if not plan_id:
            active = self._store.get_active_plans()
            if not active:
                return "没有活跃计划"
            return "\n\n".join(p.format_summary() for p in active)
        plan = self._store.load(plan_id)
        if not plan:
            return f"未找到计划: {plan_id}"
        return plan.format_summary()

    def _close(self, plan_id: str | None, new_status: str) -> str:
        if not plan_id:
            return "错误：需要 plan_id"
        plan = self._store.load(plan_id)
        if not plan:
            return f"未找到计划: {plan_id}"
        plan.status = new_status
        self._store.save(plan)
        done, total = plan.progress()
        return f"计划 {plan_id} 已标记为 {new_status} ({done}/{total} 步骤完成)"

    def _list(self) -> str:
        plans = self._store.list_plans()
        if not plans:
            return "没有计划"
        lines = []
        for p in plans:
            done, total = p.progress()
            lines.append(f"- [{p.status}] {p.id}: {p.title} ({done}/{total})")
        return "\n".join(lines)
