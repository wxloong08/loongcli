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
            "approach": {
                "type": "string",
                "description": (
                    "方案（for create，必填，Markdown）。四块：背景与范围（做什么、不做什么）；"
                    "关键决定（每条写选了什么、放弃了什么、为什么）；风险与先验证什么；需要用户拍板的问题（单列，带你的推荐）。"
                    "步骤只写怎么执行，取舍和理由写在这里。简单任务两三行即可。"
                ),
            },
            "steps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "description": {
                            "type": "string",
                            "description": "这一步改什么、产出什么（编码任务具体到文件/函数）",
                        },
                        "criteria": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 1,
                            "maxItems": 5,
                            "description": (
                                "1-5 条完成判据，每条是执行期你自己能检查的事实（跑什么命令看到什么、"
                                "文件里有什么）。需要用户本人操作的验收：要等用户结果就单列最后一步、判据写「用户回复验收通过」，"
                                "不用等就写「验收清单已列出」；不要自己写成「实机通过」。"
                                "「验证正确性」这类空话不合格。"
                            ),
                        },
                    },
                    "required": ["description", "criteria"],
                },
                "description": "Steps (for create). 标完成时每条判据都要给证据，所以判据写成可检查的。",
            },
            "step": {
                "type": "integer",
                "description": "要更新的步骤编号，从 1 起，即计划里显示的编号（for update_step）",
            },
            "step_status": {
                "type": "string",
                "enum": list(STEP_STATUSES),
                "description": "New step status (for update_step)",
            },
            "step_output": {
                "type": "string",
                "description": "What was produced in this step (for update_step); 标 skipped 时必填跳过理由",
            },
            "evidence": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "criterion": {"type": "integer", "description": "判据编号，从 1 起，即计划里显示的编号"},
                        "text": {
                            "type": "string",
                            "description": (
                                "证明这条判据成立的证据：命令 + 关键输出、文件:行号、观察到的结果。"
                                "必须证明判据本身，不能拿相邻的事实充数"
                                "（例：判据是运行时线程归属，静态导入检查不算证据）"
                            ),
                        },
                    },
                    "required": ["criterion", "text"],
                },
                "description": (
                    "for update_step: 逐条判据的证据，可分多次提交。标 completed 时每条判据都必须已有证据，否则拒绝"
                ),
            },
        },
        "required": ["operation"],
    }

    def __init__(self, plan_store: PlanStore):
        self._store = plan_store
        self._agent = None
        self._last_created: str | None = None

    def bind_agent(self, agent) -> None:
        """获知当前活跃计划——草稿覆盖时绝不误删已批准的计划。"""
        self._agent = agent

    async def execute(
        self,
        operation: str,
        plan_id: str | None = None,
        title: str | None = None,
        steps: list | None = None,
        step: int | None = None,
        step_status: str | None = None,
        step_output: str | None = None,
        evidence: list | None = None,
        approach: str | None = None,
    ) -> str:
        if operation == "create":
            return self._create(title, steps, approach)
        if operation == "update_step":
            return self._update_step(plan_id, step, step_status, step_output, evidence)
        if operation == "get":
            return self._get(plan_id)
        if operation == "complete":
            return self._close(plan_id, "completed")
        if operation == "abandon":
            return self._close(plan_id, "abandoned")
        if operation == "list":
            return self._list()
        return f"未知操作: {operation}"

    @staticmethod
    def _parse_steps(raw: list) -> tuple[list[PlanStep], str]:
        """把 create 的 steps 参数解析成 PlanStep；不合格返回 ([], 错误说明)。先校验再动草稿，免得删了旧的、新的又建不成。"""
        parsed = []
        for i, item in enumerate(raw):
            if not isinstance(item, dict):
                return [], (
                    f"错误：步骤 {i + 1} 是纯文本。每步要写成 {{\"description\": ..., \"criteria\": [...]}}，"
                    "criteria 是 1-5 条完成判据"
                )
            desc = str(item.get("description") or "").strip()
            criteria = [str(c).strip() for c in (item.get("criteria") or []) if str(c).strip()]
            if not desc:
                return [], f"错误：步骤 {i + 1} 缺 description"
            if not 1 <= len(criteria) <= 5:
                return [], f"错误：步骤 {i + 1} 需要 1-5 条 criteria（现在 {len(criteria)} 条）"
            parsed.append(PlanStep(index=i, description=desc, criteria=criteria))
        return parsed, ""

    def _create(self, title: str | None, step_descs: list | None, approach: str | None = None) -> str:
        if not title:
            return "错误：需要 title"
        if not step_descs:
            return "错误：需要 steps"
        plan_steps, err = self._parse_steps(step_descs)
        if err:
            return err
        approach = (approach or "").strip()
        if not approach:
            return ("错误：需要 approach（方案）：背景与范围、关键决定（选了什么、放弃了什么、为什么）、"
                    "风险与先验证什么、需要用户拍板的问题。简单任务两三行即可")

        # 覆盖上一份未动工的草稿：规划阶段反复 create 属于正常迭代，但旧草稿会变成
        # 孤儿垃圾（真机：一轮连开三个 plan，两个成孤儿）。三重保护缺一不删——
        # 非当前活跃计划、状态 active、所有步骤仍 pending（动过工的绝不删）。
        replaced = ""
        prev = self._last_created
        active_id = getattr(self._agent, "_active_plan_id", None) if self._agent else None
        if prev and prev != active_id:
            old = self._store.load(prev)
            if old and old.status == "active" and all(s.status == "pending" for s in old.steps):
                self._store.delete(prev)
                replaced = f"（已覆盖未动工草稿 {prev}）"

        plan = Plan(title=title, approach=approach, steps=plan_steps)
        self._store.save(plan)
        self._last_created = plan.id
        # 方案是刚写的，不回显全文（省 token）；要看用 plan get
        return f"计划已创建: {plan.id}{replaced}（方案已保存，plan get 查看）\n{plan.format_summary()}"

    def _update_step(
        self, plan_id: str | None, number: int | None,
        status: str | None, output: str | None, evidence: list | None = None,
    ) -> str:
        """number：步骤编号，从 1 起（对外编号一律从 1 起，见 PlanStep.number）。"""
        if not plan_id:
            return "错误：需要 plan_id"
        plan = self._store.load(plan_id)
        if not plan:
            return f"未找到计划: {plan_id}"
        if number is None:
            return "错误：需要 step（步骤编号，从 1 起）"
        if not 1 <= number <= len(plan.steps):
            return f"错误：step {number} 超出范围（1-{len(plan.steps)}，从 1 起）"

        step = plan.steps[number - 1]
        # 先收证据：这次标完成即使被拒，已交的证据也留下，下次只补缺的那几条
        for item in evidence or []:
            if not isinstance(item, dict):
                return "错误：evidence 每项要写成 {\"criterion\": 判据编号, \"text\": 证据}"
            try:
                ci = int(item.get("criterion")) - 1          # 对外从 1 起
            except (TypeError, ValueError):
                ci = -1
            if not 0 <= ci < len(step.criteria):
                return (f"错误：判据编号 {item.get('criterion')} 超出范围"
                        f"（步骤 {step.number} 有 {len(step.criteria)} 条判据，编号 1-{len(step.criteria)}）")
            text = str(item.get("text") or "").strip()
            if text:
                step.evidence[ci] = text
        if output:
            step.output = output

        # 状态闸：完成要每条判据都有证据，跳过要有理由。被拒时状态不变，但证据和产出照存。
        rejected = ""
        if status == "completed" and step.missing_evidence():
            missing = "\n".join(f"- 判据 {i + 1}：{step.criteria[i]}" for i in step.missing_evidence())
            rejected = (
                f"步骤 {step.number} 未标完成：以下判据还没有证据（evidence 里按编号补上，每条要能证明判据本身）：\n{missing}"
            )
        elif status == "skipped" and not step.output.strip():
            rejected = f"步骤 {step.number} 未标跳过：需要在 step_output 写跳过理由"
        elif status:
            step.status = status
        self._store.save(plan)
        if rejected:
            return rejected

        # 回带完整计划而非只回进度计数：更新那一刻工具结果就把最新全貌送到模型眼前，
        # 而工具结果天然落在对话尾部、走前缀缓存、零额外代价。系统提示词里的活跃计划段
        # 因此不再携带 format_summary()——那段在 messages[0]，随进度改写它等于每完成
        # 一步就废掉整个前缀缓存（连同其后的全部对话历史）。信息在产生的那一刻送到，
        # 好过事后每轮补一遍。
        done, total = plan.progress()
        return (
            f"步骤 {step.number} 已更新 → {step.status}（进度 {done}/{total}）\n\n"
            f"{plan.format_summary()}"
        )

    def _get(self, plan_id: str | None) -> str:
        if not plan_id:
            active = self._store.get_active_plans()
            if not active:
                return "没有活跃计划"
            return "\n\n".join(p.format_summary() for p in active)
        plan = self._store.load(plan_id)
        if not plan:
            return f"未找到计划: {plan_id}"
        summary = plan.format_summary(show_evidence=True)
        if plan.approach.strip():
            return f"## 方案\n{plan.approach.strip()}\n\n## 步骤\n{summary}"
        return summary

    def _close(self, plan_id: str | None, new_status: str) -> str:
        if not plan_id:
            return "错误：需要 plan_id"
        plan = self._store.load(plan_id)
        if not plan:
            return f"未找到计划: {plan_id}"
        # 完成闸：此前 complete 无条件接受，「完成」= 模型自称完成（dogfood 三轮都漏了计划里写明的判据）。
        # 放弃（abandon）不设闸。
        if new_status == "completed":
            problems = plan.closure_problems()
            if problems:
                return (
                    f"计划 {plan_id} 不能标为完成：\n" + "\n".join(f"- {p}" for p in problems)
                    + "\n做完并补齐证据后再 complete；确实做不了的步骤标 skipped 并写理由。"
                )
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
