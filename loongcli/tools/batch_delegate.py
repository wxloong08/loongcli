from __future__ import annotations
import asyncio
import json
import logging
from typing import TYPE_CHECKING, Callable

from loongcli.tools.base import Tool, ToolRegistry
from loongcli.tools.routing import AgentRole, filter_tools
from loongcli.tools.agent_tool import SUBAGENT_SYSTEM_PROMPT
from loongcli.core.task import (
    BATCH_RESULT_BUDGET,
    MIN_TASK_RESULT_CAP,
    TaskManager,
    TaskStatus,
    cap_task_result,
    trace_path_of,
)
from loongcli.core.agent import AgentLoop, AgentServices
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
                "description": (
                    "Soft deadline in seconds (default 300). Tasks still making "
                    "progress get exponentially backed-off extensions up to 3x total; "
                    "only stalled tasks (zero progress within a probe window) are "
                    "cancelled early as failed."
                ),
            },
        },
        "required": ["tasks"],
    }
    supports_progress = True

    # 进展探测参数：首个窗口必须长于内建工具自身的超时上限（shell 默认 120s），
    # 否则会把"正在跑长命令/长 LLM 调用"误判为卡死；窗口指数增长（退避），
    # 总时长硬顶 = 软超时 × HARD_CAP_FACTOR。
    PROBE_INITIAL = 150.0
    HARD_CAP_FACTOR = 3

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
                role=AgentRole.SUBAGENT,
                interactive=False,
                services=AgentServices(
                    compactor=compactor,
                    telemetry=self._telemetry,
                    hook_manager=self._hook_manager,
                ),
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
        # 软超时 + 进展探测续期：timeout 到点不立即杀。仍在推进（消息数增长）的
        # 任务按指数退避窗口续命至硬顶；探测窗口内零进展 = 真卡住，立即取消并
        # 计为失败——超时不是失败，卡住才是。无效循环不在这层管：AgentLoop 自带
        # 循环检测 + max_iterations 会自行终止（三闸正交）。
        _, pending = await asyncio.wait(async_tasks, timeout=timeout)
        elapsed = float(timeout)
        hard_cap = float(timeout) * self.HARD_CAP_FACTOR
        probe = self.PROBE_INITIAL
        stalled_ids: set[str] = set()
        timed_out_ids: set[str] = set()

        def _task_of(at):
            for _, tobj, _ in task_objs:
                if tobj._async_task is at:
                    return tobj
            return None

        def _msg_count(tobj) -> int:
            return len(getattr(tobj._agent_loop, "messages", None) or [])

        while pending and elapsed < hard_cap:
            snapshot = {t.id: _msg_count(t) for t in map(_task_of, pending) if t}
            window = min(probe, hard_cap - elapsed)
            _, pending = await asyncio.wait(list(pending), timeout=window)
            elapsed += window
            probe *= 2

            newly_stalled = []
            still_alive = set()
            for at in pending:
                tobj = _task_of(at)
                if tobj and _msg_count(tobj) <= snapshot.get(tobj.id, 0):
                    at.cancel()
                    stalled_ids.add(tobj.id)
                    newly_stalled.append(at)
                else:
                    still_alive.add(at)
            if newly_stalled:
                await asyncio.gather(*newly_stalled, return_exceptions=True)
            pending = still_alive

        if pending:  # 硬顶：仍在推进但总时长预算耗尽
            for at in pending:
                at.cancel()
                tobj = _task_of(at)
                if tobj:
                    timed_out_ids.add(tobj.id)
            await asyncio.gather(*pending, return_exceptions=True)

        for _, tobj, _ in task_objs:
            if tobj.id in stalled_ids:
                tobj.status = TaskStatus.FAILED
                tobj.result = self._salvage_partial(
                    tobj,
                    reason=f"探测窗口内零进展，判定卡死并取消（软超时 {timeout:.0f}s 后已续命探测）",
                )
            elif tobj.id in timed_out_ids:
                tobj.status = TaskStatus.TIMEOUT
                tobj.result = self._salvage_partial(
                    tobj,
                    reason=f"到达总时长硬顶 {hard_cap:.0f}s，取消时仍在推进——超时不是失败",
                )

        batch_ids = {tobj.id for _, tobj, _ in task_objs}
        self._task_manager._notifications = [
            n for n in self._task_manager._notifications
            if n["task_id"] not in batch_ids
        ]

        results = []
        n_completed = 0
        n_timed_out = 0
        n_failed = 0
        # 结果自管预算：总额按任务数均摊、截断附 trace 指针。入口的 ToolResultManager
        # 对本工具豁免（结果不可重放，一刀切 2000 预览曾把 7 路 37K 调研截到合成层近盲），
        # 预算责任在这里。prompt 回显只留摘要——全文是主代理上一条消息里自己写的，纯重复。
        per_cap = max(MIN_TASK_RESULT_CAP, BATCH_RESULT_BUDGET // len(task_objs))
        for i, tobj, prompt in task_objs:
            if tobj.status == TaskStatus.COMPLETED:
                n_completed += 1
            elif tobj.status == TaskStatus.TIMEOUT:
                n_timed_out += 1
            else:
                n_failed += 1
            results.append({
                "index": i,
                "prompt": prompt if len(prompt) <= 200 else prompt[:200] + "…",
                "status": tobj.status.value,
                "result": cap_task_result(tobj.result or "", per_cap, trace_path_of(tobj)),
            })

        # warning 放字典首位：合成纪律提醒挂在失败发生点，确定性文本比系统
        # 提示远水近得多——gomami 事故里主代理在 1/3 失败下宣布"信息够了"
        payload: dict = {}
        warning_parts = []
        if n_timed_out:
            warning_parts.append(
                f"{n_timed_out}/{total} 子任务到达时长硬顶（超时不是失败，取消前仍在推进）"
                "——其工具调用清单与 trace 路径已附在对应 result，可据此跟进"
            )
        if n_failed:
            warning_parts.append(
                f"{n_failed}/{total} 子任务失败或卡死——其结论不可依赖"
                "（卡死任务如有取消前成果已附在 result）"
            )
        if warning_parts:
            payload["warning"] = (
                "；".join(warning_parts)
                + "。缺口未补齐前禁止用假设/典型值填补，向用户说明缺口或重试。"
            )
        payload.update({
            "completed": n_completed,
            "timed_out": n_timed_out,
            "failed": n_failed,
            "results": results,
        })
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def _salvage_partial(self, task_obj, reason: str) -> str:
        """被取消任务的部分成果打捞——取消前的工具调用与产出仍有价值，不当零结果丢弃。

        确定性提取（不用 LLM 补摘要：会编、要钱）：工具调用清单（参数截断，
        URL 类通常完整保留）+ 最后一段文本输出 + trace 文件路径。"""
        loop = task_obj._agent_loop
        calls: list[str] = []
        last_text = ""
        for m in list(getattr(loop, "messages", None) or []):
            if m.get("role") != "assistant":
                continue
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function", {})
                args = fn.get("arguments", "") or ""
                if len(args) > 200:
                    args = args[:200] + "…"
                calls.append(f"{fn.get('name', '?')} {args}")
            if m.get("content"):
                last_text = m["content"]

        lines = [reason + "。"]
        if calls:
            shown = calls[-30:]
            header = f"取消前已执行 {len(calls)} 次工具调用"
            if len(calls) > len(shown):
                header += f"（列出最近 {len(shown)} 次）"
            lines.append(header + "：")
            lines.extend(f"- {c}" for c in shown)
        else:
            lines.append("取消前未完成任何工具调用。")
        if last_text:
            if len(last_text) > 500:
                last_text = last_text[:500] + "…"
            lines.append(f"最后输出：{last_text}")
        trace = trace_path_of(task_obj)
        if trace:
            lines.append(f"完整现场：{trace}")
        return "\n".join(lines)

    def _build_sub_registry(self, allowed_tools: list[str] | None) -> ToolRegistry:
        visible = filter_tools(self._parent_registry._tools, AgentRole.SUBAGENT)
        sub_registry = ToolRegistry()
        for name, tool in visible.items():
            if allowed_tools and name not in allowed_tools:
                continue
            sub_registry.register(tool)
        return sub_registry
