from __future__ import annotations
import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from loongcli.core.agent import AgentLoop

logger = logging.getLogger(__name__)

# delegate 类结果预算：这些结果不可重放（重跑子代理非确定且烧钱），不能交给
# ToolResultManager 的一刀切截断——其"重新调用工具"提示对 delegate 前提失效，
# 2000 字符预览曾把 7 路子代理 37K 调研截到合成层近盲（2026-07-20 真实会话事故）。
# 由 batch_delegate/task_status/wait_tasks 按此预算自行截断，附 trace 指针可恢复。
BATCH_RESULT_BUDGET = 36000     # 一批结果的总预算，按任务数均摊
MIN_TASK_RESULT_CAP = 2000      # 均摊下限：任务再多也保住可用的最小结果
SINGLE_TASK_RESULT_CAP = 16000  # 单任务查询（task_status）的上限


class TaskStatus(Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    # 超时≠失败：被超时取消的任务通常已有部分成果（工具调用、已读页面），
    # 与真正报错的任务必须可区分——gomami 事故里 8 篇已读测评被当零结果丢弃
    TIMEOUT = "timeout"


@dataclass
class Task:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    prompt: str = ""
    status: TaskStatus = TaskStatus.RUNNING
    result: str = ""
    depth: int = 1
    parent_id: str | None = None  # 委派树父任务；None = 根（主 agent 直接派发）
    mailbox: list[str] = field(default_factory=list)
    _async_task: asyncio.Task | None = field(default=None, repr=False)
    _agent_loop: AgentLoop | None = field(default=None, repr=False)

    def drain_mailbox(self) -> list[str]:
        msgs = list(self.mailbox)
        self.mailbox.clear()
        return msgs


def trace_path_of(task: Task | None) -> str | None:
    """子任务 trace 文件路径（<trace_dir>/<task_id>.json）；trace 未启用返回 None。"""
    loop = task._agent_loop if task else None
    store = getattr(loop, "conversation_store", None) if loop else None
    if store is None:
        return None
    try:
        return str(Path(store.base_dir) / f"{store.session_id}.json")
    except Exception:
        return None


def cap_task_result(text: str, cap: int, trace_path: str | None) -> str:
    """delegate 类结果的自管截断：换行边界截到 cap，提示指向 trace 恢复。
    绝不写"重新调用工具"——重跑子代理非确定且烧钱，是错误的恢复建议。"""
    if len(text) <= cap:
        return text
    kept = text[:cap]
    nl = kept.rfind("\n")
    if nl > cap * 0.7:
        kept = kept[:nl]
    if trace_path:
        notice = (
            f"\n\n[结果已截断：保留 {len(kept)}/{len(text)} 字符。"
            f"完整结果在 trace: {trace_path}，用 read_file 分页读取，不要重跑子任务]"
        )
    else:
        notice = (
            f"\n\n[结果已截断：保留 {len(kept)}/{len(text)} 字符。"
            f"trace 未启用，截断部分不可恢复]"
        )
    return kept + notice


class TaskManager:
    MAX_DEPTH = 3
    MAX_CONCURRENT = 8

    def __init__(self, trace_dir: Path | None = None):
        self._tasks: dict[str, Task] = {}
        self._notifications: list[dict] = []
        self._semaphore = asyncio.Semaphore(self.MAX_CONCURRENT)
        # 子任务 trace 落盘目录；None = 不落盘（测试默认，主装配显式开启）
        self._trace_dir = trace_dir

    def register(self, task: Task):
        self._tasks[task.id] = task

    def get(self, task_id: str) -> Task | None:
        return self._tasks.get(task_id)

    def send_message(self, task_id: str, message: str) -> str:
        task = self._tasks.get(task_id)
        if not task:
            return f"错误：未找到任务 {task_id}"

        # RUNNING：塞信箱，由运行中的 agent 循环 drain。
        if task.status == TaskStatus.RUNNING:
            task.mailbox.append(message)
            return f"消息已送达任务 {task_id}"

        # COMPLETED 且可唤醒：塞信箱并复活。
        if task.status == TaskStatus.COMPLETED and task._agent_loop:
            task.mailbox.append(message)
            task.status = TaskStatus.RUNNING
            task._async_task = asyncio.create_task(
                self._run_agent_with_limit(task, resume_message=message)
            )
            return f"已唤醒任务 {task_id}，消息已送达"

        # FAILED，或 COMPLETED 但无 agent_loop 无法唤醒——不塞死信箱假成功，明确报错。
        return f"任务 {task_id} 已结束（{task.status.value}），无法接收消息，请另派新任务"

    def get_status(self, task_id: str) -> dict[str, Any]:
        task = self._tasks.get(task_id)
        if not task:
            return {"error": f"未找到任务 {task_id}"}
        return {
            "task_id": task.id,
            "status": task.status.value,
            "prompt": task.prompt,
            "result": task.result if task.status != TaskStatus.RUNNING else "(running)",
        }

    def push_notification(self, task: Task):
        self._notifications.append({
            "task_id": task.id,
            "status": task.status.value,
            "result": task.result,
        })

    def drain_notifications(self) -> list[dict]:
        notifs = list(self._notifications)
        self._notifications.clear()
        return notifs

    def format_notification(self, notif: dict) -> str:
        return (
            f"[task-notification] 任务 {notif['task_id']} 已完成。\n"
            f"结果：{notif['result']}"
        )

    def stop(self, task_id: str, reason: str) -> str:
        task = self._tasks.get(task_id)
        if not task:
            return f"未找到任务 {task_id}"
        if task.status != TaskStatus.RUNNING:
            return f"任务 {task_id} 已结束 ({task.status.value})，无需停止"
        if task._async_task and not task._async_task.done():
            task._async_task.cancel()
        # 就地落定：stop 后紧跟 task_status 必须立即见 failed（真机 3.1 抓到
        # 推迟落定的时滞——取消送达要等下一个让出点，同一轮工具调用间没有）。
        # 执行协程的取消分支有幂等闸，不会二次落定/重复推送。
        task.status = TaskStatus.FAILED
        task.result = f"被停止: {reason}"
        self.push_notification(task)
        return f"已停止任务 {task_id}"

    async def wait(self, task_ids: list[str], timeout: float = 300) -> dict:
        results = []
        pending_tasks = []

        for tid in task_ids:
            task = self._tasks.get(tid)
            if not task:
                results.append({"task_id": tid, "status": "not_found", "result": ""})
            elif task.status != TaskStatus.RUNNING:
                results.append({
                    "task_id": tid,
                    "status": task.status.value,
                    "result": task.result,
                })
            else:
                pending_tasks.append(task)

        if pending_tasks:
            aws = [t._async_task for t in pending_tasks if t._async_task]
            if aws:
                await asyncio.wait(aws, timeout=timeout)

        timed_out = []
        for task in pending_tasks:
            entry = {
                "task_id": task.id,
                "status": task.status.value,
                "result": task.result,
            }
            if task.status == TaskStatus.RUNNING:
                entry["status"] = "running (timeout)"
                timed_out.append(task.id)
            results.append(entry)

        return {"results": results, "timed_out": timed_out}

    async def create_and_run(
        self,
        prompt: str,
        agent_loop: AgentLoop,
        depth: int = 1,
        parent_id: str | None = None,
    ) -> Task:
        task = Task(prompt=prompt, depth=depth, parent_id=parent_id)
        task._agent_loop = agent_loop
        agent_loop.task = task
        self.register(task)
        self._attach_trace(task, agent_loop)

        task._async_task = asyncio.create_task(
            self._run_agent_with_limit(task)
        )
        return task

    def _attach_trace(self, task: Task, agent_loop: AgentLoop) -> None:
        """给子 agent 挂 trace store：消息流复用会话持久化机制落盘为
        <trace_dir>/<task_id>.json，meta 记 parent_task_id 形成可回放的委派树。
        立即写一次——树结构先于执行存在，agent 跑挂了 trace 也不缺席。"""
        if self._trace_dir is None:
            return
        if getattr(agent_loop, "conversation_store", None) is not None:
            return
        from loongcli.memory.conversation import ConversationStore
        store = ConversationStore(
            base_dir=self._trace_dir,
            session_id=task.id,
            extra_meta={
                "task_id": task.id,
                "parent_task_id": task.parent_id,
                "depth": task.depth,
                "prompt": task.prompt[:200],
            },
        )
        agent_loop.conversation_store = store
        try:
            store.save(list(getattr(agent_loop, "messages", []) or []))
        except Exception:
            logger.debug("task %s 初始 trace 落盘失败", task.id, exc_info=True)

    async def _run_agent_with_limit(self, task: Task, resume_message: str | None = None):
        try:
            async with self._semaphore:
                await self._run_agent(task, resume_message)
        except asyncio.CancelledError:
            # 还在排队等并发闸就被停止：_run_agent 未进场，不兜这里的话
            # 任务会永远停在 RUNNING 且无通知
            self._finalize_stopped(task)

    def _finalize_stopped(self, task: Task):
        """取消路径的兜底落定，带幂等闸：stop() 已就地落定时跳过（单线程事件
        循环下确定性成立），只兜非 stop_task 的裸取消（如事件循环收尾）。"""
        if task.status != TaskStatus.RUNNING:
            self._persist_trace(task)
            return
        task.status = TaskStatus.FAILED
        task.result = "任务被取消"
        self.push_notification(task)
        self._persist_trace(task)

    def _persist_trace(self, task: Task) -> None:
        """取消/异常中断时尽力落盘现场——run_stream 的自然落盘点走不到，
        不补这一笔的话被停任务的 trace 停留在初始状态，失去排障价值。"""
        loop = task._agent_loop
        if loop is None or getattr(loop, "conversation_store", None) is None:
            return
        try:
            loop.conversation_store.save(list(loop.messages))
        except Exception:
            logger.debug("task %s 中断 trace 落盘失败", task.id, exc_info=True)

    async def _run_agent(self, task: Task, resume_message: str | None = None):
        try:
            agent = task._agent_loop
            prompt = resume_message or task.prompt
            final_text = ""
            async for event in agent.run_stream(prompt):
                from loongcli.core.events import AgentDone
                if isinstance(event, AgentDone):
                    final_text = event.content

            task.status = TaskStatus.COMPLETED
            task.result = final_text
        except asyncio.CancelledError:
            self._finalize_stopped(task)
            return
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.result = f"任务失败: {e}"
            logger.error("Task %s failed: %s", task.id, e)
            self._persist_trace(task)

        self.push_notification(task)
