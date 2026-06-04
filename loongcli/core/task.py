from __future__ import annotations
import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from loongcli.core.agent import AgentLoop

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Task:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    prompt: str = ""
    status: TaskStatus = TaskStatus.RUNNING
    result: str = ""
    depth: int = 1
    mailbox: list[str] = field(default_factory=list)
    _async_task: asyncio.Task | None = field(default=None, repr=False)
    _agent_loop: AgentLoop | None = field(default=None, repr=False)

    def drain_mailbox(self) -> list[str]:
        msgs = list(self.mailbox)
        self.mailbox.clear()
        return msgs


class TaskManager:
    MAX_DEPTH = 3
    MAX_CONCURRENT = 8

    def __init__(self):
        self._tasks: dict[str, Task] = {}
        self._notifications: list[dict] = []
        self._semaphore = asyncio.Semaphore(self.MAX_CONCURRENT)

    def register(self, task: Task):
        self._tasks[task.id] = task

    def get(self, task_id: str) -> Task | None:
        return self._tasks.get(task_id)

    def send_message(self, task_id: str, message: str) -> str:
        task = self._tasks.get(task_id)
        if not task:
            return f"错误：未找到任务 {task_id}"

        task.mailbox.append(message)

        if task.status == TaskStatus.COMPLETED and task._agent_loop:
            task.status = TaskStatus.RUNNING
            task._async_task = asyncio.create_task(
                self._run_agent_with_limit(task, resume_message=message)
            )
            return f"已唤醒任务 {task_id}，消息已送达"

        return f"消息已送达任务 {task_id}"

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
    ) -> Task:
        task = Task(prompt=prompt, depth=depth)
        task._agent_loop = agent_loop
        agent_loop.task = task
        self.register(task)

        task._async_task = asyncio.create_task(
            self._run_agent_with_limit(task)
        )
        return task

    async def _run_agent_with_limit(self, task: Task, resume_message: str | None = None):
        async with self._semaphore:
            await self._run_agent(task, resume_message)

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
            task.status = TaskStatus.FAILED
            task.result = "任务被取消"
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.result = f"任务失败: {e}"
            logger.error("Task %s failed: %s", task.id, e)

        self.push_notification(task)
