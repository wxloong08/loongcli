from __future__ import annotations

import asyncio
import json
import pytest
from unittest.mock import MagicMock

from loongcli.core.task import TaskManager, Task, TaskStatus
from loongcli.tools.wait_tasks import WaitTasksTool
from loongcli.tools.stop_task import StopTaskTool


def _hanging_agent(started: asyncio.Event | None = None) -> MagicMock:
    """run_stream 挂起不返回的假 agent：走 create_and_run 真实装配，
    取消落定由 _run_agent_with_limit 完成（stop 不再就地落定）。"""
    agent = MagicMock()

    async def hang_stream(prompt):
        if started:
            started.set()
        await asyncio.sleep(100)
        yield  # 永远到不了，仅为构成异步生成器

    agent.run_stream = hang_stream
    return agent


# ── WaitTasksTool ───────────────────────────────────────────────────


class TestWaitTasksTool:

    def _make(self):
        tm = TaskManager()
        tool = WaitTasksTool(tm)
        return tool, tm

    @pytest.mark.asyncio
    async def test_empty_ids(self):
        tool, _ = self._make()
        result = json.loads(await tool.execute(task_ids=[]))
        assert "error" in result

    @pytest.mark.asyncio
    async def test_not_found(self):
        tool, _ = self._make()
        result = json.loads(await tool.execute(task_ids=["nonexistent"]))
        assert result["results"][0]["status"] == "not_found"

    @pytest.mark.asyncio
    async def test_already_completed(self):
        tool, tm = self._make()
        task = Task(prompt="done")
        task.status = TaskStatus.COMPLETED
        task.result = "all good"
        tm.register(task)

        result = json.loads(await tool.execute(task_ids=[task.id]))
        assert result["results"][0]["status"] == "completed"
        assert result["results"][0]["result"] == "all good"
        assert result["timed_out"] == []

    @pytest.mark.asyncio
    async def test_wait_for_running_task(self):
        tool, tm = self._make()
        task = Task(prompt="fast")
        tm.register(task)

        async def quick_work():
            await asyncio.sleep(0.05)
            task.status = TaskStatus.COMPLETED
            task.result = "finished"

        task._async_task = asyncio.create_task(quick_work())

        result = json.loads(await tool.execute(task_ids=[task.id], timeout=5))
        assert result["results"][0]["status"] == "completed"
        assert result["timed_out"] == []

    @pytest.mark.asyncio
    async def test_timeout_does_not_cancel(self):
        tool, tm = self._make()
        task = Task(prompt="slow")
        tm.register(task)

        async def slow_work():
            await asyncio.sleep(10)
            task.status = TaskStatus.COMPLETED

        task._async_task = asyncio.create_task(slow_work())

        result = json.loads(await tool.execute(task_ids=[task.id], timeout=0.05))
        assert result["timed_out"] == [task.id]
        assert result["results"][0]["status"] == "running (timeout)"
        # Task should still be running (not cancelled)
        assert task.status == TaskStatus.RUNNING
        task._async_task.cancel()
        try:
            await task._async_task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_mixed_states(self):
        tool, tm = self._make()

        t1 = Task(prompt="done")
        t1.status = TaskStatus.COMPLETED
        t1.result = "r1"
        tm.register(t1)

        t2 = Task(prompt="fail")
        t2.status = TaskStatus.FAILED
        t2.result = "err"
        tm.register(t2)

        result = json.loads(await tool.execute(task_ids=[t1.id, t2.id, "nope"]))
        statuses = {r["task_id"]: r["status"] for r in result["results"]}
        assert statuses[t1.id] == "completed"
        assert statuses[t2.id] == "failed"
        assert statuses["nope"] == "not_found"

    @pytest.mark.asyncio
    async def test_big_results_capped_with_trace_pointer(self):
        # 自管预算（与 batch_delegate 同口径）：入口豁免后超长结果在这里按
        # 任务数均摊截断，附 trace 指针
        tool, tm = self._make()

        t1 = Task(prompt="big")
        t1.status = TaskStatus.COMPLETED
        t1.result = "数据行\n" * 10000  # 40K chars
        loop = MagicMock()
        loop.conversation_store = MagicMock(
            base_dir="/tmp/tasks", session_id="wt0000000001")
        t1._agent_loop = loop
        tm.register(t1)

        result = json.loads(await tool.execute(task_ids=[t1.id]))
        r = result["results"][0]["result"]
        assert len(r) <= 36000 + 200  # 单任务 per_cap = BATCH_RESULT_BUDGET
        assert "结果已截断" in r
        assert "wt0000000001.json" in r
        assert "重新调用工具" not in r


# ── StopTaskTool ────────────────────────────────────────────────────


class TestStopTaskTool:

    def _make(self):
        tm = TaskManager()
        tool = StopTaskTool(tm)
        return tool, tm

    @pytest.mark.asyncio
    async def test_stop_running(self):
        tool, tm = self._make()
        started = asyncio.Event()
        task = await tm.create_and_run(prompt="long", agent_loop=_hanging_agent(started))
        await started.wait()

        result = await tool.execute(task_id=task.id, reason="跑偏了")
        assert "已停止" in result
        await asyncio.sleep(0.05)  # 落定发生在执行协程的取消分支，等它跑完

        assert task.status == TaskStatus.FAILED
        assert "跑偏了" in task.result

        # Notification pushed
        notifs = tm.drain_notifications()
        assert len(notifs) == 1
        assert notifs[0]["task_id"] == task.id

    @pytest.mark.asyncio
    async def test_stop_completed(self):
        tool, tm = self._make()
        task = Task(prompt="done")
        task.status = TaskStatus.COMPLETED
        tm.register(task)

        result = await tool.execute(task_id=task.id, reason="test")
        assert "已结束" in result

    @pytest.mark.asyncio
    async def test_stop_not_found(self):
        tool, tm = self._make()
        result = await tool.execute(task_id="ghost", reason="test")
        assert "未找到" in result

    @pytest.mark.asyncio
    async def test_default_reason(self):
        tool, tm = self._make()
        started = asyncio.Event()
        task = await tm.create_and_run(prompt="x", agent_loop=_hanging_agent(started))
        await started.wait()

        await tool.execute(task_id=task.id)
        await asyncio.sleep(0.05)
        assert "手动停止" in task.result


# ── TaskManager.wait / stop / semaphore ─────────────────────────────


class TestTaskManagerExtensions:

    @pytest.mark.asyncio
    async def test_semaphore_limits_concurrency(self):
        tm = TaskManager()
        tm._semaphore = asyncio.Semaphore(2)

        running_count = 0
        max_running = 0

        async def tracked_agent(task, resume_message=None):
            nonlocal running_count, max_running
            running_count += 1
            max_running = max(max_running, running_count)
            await asyncio.sleep(0.05)
            running_count -= 1
            task.status = TaskStatus.COMPLETED
            task.result = "ok"

        tm._run_agent = tracked_agent

        tasks = []
        for i in range(5):
            t = Task(prompt=f"task-{i}")
            t._agent_loop = MagicMock()
            t._agent_loop.task = None
            tm.register(t)
            t._async_task = asyncio.create_task(tm._run_agent_with_limit(t))
            tasks.append(t)

        await asyncio.gather(*[t._async_task for t in tasks])
        assert max_running <= 2

    @pytest.mark.asyncio
    async def test_max_depth_is_3(self):
        assert TaskManager.MAX_DEPTH == 3

    @pytest.mark.asyncio
    async def test_stop_pushes_notification(self):
        tm = TaskManager()
        started = asyncio.Event()
        task = await tm.create_and_run(prompt="x", agent_loop=_hanging_agent(started))
        await started.wait()

        tm.stop(task.id, "test reason")
        await asyncio.sleep(0.05)

        notifs = tm.drain_notifications()
        assert len(notifs) == 1
        assert notifs[0]["status"] == "failed"
        assert "test reason" in notifs[0]["result"]
