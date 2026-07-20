import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from loongcli.core.task import Task, TaskManager, TaskStatus
from loongcli.core.events import AgentDone


def test_task_drain_mailbox():
    task = Task(prompt="test")
    task.mailbox.append("msg1")
    task.mailbox.append("msg2")
    msgs = task.drain_mailbox()
    assert msgs == ["msg1", "msg2"]
    assert task.mailbox == []


def test_task_manager_register_and_get():
    tm = TaskManager()
    task = Task(prompt="test")
    tm.register(task)
    assert tm.get(task.id) is task
    assert tm.get("nonexistent") is None


def test_send_message_to_running_task():
    tm = TaskManager()
    task = Task(prompt="test")
    tm.register(task)
    result = tm.send_message(task.id, "hello")
    assert "已送达" in result
    assert task.mailbox == ["hello"]


def test_send_message_nonexistent():
    tm = TaskManager()
    result = tm.send_message("bad-id", "hello")
    assert "错误" in result


def test_send_message_to_failed_task_reports_failure():
    """发给 FAILED 任务不塞死信箱假成功，明确报无法接收。"""
    tm = TaskManager()
    task = Task(prompt="test", status=TaskStatus.FAILED)
    tm.register(task)
    result = tm.send_message(task.id, "hello")
    assert "无法接收" in result
    assert task.mailbox == []  # 不塞死信箱


def test_get_status():
    tm = TaskManager()
    task = Task(prompt="do stuff")
    tm.register(task)
    info = tm.get_status(task.id)
    assert info["status"] == "running"
    assert info["prompt"] == "do stuff"


def test_get_status_nonexistent():
    tm = TaskManager()
    info = tm.get_status("bad-id")
    assert "error" in info


def test_notifications():
    tm = TaskManager()
    task = Task(prompt="test")
    task.status = TaskStatus.COMPLETED
    task.result = "done"
    tm.push_notification(task)

    notifs = tm.drain_notifications()
    assert len(notifs) == 1
    assert notifs[0]["task_id"] == task.id
    assert notifs[0]["result"] == "done"
    assert tm.drain_notifications() == []


def test_format_notification():
    tm = TaskManager()
    notif = {"task_id": "abc123", "status": "completed", "result": "All good"}
    text = tm.format_notification(notif)
    assert "abc123" in text
    assert "All good" in text


@pytest.mark.asyncio
async def test_create_and_run():
    tm = TaskManager()

    mock_agent = MagicMock()

    async def fake_stream(prompt):
        yield AgentDone(content="task result here")

    mock_agent.run_stream = fake_stream

    task = await tm.create_and_run(prompt="do something", agent_loop=mock_agent, depth=1)
    assert task.status == TaskStatus.RUNNING

    await asyncio.sleep(0.1)

    assert task.status == TaskStatus.COMPLETED
    assert task.result == "task result here"

    notifs = tm.drain_notifications()
    assert len(notifs) == 1
    assert notifs[0]["task_id"] == task.id


def _hanging_agent(started: asyncio.Event) -> MagicMock:
    """run_stream 挂起不返回的假 agent，用于取消路径测试。"""
    mock_agent = MagicMock()

    async def hang_stream(prompt):
        started.set()
        await asyncio.sleep(60)
        yield AgentDone(content="never")

    mock_agent.run_stream = hang_stream
    return mock_agent


@pytest.mark.asyncio
async def test_stop_preserves_reason_and_pushes_single_notification():
    # 回归 1：无幂等闸时取消分支会二次落定——reason 被"任务被取消"覆盖 + 推两条通知
    # 回归 2（真机 3.1）：落定必须就地完成，stop 返回后立即可观测，
    # 不等取消送达（同一轮工具调用间没有让出点）
    tm = TaskManager()
    started = asyncio.Event()
    task = await tm.create_and_run(prompt="long job", agent_loop=_hanging_agent(started))
    await started.wait()

    msg = tm.stop(task.id, "跑偏了")
    assert task.id in msg

    # 立即断言，不 sleep——时滞本身就是 bug
    assert task.status == TaskStatus.FAILED
    assert "跑偏了" in task.result
    notifs = tm.drain_notifications()
    assert len(notifs) == 1
    assert notifs[0]["task_id"] == task.id
    assert "跑偏了" in notifs[0]["result"]

    # 取消送达后：幂等闸挡住二次落定，无第二条通知、reason 不被覆盖
    await asyncio.sleep(0.05)
    assert "跑偏了" in task.result
    assert tm.drain_notifications() == []


@pytest.mark.asyncio
async def test_direct_cancel_without_stop_reason():
    # 非 stop_task 的裸取消（如事件循环收尾）：结果落"任务被取消"，通知一条
    tm = TaskManager()
    started = asyncio.Event()
    task = await tm.create_and_run(prompt="long job", agent_loop=_hanging_agent(started))
    await started.wait()

    task._async_task.cancel()
    await asyncio.sleep(0.05)

    assert task.status == TaskStatus.FAILED
    assert task.result == "任务被取消"
    assert len(tm.drain_notifications()) == 1


@pytest.mark.asyncio
async def test_stop_task_queued_behind_semaphore():
    # 回归：取消打在信号量 acquire 上时 _run_agent 未进场——
    # 不在 _run_agent_with_limit 兜底的话，任务永远 RUNNING 且无通知
    tm = TaskManager()
    tm._semaphore = asyncio.Semaphore(1)

    started = asyncio.Event()
    occupier = await tm.create_and_run(prompt="occupy", agent_loop=_hanging_agent(started))
    await started.wait()

    queued = await tm.create_and_run(prompt="queued job", agent_loop=_hanging_agent(asyncio.Event()))
    await asyncio.sleep(0.05)  # 让它挂到 acquire 上
    assert queued.status == TaskStatus.RUNNING

    tm.stop(queued.id, "排队时停止")

    assert queued.status == TaskStatus.FAILED
    assert "排队时停止" in queued.result
    notifs = tm.drain_notifications()
    assert len(notifs) == 1
    assert notifs[0]["task_id"] == queued.id

    # 取消打在 acquire 上：_run_agent_with_limit 兜底的幂等闸挡住二次推送
    await asyncio.sleep(0.05)
    assert tm.drain_notifications() == []

    tm.stop(occupier.id, "收尾")
    await asyncio.sleep(0.05)


def test_stop_task_without_async_task_finalizes_inline():
    # 防御路径：手工注册、无执行协程的任务，stop 就地落定（没人会替它落）
    tm = TaskManager()
    task = Task(prompt="manual")
    tm.register(task)

    msg = tm.stop(task.id, "手工停止")
    assert "已停止" in msg
    assert task.status == TaskStatus.FAILED
    assert "手工停止" in task.result
    assert len(tm.drain_notifications()) == 1


# ---- delegate 类结果自管截断（cap_task_result / trace_path_of）----

from loongcli.core.task import cap_task_result, trace_path_of  # noqa: E402


def test_cap_task_result_short_text_unchanged():
    assert cap_task_result("短结果", 1000, None) == "短结果"


def test_cap_task_result_cuts_at_newline_with_trace_pointer():
    text = "\n".join(f"line-{i:04d}" for i in range(1000))  # ~10K
    capped = cap_task_result(text, 3000, "/tmp/tasks/abc123.json")
    assert len(capped) < len(text)
    # 换行边界截断：保留部分是原文前缀、且以完整行结尾
    import re
    body = capped.split("\n\n[结果已截断")[0]
    assert body == text[: len(body)]
    assert re.fullmatch(r"line-\d{4}", body.splitlines()[-1])
    assert "/tmp/tasks/abc123.json" in capped
    assert "read_file" in capped
    assert "不要重跑" in capped
    # 绝不出现误导性恢复建议——delegate 结果不可重放
    assert "重新调用工具" not in capped


def test_cap_task_result_without_trace_says_unrecoverable():
    capped = cap_task_result("x" * 5000, 2000, None)
    assert "trace 未启用" in capped
    assert "重新调用工具" not in capped


def test_trace_path_of_reads_conversation_store():
    task = Task(prompt="t")
    loop = MagicMock()
    loop.conversation_store = MagicMock(base_dir="/tmp/tasks", session_id="abc123def456")
    task._agent_loop = loop
    path = trace_path_of(task)
    assert path is not None and path.endswith("abc123def456.json")


def test_trace_path_of_none_when_no_store():
    task = Task(prompt="t")
    task._agent_loop = MagicMock(conversation_store=None)
    assert trace_path_of(task) is None
    assert trace_path_of(None) is None
