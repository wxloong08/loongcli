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
