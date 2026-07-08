"""per-task trace 树：子任务消息流落盘 + parent_task_id 关联（todo Step 2）。

判据：delegate 一个任务后磁盘存在该 task 的 json 且含 parent_task_id；
coordinator 三层链（主→协调者→worker）的 trace 能按父子关系串起来。
"""
import asyncio
import json

import pytest
from unittest.mock import MagicMock

from loongcli.core.agent import AgentLoop
from loongcli.core.events import AgentDone
from loongcli.core.llm import LLMClient
from loongcli.core.task import TaskManager
from loongcli.security.permissions import PermissionChecker
from loongcli.tools.agent_tool import AgentTool
from loongcli.tools.base import ToolRegistry

from tests.core.test_agent import _make_text_chunks, _make_tool_call_chunks


class _FakeAgent:
    """最小假 agent：conversation_store 显式为 None 才会被挂 trace
    （MagicMock 的自动属性非 None，会被挂钩误判为已有 store）。"""

    def __init__(self):
        self.conversation_store = None
        self.messages: list[dict] = []
        self.task = None

    async def run_stream(self, prompt):
        yield AgentDone(content="done")


@pytest.mark.asyncio
async def test_trace_file_written_with_parent_link(tmp_path):
    tm = TaskManager(trace_dir=tmp_path)
    agent = _FakeAgent()

    task = await tm.create_and_run(
        prompt="调研 X", agent_loop=agent, depth=2, parent_id="coord123",
    )
    await asyncio.sleep(0.05)

    trace_file = tmp_path / f"{task.id}.json"
    assert trace_file.exists()  # 树结构先于执行存在（创建时即落 meta）
    meta = json.loads(trace_file.read_text(encoding="utf-8"))["meta"]
    assert meta["task_id"] == task.id
    assert meta["parent_task_id"] == "coord123"
    assert meta["depth"] == 2
    assert "调研 X" in meta["prompt"]


@pytest.mark.asyncio
async def test_trace_disabled_without_trace_dir(tmp_path):
    # 测试默认（trace_dir=None）不落盘、不挂 store——不污染真实用户目录
    tm = TaskManager()
    agent = _FakeAgent()
    await tm.create_and_run(prompt="x", agent_loop=agent)
    await asyncio.sleep(0.05)
    assert agent.conversation_store is None
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_trace_persisted_on_stop(tmp_path):
    # 被停任务的 trace 不停留在初始状态：取消落定时尽力落盘现场
    tm = TaskManager(trace_dir=tmp_path)

    class HangingAgent(_FakeAgent):
        def __init__(self, started):
            super().__init__()
            self._started = started

        async def run_stream(self, prompt):
            self.messages.append({"role": "user", "content": prompt})
            self._started.set()
            await asyncio.sleep(60)
            yield AgentDone(content="never")

    started = asyncio.Event()
    agent = HangingAgent(started)
    task = await tm.create_and_run(prompt="long", agent_loop=agent)
    await started.wait()

    tm.stop(task.id, "现场落盘测试")
    await asyncio.sleep(0.05)

    data = json.loads((tmp_path / f"{task.id}.json").read_text(encoding="utf-8"))
    assert data["messages"], "中断落盘缺席——trace 停在初始状态"


@pytest.mark.asyncio
async def test_coordinator_chain_traces_link_up(tmp_path):
    """主→协调者→worker 三层链：worker trace 的 parent_task_id 指向协调者任务，
    协调者 trace 是根（parent 为 None），链路可回放。"""
    llm = LLMClient(api_key="test")

    # 脚本化三次 LLM 调用：协调者派 worker → worker 干活 → 协调者收尾
    scripts = [
        _make_tool_call_chunks("call_1", "delegate", {"prompt": "工人任务", "mode": "sync"}),
        _make_text_chunks(["WORKER_DONE"]),
        _make_text_chunks(["COORD_DONE"]),
    ]
    call_idx = 0

    async def scripted_stream(**kwargs):
        nonlocal call_idx
        chunks = scripts[min(call_idx, len(scripts) - 1)]
        call_idx += 1
        for c in chunks:
            yield c

    llm.chat_stream = scripted_stream

    tm = TaskManager(trace_dir=tmp_path)
    registry = ToolRegistry()
    tool = AgentTool(
        task_manager=tm,
        llm=llm,
        parent_registry=registry,
        security=PermissionChecker(),
    )
    registry.register(tool)  # 与主装配一致：delegate 在父注册表内，协调者才拿得到克隆

    result = json.loads(await tool.execute(prompt="协调任务", coordinator=True, mode="sync"))
    assert result["status"] == "completed"
    coord_id = result["task_id"]

    traces = {}
    for f in tmp_path.glob("*.json"):
        meta = json.loads(f.read_text(encoding="utf-8"))["meta"]
        traces[meta["task_id"]] = meta

    assert len(traces) == 2, f"应有协调者+worker 两份 trace，实际 {list(traces)}"
    coord_meta = traces[coord_id]
    worker_meta = next(m for tid, m in traces.items() if tid != coord_id)

    assert coord_meta["parent_task_id"] is None      # 主 agent 派出的根
    assert coord_meta["depth"] == 1
    assert worker_meta["parent_task_id"] == coord_id  # worker 挂在协调者名下
    assert worker_meta["depth"] == 2
    assert "工人任务" in worker_meta["prompt"]

    # worker 的消息流真实落盘（run_stream 自然落盘点）
    worker_file = tmp_path / f"{worker_meta['task_id']}.json"
    worker_msgs = json.loads(worker_file.read_text(encoding="utf-8"))["messages"]
    assert any("WORKER_DONE" in str(m.get("content", "")) for m in worker_msgs)
