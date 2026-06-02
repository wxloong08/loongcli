import asyncio
import json
import pytest
from unittest.mock import MagicMock, patch

from loongcli.tools.batch_delegate import BatchDelegateTool
from loongcli.tools.base import ToolRegistry
from loongcli.core.task import TaskManager, TaskStatus
from loongcli.core.events import AgentDone, BatchProgress


class FakeAgent:
    def __init__(self, **kwargs):
        self.task = None

    async def run_stream(self, prompt):
        yield AgentDone(content=f"result: {prompt}")


class FailingAgent:
    def __init__(self, **kwargs):
        self.task = None

    async def run_stream(self, prompt):
        raise RuntimeError("agent failed")
        yield  # noqa: make it a generator


class SlowAgent:
    def __init__(self, **kwargs):
        self.task = None

    async def run_stream(self, prompt):
        await asyncio.sleep(10)
        yield AgentDone(content="unreachable")


class TestBatchDelegateTool:
    def _make_tool(self, depth=0):
        tm = TaskManager()
        llm = MagicMock()
        registry = ToolRegistry()

        dummy = MagicMock()
        dummy.name = "shell"
        registry.register(dummy)

        delegate_dummy = MagicMock()
        delegate_dummy.name = "delegate"
        registry.register(delegate_dummy)

        security = MagicMock()
        tool = BatchDelegateTool(
            task_manager=tm,
            llm=llm,
            parent_registry=registry,
            security=security,
            depth=depth,
        )
        return tool, tm

    @pytest.mark.asyncio
    async def test_max_depth_rejected(self):
        tool, _ = self._make_tool(depth=TaskManager.MAX_DEPTH - 1)
        result = await tool.execute(tasks=[{"prompt": "test"}])
        data = json.loads(result)
        assert "error" in data
        assert "嵌套深度" in data["error"]

    @pytest.mark.asyncio
    async def test_empty_tasks(self):
        tool, _ = self._make_tool()
        result = await tool.execute(tasks=[])
        data = json.loads(result)
        assert "error" in data
        assert "为空" in data["error"]

    def test_supports_progress(self):
        tool, _ = self._make_tool()
        assert tool.supports_progress is True

    def test_tool_schema(self):
        tool, _ = self._make_tool()
        reg = ToolRegistry()
        reg.register(tool)
        schemas = reg.get_tool_schemas()
        assert len(schemas) == 1
        fn = schemas[0]["function"]
        assert fn["name"] == "batch_delegate"
        assert "tasks" in fn["parameters"]["properties"]
        assert "timeout" in fn["parameters"]["properties"]

    def test_sub_registry_excludes_blacklisted(self):
        tool, _ = self._make_tool()
        sub = tool._build_sub_registry(None)
        names = {s["function"]["name"] for s in sub.get_tool_schemas()}
        assert "shell" in names
        assert "delegate" not in names

    def test_sub_registry_with_tool_filter(self):
        tool, _ = self._make_tool()
        sub = tool._build_sub_registry(["shell"])
        names = {s["function"]["name"] for s in sub.get_tool_schemas()}
        assert names == {"shell"}

    @pytest.mark.asyncio
    @patch("loongcli.tools.batch_delegate.Compactor")
    @patch("loongcli.tools.batch_delegate.AgentLoop", FakeAgent)
    async def test_basic_fan_out(self, _mock_compactor):
        tool, _ = self._make_tool()
        result = await tool.execute(tasks=[
            {"prompt": "task A"},
            {"prompt": "task B"},
            {"prompt": "task C"},
        ])
        data = json.loads(result)
        assert data["completed"] == 3
        assert data["failed"] == 0
        assert len(data["results"]) == 3
        for r in data["results"]:
            assert r["status"] == "completed"
            assert "result:" in r["result"]

    @pytest.mark.asyncio
    @patch("loongcli.tools.batch_delegate.Compactor")
    @patch("loongcli.tools.batch_delegate.AgentLoop", FakeAgent)
    async def test_single_task(self, _mock_compactor):
        tool, _ = self._make_tool()
        result = await tool.execute(tasks=[{"prompt": "only one"}])
        data = json.loads(result)
        assert data["completed"] == 1
        assert len(data["results"]) == 1

    @pytest.mark.asyncio
    @patch("loongcli.tools.batch_delegate.Compactor")
    @patch("loongcli.tools.batch_delegate.AgentLoop", FailingAgent)
    async def test_all_fail(self, _mock_compactor):
        tool, _ = self._make_tool()
        result = await tool.execute(tasks=[
            {"prompt": "task A"},
            {"prompt": "task B"},
        ])
        data = json.loads(result)
        assert data["completed"] == 0
        assert data["failed"] == 2
        for r in data["results"]:
            assert r["status"] == "failed"

    @pytest.mark.asyncio
    @patch("loongcli.tools.batch_delegate.Compactor")
    async def test_partial_failure(self, _mock_compactor):
        call_count = 0

        class MixedAgent:
            def __init__(self, **kwargs):
                nonlocal call_count
                self._idx = call_count
                call_count += 1
                self.task = None

            async def run_stream(self, prompt):
                if self._idx == 1:
                    raise RuntimeError("fail")
                yield AgentDone(content=f"ok: {prompt}")

        with patch("loongcli.tools.batch_delegate.AgentLoop", MixedAgent):
            tool, _ = self._make_tool()
            result = await tool.execute(tasks=[
                {"prompt": "good"},
                {"prompt": "bad"},
                {"prompt": "good2"},
            ])

        data = json.loads(result)
        assert data["completed"] == 2
        assert data["failed"] == 1
        assert data["results"][0]["status"] == "completed"
        assert data["results"][1]["status"] == "failed"
        assert data["results"][2]["status"] == "completed"

    @pytest.mark.asyncio
    @patch("loongcli.tools.batch_delegate.Compactor")
    @patch("loongcli.tools.batch_delegate.AgentLoop", SlowAgent)
    async def test_timeout(self, _mock_compactor):
        tool, _ = self._make_tool()
        result = await tool.execute(tasks=[{"prompt": "slow"}], timeout=0.1)
        data = json.loads(result)
        assert data["failed"] == 1
        assert data["results"][0]["status"] == "failed"
        assert "超时" in data["results"][0]["result"]

    @pytest.mark.asyncio
    @patch("loongcli.tools.batch_delegate.Compactor")
    @patch("loongcli.tools.batch_delegate.AgentLoop", FakeAgent)
    async def test_progress_callback(self, _mock_compactor):
        tool, _ = self._make_tool()
        progress_events = []
        tool._progress_callback = lambda evt: progress_events.append(evt)

        await tool.execute(tasks=[
            {"prompt": "task A"},
            {"prompt": "task B"},
        ])
        await asyncio.sleep(0.05)

        assert len(progress_events) == 2
        assert all(isinstance(e, BatchProgress) for e in progress_events)
        totals = {e.total for e in progress_events}
        assert totals == {2}
        completeds = sorted(e.completed for e in progress_events)
        assert completeds == [1, 2]

    @pytest.mark.asyncio
    @patch("loongcli.tools.batch_delegate.Compactor")
    @patch("loongcli.tools.batch_delegate.AgentLoop", FakeAgent)
    async def test_notifications_drained(self, _mock_compactor):
        tool, tm = self._make_tool()
        await tool.execute(tasks=[
            {"prompt": "task A"},
            {"prompt": "task B"},
        ])
        notifs = tm.drain_notifications()
        assert len(notifs) == 0

    @pytest.mark.asyncio
    @patch("loongcli.tools.batch_delegate.Compactor")
    @patch("loongcli.tools.batch_delegate.AgentLoop", FakeAgent)
    async def test_result_preserves_order(self, _mock_compactor):
        tool, _ = self._make_tool()
        prompts = [f"task-{i}" for i in range(5)]
        result = await tool.execute(tasks=[{"prompt": p} for p in prompts])
        data = json.loads(result)
        for i, r in enumerate(data["results"]):
            assert r["index"] == i
            assert r["prompt"] == prompts[i]

    def test_sub_llm_defaults_to_llm(self):
        tool, _ = self._make_tool()
        assert tool._sub_llm is tool._llm

    def test_sub_llm_override(self):
        tm = TaskManager()
        llm = MagicMock()
        sub_llm = MagicMock()
        registry = ToolRegistry()
        security = MagicMock()
        tool = BatchDelegateTool(
            task_manager=tm, llm=llm, parent_registry=registry,
            security=security, sub_llm=sub_llm,
        )
        assert tool._sub_llm is sub_llm
        assert tool._sub_llm is not tool._llm
