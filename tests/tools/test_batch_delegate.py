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
        # 合成纪律标记：失败=结论不可依赖，确定性文本挂在失败发生点
        assert "结论不可依赖" in data["warning"]
        assert "禁止用假设" in data["warning"]

    @pytest.mark.asyncio
    @patch("loongcli.tools.batch_delegate.Compactor")
    @patch("loongcli.tools.batch_delegate.AgentLoop", SlowAgent)
    async def test_stalled_subagent_counted_failed(self, _mock_compactor):
        # 探测窗口内零进展 = 真卡住 → 失败（只有这种超时算失败）
        tool, _ = self._make_tool()
        tool.PROBE_INITIAL = 0.05
        result = await tool.execute(tasks=[{"prompt": "slow"}], timeout=0.05)
        data = json.loads(result)
        assert data["failed"] == 1
        assert data["timed_out"] == 0
        assert data["results"][0]["status"] == "failed"
        assert "零进展" in data["results"][0]["result"]
        assert "失败或卡死" in data["warning"]

    @pytest.mark.asyncio
    @patch("loongcli.tools.batch_delegate.Compactor")
    async def test_progressing_subagent_extended_then_timeout(self, _mock_compactor):
        # 仍在推进（消息数增长）的任务不被当卡死杀掉，续命到硬顶后标 timeout 非失败
        class BusyAgent:
            def __init__(self, **kwargs):
                self.task = None
                self.messages = []
                self.conversation_store = MagicMock(
                    base_dir="/tmp/tasks", session_id="busy00000001")

            async def run_stream(self, prompt):
                while True:
                    await asyncio.sleep(0.01)
                    self.messages.append(
                        {"role": "assistant", "content": f"step {len(self.messages)}"})
                yield AgentDone(content="unreachable")  # pragma: no cover

        with patch("loongcli.tools.batch_delegate.AgentLoop", BusyAgent):
            tool, _ = self._make_tool()
            tool.PROBE_INITIAL = 0.03
            result = await tool.execute(tasks=[{"prompt": "busy"}], timeout=0.05)

        data = json.loads(result)
        assert data["timed_out"] == 1
        assert data["failed"] == 0
        assert data["results"][0]["status"] == "timeout"
        assert "硬顶" in data["results"][0]["result"]
        assert "超时不是失败" in data["warning"]

    @pytest.mark.asyncio
    @patch("loongcli.tools.batch_delegate.Compactor")
    async def test_timeout_salvages_partial_work(self, _mock_compactor):
        # 打捞：取消前的工具调用清单（含 URL）+ 最后输出 + trace 路径必须回传，
        # 不再颗粒无收（gomami 事故：8 篇已读测评帖被"超时取消"四字吞掉）
        class SlowAgentWithWork:
            def __init__(self, **kwargs):
                self.task = None
                self.messages = [
                    {"role": "user", "content": "调研"},
                    {"role": "assistant", "content": "", "tool_calls": [{
                        "id": "t1", "type": "function",
                        "function": {"name": "url_read",
                                     "arguments": '{"url": "https://nodeseek.com/post-1"}'},
                    }]},
                    {"role": "tool", "tool_call_id": "t1", "content": "帖子正文"},
                    {"role": "assistant", "content": "已读完第一篇，正在继续"},
                ]
                self.conversation_store = MagicMock(
                    base_dir="/tmp/tasks", session_id="abc123def456")

            async def run_stream(self, prompt):
                await asyncio.sleep(10)
                yield AgentDone(content="unreachable")

        with patch("loongcli.tools.batch_delegate.AgentLoop", SlowAgentWithWork):
            tool, _ = self._make_tool()
            tool.PROBE_INITIAL = 0.05
            result = await tool.execute(tasks=[{"prompt": "slow"}], timeout=0.05)

        data = json.loads(result)
        salvaged = data["results"][0]["result"]  # 静态消息无增长 → 卡死路径，成果仍打捞
        assert data["results"][0]["status"] == "failed"
        assert "url_read" in salvaged
        assert "https://nodeseek.com/post-1" in salvaged
        assert "已读完第一篇" in salvaged
        assert "abc123def456.json" in salvaged  # trace 路径指引

    @pytest.mark.asyncio
    @patch("loongcli.tools.batch_delegate.Compactor")
    @patch("loongcli.tools.batch_delegate.AgentLoop", FakeAgent)
    async def test_no_warning_when_all_completed(self, _mock_compactor):
        tool, _ = self._make_tool()
        result = await tool.execute(tasks=[{"prompt": "a"}, {"prompt": "b"}])
        data = json.loads(result)
        assert "warning" not in data
        assert data["timed_out"] == 0

    def test_subagent_prompt_has_data_discipline(self):
        from loongcli.tools.agent_tool import SUBAGENT_SYSTEM_PROMPT
        assert "工具结果为准" in SUBAGENT_SYSTEM_PROMPT
        assert "未知" in SUBAGENT_SYSTEM_PROMPT

    def test_coordinator_prompt_has_synthesis_discipline(self):
        from loongcli.tools.agent_tool import COORDINATOR_SYSTEM_PROMPT
        assert "合成纪律" in COORDINATOR_SYSTEM_PROMPT
        assert "不要用你的印象或典型值补齐" in COORDINATOR_SYSTEM_PROMPT

    @pytest.mark.asyncio
    @patch("loongcli.tools.batch_delegate.Compactor")
    async def test_hook_manager_forwarded_to_subagent(self, _mock_compactor):
        # hook 当安全闸用时子代理不能是旁路——注入的 hook_manager 必须进 AgentServices
        captured = {}

        class CapturingAgent(FakeAgent):
            def __init__(self, **kwargs):
                captured.update(kwargs)
                super().__init__(**kwargs)

        sentinel = object()
        with patch("loongcli.tools.batch_delegate.AgentLoop", CapturingAgent):
            tm = TaskManager()
            registry = ToolRegistry()
            tool = BatchDelegateTool(
                task_manager=tm,
                llm=MagicMock(),
                parent_registry=registry,
                security=MagicMock(),
                hook_manager=sentinel,
            )
            await tool.execute(tasks=[{"prompt": "x"}])

        assert captured["services"].hook_manager is sentinel

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

    @pytest.mark.asyncio
    @patch("loongcli.tools.batch_delegate.Compactor")
    async def test_big_result_single_task_mostly_preserved(self, _mock_compactor):
        # 真实事故复现防护：37K 合成结果曾被入口截成 2K。单任务预算 36000，
        # 40K 结果须保留 ≥30K（而非 2000 预览），截断提示附 trace 指针、无误导文案。
        from loongcli.core.task import BATCH_RESULT_BUDGET

        class BigResultAgent(FakeAgent):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.conversation_store = MagicMock(
                    base_dir="/tmp/tasks", session_id="big000000001")

            async def run_stream(self, prompt):
                yield AgentDone(content="调研发现一行\n" * 6000)  # 42K chars

        with patch("loongcli.tools.batch_delegate.AgentLoop", BigResultAgent):
            tool, _ = self._make_tool()
            result = await tool.execute(tasks=[{"prompt": "深度调研"}])

        data = json.loads(result)
        r = data["results"][0]["result"]
        assert len(r) >= BATCH_RESULT_BUDGET * 0.8
        assert "结果已截断" in r
        assert "big000000001.json" in r
        assert "重新调用工具" not in r

    @pytest.mark.asyncio
    @patch("loongcli.tools.batch_delegate.Compactor")
    async def test_big_results_split_budget_across_tasks(self, _mock_compactor):
        # 3 任务均摊 36000 预算：各保留 ~12K，总量有界且每路都有实质内容
        class BigResultAgent(FakeAgent):
            async def run_stream(self, prompt):
                yield AgentDone(content=f"{prompt}-数据\n" * 3000)  # ~20K each

        with patch("loongcli.tools.batch_delegate.AgentLoop", BigResultAgent):
            tool, _ = self._make_tool()
            result = await tool.execute(tasks=[
                {"prompt": "调研A"}, {"prompt": "调研B"}, {"prompt": "调研C"},
            ])

        data = json.loads(result)
        assert data["completed"] == 3
        for i, entry in enumerate(data["results"]):
            r = entry["result"]
            assert 10000 <= len(r) <= 13000  # per_cap=12000 + 截断提示
            assert f"调研{'ABC'[i]}-数据" in r
            assert "结果已截断" in r

    @pytest.mark.asyncio
    @patch("loongcli.tools.batch_delegate.Compactor")
    @patch("loongcli.tools.batch_delegate.AgentLoop", FakeAgent)
    async def test_long_prompt_echo_trimmed(self, _mock_compactor):
        # prompt 回显是主代理自己上一条消息的复制品，只留 200 字符摘要
        long_prompt = "你是深度调研系统的调研员。" * 200  # ~2.4K
        tool, _ = self._make_tool()
        result = await tool.execute(tasks=[{"prompt": long_prompt}])
        data = json.loads(result)
        echoed = data["results"][0]["prompt"]
        assert len(echoed) == 201  # 200 + "…"
        assert echoed.startswith("你是深度调研系统的调研员。")
        assert echoed.endswith("…")

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
