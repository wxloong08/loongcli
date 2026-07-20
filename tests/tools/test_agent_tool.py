import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from loongcli.tools.agent_tool import AgentTool
from loongcli.tools.routing import SUBAGENT_BLACKLIST
from loongcli.tools.send_message import SendMessageTool
from loongcli.tools.task_status import TaskStatusTool
from loongcli.tools.base import ToolRegistry
from loongcli.core.task import TaskManager, Task, TaskStatus


class TestAgentTool:
    def _make_tool(self, depth=0):
        tm = TaskManager()
        llm = MagicMock()
        registry = ToolRegistry()

        dummy = MagicMock()
        dummy.name = "shell"
        dummy.get_schema.return_value = {
            "type": "function",
            "function": {"name": "shell", "parameters": {}},
        }
        registry.register(dummy)

        delegate_dummy = MagicMock()
        delegate_dummy.name = "delegate"
        delegate_dummy.get_schema.return_value = {
            "type": "function",
            "function": {"name": "delegate", "parameters": {}},
        }
        registry.register(delegate_dummy)

        security = MagicMock()
        tool = AgentTool(
            task_manager=tm,
            llm=llm,
            parent_registry=registry,
            security=security,
            depth=depth,
        )
        return tool, tm

    @pytest.mark.asyncio
    async def test_max_depth_rejected(self):
        tool, tm = self._make_tool(depth=TaskManager.MAX_DEPTH - 1)
        result = await tool.execute(prompt="test")
        assert "最大嵌套深度" in result

    @pytest.mark.asyncio
    async def test_subagent_inherits_parent_checker_noninteractive(self):
        """SubAgents must reuse the parent's permission checker (not a blanket
        SKIP) and run non-interactively, so they can't bypass confirmation."""
        tool, tm = self._make_tool()
        captured = {}

        class FakeLoop:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        async def fake_create_and_run(prompt, agent_loop, depth, parent_id=None):
            return Task(prompt=prompt)

        with patch("loongcli.tools.agent_tool.AgentLoop", FakeLoop):
            tm.create_and_run = fake_create_and_run
            await tool.execute(prompt="do it", mode="background")

        assert captured["interactive"] is False
        assert captured["permission_checker"] is tool._permission_checker

    @pytest.mark.asyncio
    async def test_hook_manager_forwarded_to_subagent(self):
        """hook 当安全闸用时子代理不能是旁路——注入的 hook_manager 进 AgentServices。"""
        tm = TaskManager()
        registry = ToolRegistry()
        sentinel = object()
        tool = AgentTool(
            task_manager=tm,
            llm=MagicMock(),
            parent_registry=registry,
            security=MagicMock(),
            hook_manager=sentinel,
        )
        captured = {}

        class FakeLoop:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        async def fake_create_and_run(prompt, agent_loop, depth, parent_id=None):
            return Task(prompt=prompt)

        with patch("loongcli.tools.agent_tool.AgentLoop", FakeLoop):
            tm.create_and_run = fake_create_and_run
            await tool.execute(prompt="do it", mode="background")

        assert captured["services"].hook_manager is sentinel

    def test_build_sub_registry_excludes_blacklisted(self):
        tool, _ = self._make_tool()
        sub = tool._build_sub_registry(None)
        names = {s["function"]["name"] for s in sub.get_tool_schemas()}
        assert "shell" in names
        assert "delegate" not in names

    def test_build_sub_registry_with_allowed_filter(self):
        tool, _ = self._make_tool()
        sub = tool._build_sub_registry(["shell"])
        names = {s["function"]["name"] for s in sub.get_tool_schemas()}
        assert names == {"shell"}

    def test_sub_llm_defaults_to_llm(self):
        tool, _ = self._make_tool()
        assert tool._sub_llm is tool._llm

    def test_sub_llm_override(self):
        tm = TaskManager()
        llm = MagicMock()
        sub_llm = MagicMock()
        registry = ToolRegistry()
        security = MagicMock()
        tool = AgentTool(
            task_manager=tm, llm=llm, parent_registry=registry,
            security=security, sub_llm=sub_llm,
        )
        assert tool._sub_llm is sub_llm
        assert tool._sub_llm is not tool._llm

    def test_tool_schema(self):
        tool, _ = self._make_tool()
        reg = ToolRegistry()
        reg.register(tool)
        schemas = reg.get_tool_schemas()
        assert len(schemas) == 1
        assert schemas[0]["function"]["name"] == "delegate"
        assert "prompt" in schemas[0]["function"]["parameters"]["properties"]


class TestSendMessageTool:
    @pytest.mark.asyncio
    async def test_send_message(self):
        tm = TaskManager()
        task = Task(prompt="test")
        tm.register(task)
        tool = SendMessageTool(tm)
        result = await tool.execute(task_id=task.id, message="hello")
        assert "已送达" in result
        assert task.mailbox == ["hello"]


class TestTaskStatusTool:
    @pytest.mark.asyncio
    async def test_status(self):
        tm = TaskManager()
        task = Task(prompt="check")
        tm.register(task)
        tool = TaskStatusTool(tm)
        result = await tool.execute(task_id=task.id)
        assert "running" in result
        assert task.id in result

    @pytest.mark.asyncio
    async def test_big_result_capped_with_trace_pointer(self):
        # 自管预算：入口豁免后无人兜底，超长结果须在这里截断并附 trace 指针
        # （入口一刀切时重调本工具会返回同样的截断——死循环陷阱）
        import json
        from loongcli.core.task import SINGLE_TASK_RESULT_CAP

        tm = TaskManager()
        task = Task(prompt="big")
        task.status = TaskStatus.COMPLETED
        task.result = "调研数据行\n" * 5000  # 30K chars
        loop = MagicMock()
        loop.conversation_store = MagicMock(
            base_dir="/tmp/tasks", session_id="cap000000001")
        task._agent_loop = loop
        tm.register(task)

        tool = TaskStatusTool(tm)
        info = json.loads(await tool.execute(task_id=task.id))
        assert len(info["result"]) <= SINGLE_TASK_RESULT_CAP + 200
        assert "结果已截断" in info["result"]
        assert "cap000000001.json" in info["result"]
        assert "重新调用工具" not in info["result"]
