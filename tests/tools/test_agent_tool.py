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
