from __future__ import annotations

import asyncio
import json
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from loongcli.tools.agent_tool import AgentTool, COORDINATOR_SYSTEM_PROMPT, AUTO_TIMEOUT
from loongcli.tools.routing import AgentRole, SUBAGENT_BLACKLIST, COORDINATOR_ALLOWED
from loongcli.tools.base import ToolRegistry
from loongcli.core.task import TaskManager, Task, TaskStatus


def _make_registry():
    registry = ToolRegistry()
    for name in ["shell", "read_file", "glob", "grep", "write_file", "edit_file",
                 "delegate", "wait_tasks", "stop_task", "send_message", "task_status",
                 "memorize", "plan"]:
        dummy = MagicMock()
        dummy.name = name
        dummy.get_schema.return_value = {
            "type": "function",
            "function": {"name": name, "parameters": {}},
        }
        registry.register(dummy)
    return registry


def _make_tool(depth=0):
    tm = TaskManager()
    llm = MagicMock()
    registry = _make_registry()
    security = MagicMock()
    tool = AgentTool(
        task_manager=tm, llm=llm, parent_registry=registry,
        security=security, depth=depth,
    )
    return tool, tm


# ── Mode tests ──────────────────────────────────────────────────────


class TestBackgroundMode:

    @pytest.mark.asyncio
    async def test_background_returns_immediately(self):
        tool, tm = _make_tool()
        with patch.object(tm, "create_and_run", new_callable=AsyncMock) as mock_run:
            fake_task = Task(prompt="bg")
            fake_task._async_task = asyncio.create_task(asyncio.sleep(100))
            mock_run.return_value = fake_task

            result = await tool.execute(prompt="do stuff", mode="background")

            assert "task_id" in result
            assert fake_task.id in result
            assert "后台" in result

            fake_task._async_task.cancel()
            try:
                await fake_task._async_task
            except asyncio.CancelledError:
                pass


class TestAutoMode:

    @pytest.mark.asyncio
    async def test_auto_returns_result_if_fast(self):
        tool, tm = _make_tool()

        async def fast_run(prompt, agent_loop, depth):
            task = Task(prompt=prompt)
            tm.register(task)

            async def quick():
                await asyncio.sleep(0.01)
                task.status = TaskStatus.COMPLETED
                task.result = "fast result"
                tm.push_notification(task)

            task._async_task = asyncio.create_task(quick())
            task._agent_loop = agent_loop
            return task

        tm.create_and_run = fast_run

        result = await tool.execute(prompt="quick task", mode="auto", timeout=5)
        parsed = json.loads(result)
        assert parsed["status"] == "completed"
        assert parsed["result"] == "fast result"

    @pytest.mark.asyncio
    async def test_auto_backgrounds_on_timeout(self):
        tool, tm = _make_tool()

        async def slow_run(prompt, agent_loop, depth):
            task = Task(prompt=prompt)
            tm.register(task)
            task._async_task = asyncio.create_task(asyncio.sleep(100))
            task._agent_loop = agent_loop
            return task

        tm.create_and_run = slow_run

        result = await tool.execute(prompt="slow task", mode="auto", timeout=0.05)
        assert "自动转后台" in result
        assert "task_id" in result

        # Cleanup
        for t in tm._tasks.values():
            if t._async_task and not t._async_task.done():
                t._async_task.cancel()
                try:
                    await t._async_task
                except asyncio.CancelledError:
                    pass


class TestSyncMode:

    @pytest.mark.asyncio
    async def test_sync_returns_result(self):
        tool, tm = _make_tool()

        async def fast_run(prompt, agent_loop, depth):
            task = Task(prompt=prompt)
            tm.register(task)

            async def quick():
                await asyncio.sleep(0.01)
                task.status = TaskStatus.COMPLETED
                task.result = "sync done"

            task._async_task = asyncio.create_task(quick())
            task._agent_loop = agent_loop
            return task

        tm.create_and_run = fast_run

        result = await tool.execute(prompt="sync task", mode="sync", timeout=5)
        parsed = json.loads(result)
        assert parsed["status"] == "completed"

    @pytest.mark.asyncio
    async def test_sync_timeout_returns_timeout_status(self):
        tool, tm = _make_tool()

        async def slow_run(prompt, agent_loop, depth):
            task = Task(prompt=prompt)
            tm.register(task)
            task._async_task = asyncio.create_task(asyncio.sleep(100))
            task._agent_loop = agent_loop
            return task

        tm.create_and_run = slow_run

        result = await tool.execute(prompt="slow", mode="sync", timeout=0.05)
        parsed = json.loads(result)
        assert parsed["status"] == "timeout"

        for t in tm._tasks.values():
            if t._async_task and not t._async_task.done():
                t._async_task.cancel()
                try:
                    await t._async_task
                except asyncio.CancelledError:
                    pass


# ── Coordinator routing ─────────────────────────────────────────────


class TestCoordinatorRouting:

    def test_coordinator_registry_includes_coord_tools(self):
        tool, _ = _make_tool()
        reg = tool._build_coordinator_registry()
        names = {s["function"]["name"] for s in reg.get_tool_schemas()}
        assert "delegate" in names
        assert "wait_tasks" in names
        assert "stop_task" in names
        assert "send_message" in names
        assert "task_status" in names
        assert "read_file" in names
        assert "glob" in names
        assert "grep" in names

    def test_coordinator_registry_excludes_write_tools(self):
        tool, _ = _make_tool()
        reg = tool._build_coordinator_registry()
        names = {s["function"]["name"] for s in reg.get_tool_schemas()}
        assert "write_file" not in names
        assert "edit_file" not in names
        assert "shell" not in names
        assert "memorize" not in names
        assert "plan" not in names

    def test_subagent_registry_excludes_coord_tools(self):
        tool, _ = _make_tool()
        reg = tool._build_sub_registry(None)
        names = {s["function"]["name"] for s in reg.get_tool_schemas()}
        assert "delegate" not in names
        assert "wait_tasks" not in names
        assert "stop_task" not in names
        assert "send_message" not in names
        assert "task_status" not in names
        assert "memorize" not in names

    def test_subagent_registry_keeps_execution_tools(self):
        tool, _ = _make_tool()
        reg = tool._build_sub_registry(None)
        names = {s["function"]["name"] for s in reg.get_tool_schemas()}
        assert "shell" in names
        assert "read_file" in names
        assert "write_file" in names
        assert "edit_file" in names
        assert "glob" in names
        assert "grep" in names


class TestCoordinatorPrompt:

    def test_coordinator_prompt_has_key_sections(self):
        assert "协调者" in COORDINATOR_SYSTEM_PROMPT
        assert "四阶段" in COORDINATOR_SYSTEM_PROMPT
        assert "调研" in COORDINATOR_SYSTEM_PROMPT
        assert "合成" in COORDINATOR_SYSTEM_PROMPT
        assert "实现" in COORDINATOR_SYSTEM_PROMPT
        assert "验证" in COORDINATOR_SYSTEM_PROMPT
        assert "并行" in COORDINATOR_SYSTEM_PROMPT
        assert "stop_task" in COORDINATOR_SYSTEM_PROMPT

    def test_coordinator_prompt_forbids_direct_execution(self):
        assert "不直接执行" in COORDINATOR_SYSTEM_PROMPT
        assert "不写文件" in COORDINATOR_SYSTEM_PROMPT


class TestDepthLimit:

    @pytest.mark.asyncio
    async def test_depth_2_can_still_delegate(self):
        """Coordinator at depth 1 can spawn worker at depth 2 (MAX_DEPTH=3)."""
        tool, tm = _make_tool(depth=1)
        # depth+1 = 2 < MAX_DEPTH(3), should NOT be rejected
        assert tool._depth + 1 < TaskManager.MAX_DEPTH

    @pytest.mark.asyncio
    async def test_depth_at_max_rejected(self):
        tool, tm = _make_tool(depth=TaskManager.MAX_DEPTH - 1)
        result = await tool.execute(prompt="test")
        assert "最大嵌套深度" in result
