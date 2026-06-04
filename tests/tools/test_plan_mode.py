from __future__ import annotations

import asyncio
import json
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from pathlib import Path

from loongcli.core.agent import AgentLoop, PLAN_MODE_TOOLS, PLAN_MODE_SYSTEM_INJECTION, ACTIVE_PLAN_INJECTION_TEMPLATE
from loongcli.core.events import PlanApproval
from loongcli.tools.enter_plan_mode import EnterPlanModeTool
from loongcli.tools.exit_plan_mode import ExitPlanModeTool
from loongcli.tools.routing import AgentRole, SUBAGENT_BLACKLIST
from loongcli.tools.base import ToolRegistry
from loongcli.plan.store import PlanStore, Plan, PlanStep
from loongcli.security.permissions import PermissionChecker, PermissionMode


def _make_agent(**overrides):
    llm = MagicMock()
    llm.model = "deepseek-v4-flash"
    llm.chat_stream = AsyncMock()
    registry = ToolRegistry()
    for name in ["read_file", "write_file", "edit_file", "shell", "glob", "grep",
                 "plan", "enter_plan_mode", "exit_plan_mode"]:
        dummy = MagicMock()
        dummy.name = name
        dummy.get_schema = MagicMock(return_value={
            "type": "function",
            "function": {"name": name, "parameters": {}},
        })
        registry.register(dummy)

    perm = PermissionChecker(PermissionMode.SKIP)
    agent = AgentLoop(
        llm=llm,
        tool_registry=registry,
        permission_checker=perm,
        system_prompt="test system prompt",
        **overrides,
    )
    return agent


# ── EnterPlanModeTool ──────────────────────────────────────────────


class TestEnterPlanMode:

    @pytest.mark.asyncio
    async def test_enter_plan_mode(self):
        agent = _make_agent()
        tool = EnterPlanModeTool()
        tool.bind_agent(agent)

        assert not agent.plan_mode
        result = await tool.execute()
        assert agent.plan_mode
        assert "规划模式" in result

    @pytest.mark.asyncio
    async def test_already_in_plan_mode(self):
        agent = _make_agent()
        tool = EnterPlanModeTool()
        tool.bind_agent(agent)

        agent.enter_plan_mode()
        result = await tool.execute()
        assert "已经处于规划模式" in result

    @pytest.mark.asyncio
    async def test_subagent_cannot_enter(self):
        agent = _make_agent(role=AgentRole.SUBAGENT)
        tool = EnterPlanModeTool()
        tool.bind_agent(agent)

        result = await tool.execute()
        assert "SubAgent" in result
        assert not agent.plan_mode

    @pytest.mark.asyncio
    async def test_no_agent_bound(self):
        tool = EnterPlanModeTool()
        result = await tool.execute()
        assert "未绑定" in result


# ── ExitPlanModeTool ───────────────────────────────────────────────


class TestExitPlanMode:

    def _make_store(self, tmp_path: Path) -> PlanStore:
        return PlanStore(base_dir=tmp_path / "plans")

    @pytest.mark.asyncio
    async def test_exit_returns_approval_marker(self, tmp_path):
        agent = _make_agent()
        agent.enter_plan_mode()
        store = self._make_store(tmp_path)
        plan = Plan(title="Test", steps=[PlanStep(index=0, description="step1")])
        store.save(plan)

        tool = ExitPlanModeTool()
        tool.bind_agent(agent)
        tool.bind_plan_store(store)

        result = await tool.execute(plan_id=plan.id)
        data = json.loads(result)
        assert data["__plan_approval__"] is True
        assert data["plan_id"] == plan.id
        assert "Test" in data["plan_summary"]

    @pytest.mark.asyncio
    async def test_not_in_plan_mode(self, tmp_path):
        agent = _make_agent()
        store = self._make_store(tmp_path)
        tool = ExitPlanModeTool()
        tool.bind_agent(agent)
        tool.bind_plan_store(store)

        result = await tool.execute(plan_id="xxx")
        assert "不在规划模式" in result

    @pytest.mark.asyncio
    async def test_plan_not_found(self, tmp_path):
        agent = _make_agent()
        agent.enter_plan_mode()
        store = self._make_store(tmp_path)
        tool = ExitPlanModeTool()
        tool.bind_agent(agent)
        tool.bind_plan_store(store)

        result = await tool.execute(plan_id="nonexistent")
        assert "未找到" in result

    @pytest.mark.asyncio
    async def test_empty_steps_rejected(self, tmp_path):
        agent = _make_agent()
        agent.enter_plan_mode()
        store = self._make_store(tmp_path)
        plan = Plan(title="Empty", steps=[])
        store.save(plan)

        tool = ExitPlanModeTool()
        tool.bind_agent(agent)
        tool.bind_plan_store(store)

        result = await tool.execute(plan_id=plan.id)
        assert "没有步骤" in result


# ── AgentLoop Plan Mode State ─────────────────────────────────────


class TestAgentPlanState:

    def test_initial_state(self):
        agent = _make_agent()
        assert not agent.plan_mode
        assert agent._active_plan_id is None

    def test_enter_exit(self):
        agent = _make_agent()
        agent.enter_plan_mode()
        assert agent.plan_mode

        agent.exit_plan_mode(plan_id="test-123")
        assert not agent.plan_mode
        assert agent._active_plan_id == "test-123"

    def test_exit_without_plan(self):
        agent = _make_agent()
        agent.enter_plan_mode()
        agent.exit_plan_mode()
        assert not agent.plan_mode
        assert agent._active_plan_id is None


# ── Tool Filtering ────────────────────────────────────────────────


class TestPlanModeToolFiltering:

    def test_plan_mode_tools_constant(self):
        assert "read_file" in PLAN_MODE_TOOLS
        assert "glob" in PLAN_MODE_TOOLS
        assert "grep" in PLAN_MODE_TOOLS
        assert "plan" in PLAN_MODE_TOOLS
        assert "exit_plan_mode" in PLAN_MODE_TOOLS
        assert "write_file" not in PLAN_MODE_TOOLS
        assert "edit_file" not in PLAN_MODE_TOOLS
        assert "shell" not in PLAN_MODE_TOOLS

    def test_normal_mode_all_tools(self):
        agent = _make_agent()
        schemas = agent.tool_registry.get_tool_schemas(role=agent.role)
        names = {s["function"]["name"] for s in schemas}
        assert "write_file" in names
        assert "shell" in names

    def test_plan_mode_filters_schemas(self):
        agent = _make_agent()
        agent.enter_plan_mode()
        schemas = agent.tool_registry.get_tool_schemas(role=agent.role)
        filtered = [s for s in schemas if s["function"]["name"] in PLAN_MODE_TOOLS]
        assert len(filtered) <= len(PLAN_MODE_TOOLS)
        for s in filtered:
            assert s["function"]["name"] in PLAN_MODE_TOOLS


# ── System Prompt Injection ───────────────────────────────────────


class TestPlanPromptInjection:

    def test_plan_mode_injection(self):
        agent = _make_agent()
        agent.enter_plan_mode()
        injection = agent._build_plan_injection()
        assert "规划模式" in injection
        assert "read_file" in injection

    def test_active_plan_injection(self, tmp_path):
        agent = _make_agent()
        store = PlanStore(base_dir=tmp_path / "plans")
        plan = Plan(title="My Plan", steps=[
            PlanStep(index=0, description="do thing"),
        ])
        store.save(plan)

        agent.plan_store = store
        agent._active_plan_id = plan.id

        injection = agent._build_plan_injection()
        assert "My Plan" in injection
        assert "do thing" in injection
        assert "update_step" in injection

    def test_no_injection_normal_mode(self):
        agent = _make_agent()
        injection = agent._build_plan_injection()
        assert injection == ""

    def test_rebuild_includes_plan_injection(self):
        builder_called = []

        def mock_builder():
            builder_called.append(1)
            return "base prompt"

        agent = _make_agent()
        agent._system_prompt_builder = mock_builder
        agent.enter_plan_mode()
        agent._rebuild_system_prompt()

        content = agent.messages[0]["content"]
        assert "base prompt" in content
        assert "规划模式" in content


# ── Routing Blacklist ─────────────────────────────────────────────


class TestPlanModeRouting:

    def test_plan_mode_tools_in_subagent_blacklist(self):
        assert "enter_plan_mode" in SUBAGENT_BLACKLIST
        assert "exit_plan_mode" in SUBAGENT_BLACKLIST


# ── PlanApproval Event ────────────────────────────────────────────


class TestPlanApprovalEvent:

    def test_event_fields(self):
        loop = asyncio.new_event_loop()
        future = loop.create_future()
        event = PlanApproval(
            plan_id="test",
            plan_summary="my plan",
            future=future,
        )
        assert event.plan_id == "test"
        assert event.plan_summary == "my plan"
        loop.close()


# ── /plan command ─────────────────────────────────────────────────


class TestPlanCommand:

    @pytest.mark.asyncio
    async def test_no_args_shows_usage(self):
        from loongcli.tui.commands import PlanCommand, CommandContext
        from rich.console import Console
        from io import StringIO

        console = Console(file=StringIO(), force_terminal=True)
        agent = MagicMock()
        agent.plan_mode = False
        agent._active_plan_id = None
        agent.plan_store = None
        tui = MagicMock()
        ctx = CommandContext(console=console, agent=agent, memory=None, registry=MagicMock(), tui=tui)

        cmd = PlanCommand()
        await cmd.run([], ctx)
        output = console.file.getvalue()
        assert "用法" in output

    @pytest.mark.asyncio
    async def test_no_args_in_plan_mode(self):
        from loongcli.tui.commands import PlanCommand, CommandContext
        from rich.console import Console
        from io import StringIO

        console = Console(file=StringIO(), force_terminal=True)
        agent = MagicMock()
        agent.plan_mode = True
        tui = MagicMock()
        ctx = CommandContext(console=console, agent=agent, memory=None, registry=MagicMock(), tui=tui)

        cmd = PlanCommand()
        await cmd.run([], ctx)
        output = console.file.getvalue()
        assert "规划模式" in output

    @pytest.mark.asyncio
    async def test_with_args_enters_plan_mode(self):
        from loongcli.tui.commands import PlanCommand, CommandContext
        from rich.console import Console
        from io import StringIO

        console = Console(file=StringIO(), force_terminal=True)
        agent = MagicMock()
        agent.plan_mode = False
        agent._active_plan_id = None
        agent.plan_store = None
        tui = MagicMock()
        tui._handle_agent_response = AsyncMock()
        ctx = CommandContext(console=console, agent=agent, memory=None, registry=MagicMock(), tui=tui)

        cmd = PlanCommand()
        await cmd.run(["refactor", "auth"], ctx)

        agent.enter_plan_mode.assert_called_once()
        tui._handle_agent_response.assert_called_once()
        call_args = tui._handle_agent_response.call_args
        assert "refactor auth" in call_args[0][1]

    @pytest.mark.asyncio
    async def test_no_args_shows_active_plan(self):
        from loongcli.tui.commands import PlanCommand, CommandContext
        from rich.console import Console
        from io import StringIO

        console = Console(file=StringIO(), force_terminal=True)
        agent = MagicMock()
        agent.plan_mode = False
        agent._active_plan_id = "abc123"
        mock_plan = MagicMock()
        mock_plan.format_summary.return_value = "**Test Plan** (0/1 完成)\n  ○ 0. step one"
        agent.plan_store = MagicMock()
        agent.plan_store.load.return_value = mock_plan
        tui = MagicMock()
        ctx = CommandContext(console=console, agent=agent, memory=None, registry=MagicMock(), tui=tui)

        cmd = PlanCommand()
        await cmd.run([], ctx)
        output = console.file.getvalue()
        assert "活跃计划" in output
