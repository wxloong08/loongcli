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
        assert "shell" in PLAN_MODE_TOOLS          # 只读：执行层按白名单拦（2026-11-19 替代 glob/grep/lsp_*）
        assert "plan" in PLAN_MODE_TOOLS
        assert "exit_plan_mode" in PLAN_MODE_TOOLS
        assert "write_file" not in PLAN_MODE_TOOLS
        assert "edit_file" not in PLAN_MODE_TOOLS

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
        # 注入只带恒定信息（计划 id + 去哪儿看），刻意不带 format_summary()——
        # 它在 messages[0]，随进度改写会废掉整个前缀缓存。全貌走 update_step 返回值。
        assert plan.id in injection
        assert "update_step" in injection
        assert "plan get" in injection
        assert "do thing" not in injection

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

    def test_goal_injection_set_and_clear(self):
        """/goal 执行期目标原话 + 审计规则常驻系统提示（系统消息不参与 compact，目标不会丢）；结束后撤掉。"""
        agent = _make_agent()
        agent._system_prompt_builder = lambda: "base prompt"
        agent.set_goal("实施 P1 和 P2，P2 做完就停", "plan123")
        content = agent.messages[0]["content"]
        assert "实施 P1 和 P2，P2 做完就停" in content
        assert "plan123" in content
        assert "局部验证通过不等于目标完成" in content
        assert agent._active_plan_id == "plan123"

        agent.clear_goal()
        content = agent.messages[0]["content"]
        assert "实施 P1 和 P2" not in content
        assert agent._active_plan_id is None

    def test_goal_injection_has_no_progress(self, tmp_path):
        """目标注入期间恒定：不带步骤进度，否则每完成一步改写 messages[0]、废掉前缀缓存。"""
        agent = _make_agent()
        store = PlanStore(base_dir=tmp_path / "plans")
        plan = Plan(title="T", steps=[PlanStep(index=0, description="do thing", criteria=["c"])])
        store.save(plan)
        agent.plan_store = store
        agent._system_prompt_builder = lambda: "base"
        agent.set_goal("目标", plan.id)
        before = agent._build_plan_injection()
        plan.steps[0].status = "completed"
        store.save(plan)
        assert agent._build_plan_injection() == before
        assert "do thing" not in before


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


# ── 活跃计划注入的前缀稳定性 ───────────────────────


class TestActivePlanInjectionCacheStability:
    """计划推进不得改写 messages[0]——前缀缓存的承重不变量。

    _rebuild_system_prompt 在计划执行期每轮触发。若活跃计划段携带
    format_summary()，每完成一步 messages[0] 就变一次，前缀缓存连同其后的
    全部对话历史一起作废。计划全貌改由 plan update_step 的返回值携带。
    """

    def _store_with_plan(self, tmp_path):
        store = PlanStore(base_dir=tmp_path / "plans")
        plan = Plan(
            title="Test", status="active",
            steps=[PlanStep(index=i, description=f"step {i}") for i in range(3)],
        )
        store.save(plan)
        return store, plan

    def test_injection_stable_across_step_progress(self, tmp_path):
        from loongcli.tools.plan_tool import PlanTool

        store, plan = self._store_with_plan(tmp_path)
        agent = _make_agent()
        agent.plan_store = store
        agent._active_plan_id = plan.id

        before = agent._build_plan_injection()
        assert plan.id in before

        tool = PlanTool(store)
        asyncio.run(tool.execute(operation="update_step", plan_id=plan.id,
                                 step=1, step_status="completed"))

        after = agent._build_plan_injection()
        assert after == before, "计划推进后活跃计划注入必须字节不变"
        assert "step 0" not in after, "注入不得携带步骤全文（那会随进度变化）"

    def test_update_step_result_carries_full_plan(self, tmp_path):
        """全貌改走工具返回值——落在对话尾部，走缓存，零额外代价。"""
        from loongcli.tools.plan_tool import PlanTool

        store, plan = self._store_with_plan(tmp_path)
        tool = PlanTool(store)
        result = asyncio.run(tool.execute(operation="update_step", plan_id=plan.id,
                                          step=2, step_status="completed"))
        assert "步骤 2 已更新" in result
        assert "step 0" in result and "step 2" in result, "返回值应含完整计划"
        assert "1/3" in result


def test_plan_mode_injection_design_guidance():
    # 第 23.2 项：方案四块；调研深度随任务（删掉一刀切的「一两次调用足矣」）；判据查内容不查格式
    for word in ("approach", "背景与范围", "放弃了什么", "需要用户拍板", "说出理由", "查内容不查格式", "章节数"):
        assert word in PLAN_MODE_SYSTEM_INJECTION, word
    assert "简单任务一两次调用足矣" not in PLAN_MODE_SYSTEM_INJECTION


def test_wrap_up_gives_next_steps():
    # 第 23.4 项：收尾给 1-2 条下一步建议（用户：Manus 做完一步会引导下一步做什么）
    from loongcli.core.agent import GOAL_INJECTION_TEMPLATE
    assert "下一步建议" in GOAL_INJECTION_TEMPLATE and "不是核对表" in GOAL_INJECTION_TEMPLATE
    assert "下一步建议" in ACTIVE_PLAN_INJECTION_TEMPLATE
