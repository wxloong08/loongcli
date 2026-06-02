import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from io import StringIO

from rich.console import Console

from loongcli.tui.app import TUI
from loongcli.tui.commands import GoalCommand, CommandContext
from loongcli.plan.store import PlanStore, Plan, PlanStep
from loongcli.core.intent import StopIntent


@pytest.fixture
def plan_store(tmp_path):
    return PlanStore(base_dir=tmp_path / "plans")


@pytest.fixture
def tui(plan_store):
    t = TUI(plan_store=plan_store)
    t.console = Console(file=StringIO(), force_terminal=True)
    return t


def _mock_intent(intent: StopIntent):
    async def _detect(llm, text):
        return intent
    return _detect


class TestGetPlanHint:
    def test_no_plan_store(self):
        t = TUI()
        t.console = Console(file=StringIO(), force_terminal=True)
        hint, completed = t._get_plan_hint()
        assert hint == ""
        assert completed is False

    def test_no_active_plans(self, tui):
        hint, completed = tui._get_plan_hint()
        assert hint == ""
        assert completed is False

    def test_active_plan_with_pending_steps(self, tui, plan_store):
        plan = Plan(title="Test", steps=[
            PlanStep(index=0, description="Step A", status="completed"),
            PlanStep(index=1, description="Step B", status="pending"),
            PlanStep(index=2, description="Step C", status="pending"),
        ])
        plan_store.save(plan)
        hint, completed = tui._get_plan_hint()
        assert "1/3" in hint
        assert "Step B" in hint
        assert completed is False

    def test_all_steps_completed(self, tui, plan_store):
        plan = Plan(title="Test", steps=[
            PlanStep(index=0, description="A", status="completed"),
            PlanStep(index=1, description="B", status="completed"),
        ])
        plan_store.save(plan)
        hint, completed = tui._get_plan_hint()
        assert completed is True

    def test_abandoned_plan(self, tui, plan_store):
        plan = Plan(title="Test", status="abandoned")
        plan_store.save(plan)
        hint, completed = tui._get_plan_hint()
        assert hint == ""
        assert completed is False

    def test_completed_plan_not_active(self, tui, plan_store):
        plan = Plan(title="Test", status="completed")
        plan_store.save(plan)
        hint, completed = tui._get_plan_hint()
        assert completed is True

    def test_in_progress_step_is_next(self, tui, plan_store):
        plan = Plan(title="Test", steps=[
            PlanStep(index=0, description="Done", status="completed"),
            PlanStep(index=1, description="Doing", status="in_progress"),
            PlanStep(index=2, description="Todo", status="pending"),
        ])
        plan_store.save(plan)
        hint, completed = tui._get_plan_hint()
        assert "1/3" in hint
        assert "Doing" in hint
        assert completed is False

    def test_skipped_counts_as_done(self, tui, plan_store):
        plan = Plan(title="Test", steps=[
            PlanStep(index=0, description="A", status="completed"),
            PlanStep(index=1, description="B", status="skipped"),
        ])
        plan_store.save(plan)
        hint, completed = tui._get_plan_hint()
        assert completed is True


class TestGoalPrompts:
    def test_planning_prompt(self, tui):
        prompt = tui._goal_planning_prompt("重构认证模块")
        assert "重构认证模块" in prompt
        assert "plan(create)" in prompt
        assert "只创建计划" in prompt

    def test_replan_prompt(self, tui):
        prompt = tui._goal_replan_prompt("重构", "加一个测试步骤")
        assert "重构" in prompt
        assert "加一个测试步骤" in prompt
        assert "重新创建" in prompt

    def test_execution_prompt(self, tui):
        prompt = tui._goal_execution_prompt("重构", "当前进度: 0/3")
        assert "重构" in prompt
        assert "已确认" in prompt
        assert "0/3" in prompt

    def test_execution_prompt_no_hint(self, tui):
        prompt = tui._goal_execution_prompt("调研方案")
        assert "调研方案" in prompt
        assert "已确认" in prompt

    def test_continue_prompt(self, tui):
        prompt = tui._goal_continue_prompt("目标", "当前进度: 2/5\n下一步: 写测试")
        assert "目标" in prompt
        assert "2/5" in prompt
        assert "继续" in prompt


class TestGoalPlanningPhase:
    @pytest.mark.asyncio
    async def test_plan_created_and_displayed(self, tui, plan_store):
        async def fake_handle(agent, prompt, allowed_tools=None):
            assert allowed_tools == {"plan"}
            plan = Plan(title="Test Plan", steps=[
                PlanStep(index=0, description="Step A"),
                PlanStep(index=1, description="Step B"),
            ])
            plan_store.save(plan)

        tui._handle_agent_response = fake_handle

        with patch.object(tui, '_ask_plan_approval', new_callable=AsyncMock, return_value="n"):
            agent = MagicMock()
            await tui.run_goal(agent, "test goal")

        assert tui._goal_mode is False
        output = tui.console.file.getvalue()
        assert "规划阶段" in output
        assert "计划已取消" in output

    @pytest.mark.asyncio
    async def test_plan_creation_retries(self, tui, plan_store):
        call_count = 0

        async def fake_handle(agent, prompt, allowed_tools=None):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                plan = Plan(title="Retry Plan", steps=[
                    PlanStep(index=0, description="Step"),
                ])
                plan_store.save(plan)

        tui._handle_agent_response = fake_handle

        with patch.object(tui, '_ask_plan_approval', new_callable=AsyncMock, return_value="n"):
            agent = MagicMock()
            await tui.run_goal(agent, "test goal")

        assert call_count == 2
        output = tui.console.file.getvalue()
        assert "重试" in output

    @pytest.mark.asyncio
    async def test_plan_creation_fails(self, tui, plan_store):
        async def fake_handle(agent, prompt, allowed_tools=None):
            pass

        tui._handle_agent_response = fake_handle

        agent = MagicMock()
        await tui.run_goal(agent, "impossible")

        assert tui._goal_mode is False
        output = tui.console.file.getvalue()
        assert "未能创建" in output


class TestGoalApprovalPhase:
    def _setup_plan(self, plan_store):
        plan = Plan(title="Approve Me", steps=[
            PlanStep(index=0, description="Do A"),
            PlanStep(index=1, description="Do B"),
        ])
        plan_store.save(plan)
        return plan

    @pytest.mark.asyncio
    async def test_approve_enters_execution(self, tui, plan_store):
        plan = self._setup_plan(plan_store)
        handle_calls = []

        async def fake_handle(agent, prompt, allowed_tools=None):
            handle_calls.append({"prompt": prompt, "allowed_tools": allowed_tools})
            if allowed_tools is None:
                for s in plan.steps:
                    s.status = "completed"
                plan_store.save(plan)

        tui._handle_agent_response = fake_handle

        approval_calls = [AsyncMock(return_value="")]
        with patch.object(tui, '_ask_plan_approval', side_effect=[""]):
            agent = MagicMock()
            agent.messages = [{"role": "assistant", "content": "done"}]
            with patch("loongcli.tui.app.detect_stop_intent", _mock_intent(StopIntent.COMPLETED)):
                await tui.run_goal(agent, "test goal")

        assert any(c["allowed_tools"] == {"plan"} for c in handle_calls)
        assert any(c["allowed_tools"] is None for c in handle_calls)
        output = tui.console.file.getvalue()
        assert "执行阶段" in output

    @pytest.mark.asyncio
    async def test_cancel_abandons_plan(self, tui, plan_store):
        plan = self._setup_plan(plan_store)

        async def fake_handle(agent, prompt, allowed_tools=None):
            pass

        tui._handle_agent_response = fake_handle

        with patch.object(tui, '_ask_plan_approval', new_callable=AsyncMock, return_value="n"):
            agent = MagicMock()
            await tui.run_goal(agent, "test goal")

        reloaded = plan_store.load(plan.id)
        assert reloaded.status == "abandoned"
        output = tui.console.file.getvalue()
        assert "计划已取消" in output

    @pytest.mark.asyncio
    async def test_feedback_triggers_replan(self, tui, plan_store):
        original = self._setup_plan(plan_store)
        replan_called = False

        async def fake_handle(agent, prompt, allowed_tools=None):
            nonlocal replan_called
            if "反馈" in prompt:
                replan_called = True
                new_plan = Plan(title="Revised", steps=[
                    PlanStep(index=0, description="New Step"),
                ])
                plan_store.save(new_plan)

        tui._handle_agent_response = fake_handle

        responses = iter(["加一个验证步骤", "n"])
        with patch.object(tui, '_ask_plan_approval', side_effect=responses):
            agent = MagicMock()
            await tui.run_goal(agent, "test goal")

        assert replan_called
        reloaded_original = plan_store.load(original.id)
        assert reloaded_original.status == "abandoned"


class TestGoalExecutionPhase:
    def _setup_approved_goal(self, tui, plan_store):
        plan = Plan(title="Exec Test", steps=[
            PlanStep(index=0, description="First"),
            PlanStep(index=1, description="Second"),
        ])
        plan_store.save(plan)
        return plan

    @pytest.mark.asyncio
    async def test_completes_via_plan(self, tui, plan_store):
        plan = self._setup_approved_goal(tui, plan_store)
        call_count = 0

        async def fake_handle(agent, prompt, allowed_tools=None):
            nonlocal call_count
            call_count += 1
            if allowed_tools is None:
                for s in plan.steps:
                    s.status = "completed"
                plan_store.save(plan)

        tui._handle_agent_response = fake_handle

        with patch.object(tui, '_ask_plan_approval', new_callable=AsyncMock, return_value=""):
            agent = MagicMock()
            agent.messages = [{"role": "assistant", "content": "done"}]
            await tui.run_goal(agent, "exec test")

        output = tui.console.file.getvalue()
        assert "目标完成" in output

    @pytest.mark.asyncio
    async def test_stops_on_needs_input(self, tui, plan_store):
        self._setup_approved_goal(tui, plan_store)

        async def fake_handle(agent, prompt, allowed_tools=None):
            pass

        tui._handle_agent_response = fake_handle

        with patch.object(tui, '_ask_plan_approval', new_callable=AsyncMock, return_value=""):
            agent = MagicMock()
            agent.messages = [{"role": "assistant", "content": "你想用什么格式？"}]
            with patch("loongcli.tui.app.detect_stop_intent", _mock_intent(StopIntent.NEEDS_INPUT)):
                await tui.run_goal(agent, "test goal")

        assert tui._goal_mode is False
        output = tui.console.file.getvalue()
        assert "等待用户输入" in output

    @pytest.mark.asyncio
    async def test_stops_on_stuck(self, tui, plan_store):
        self._setup_approved_goal(tui, plan_store)

        async def fake_handle(agent, prompt, allowed_tools=None):
            pass

        tui._handle_agent_response = fake_handle

        with patch.object(tui, '_ask_plan_approval', new_callable=AsyncMock, return_value=""):
            agent = MagicMock()
            agent.messages = [{"role": "assistant", "content": "无法连接"}]
            with patch("loongcli.tui.app.detect_stop_intent", _mock_intent(StopIntent.STUCK)):
                await tui.run_goal(agent, "test goal")

        assert tui._goal_mode is False
        output = tui.console.file.getvalue()
        assert "受阻" in output

    @pytest.mark.asyncio
    async def test_auto_continues(self, tui, plan_store):
        plan = self._setup_approved_goal(tui, plan_store)
        exec_count = 0

        async def fake_handle(agent, prompt, allowed_tools=None):
            nonlocal exec_count
            if allowed_tools is None:
                exec_count += 1
                if exec_count == 1:
                    plan.steps[0].status = "completed"
                    plan_store.save(plan)
                elif exec_count == 2:
                    plan.steps[1].status = "completed"
                    plan_store.save(plan)

        tui._handle_agent_response = fake_handle

        async def intent_seq(llm, text):
            if exec_count < 2:
                return StopIntent.CONTINUE
            return StopIntent.COMPLETED

        with patch.object(tui, '_ask_plan_approval', new_callable=AsyncMock, return_value=""):
            agent = MagicMock()
            agent.messages = [{"role": "assistant", "content": "继续"}]
            with patch("loongcli.tui.app.detect_stop_intent", intent_seq):
                await tui.run_goal(agent, "multi step")

        assert exec_count == 2
        output = tui.console.file.getvalue()
        assert "目标完成" in output

    @pytest.mark.asyncio
    async def test_max_iterations(self, tui, plan_store):
        tui.MAX_GOAL_ITERATIONS = 3
        self._setup_approved_goal(tui, plan_store)

        async def fake_handle(agent, prompt, allowed_tools=None):
            pass

        tui._handle_agent_response = fake_handle

        with patch.object(tui, '_ask_plan_approval', new_callable=AsyncMock, return_value=""):
            agent = MagicMock()
            agent.messages = [{"role": "assistant", "content": "处理中..."}]
            with patch("loongcli.tui.app.detect_stop_intent", _mock_intent(StopIntent.CONTINUE)):
                await tui.run_goal(agent, "infinite goal")

        assert tui._goal_mode is False
        output = tui.console.file.getvalue()
        assert "迭代上限" in output


class TestGoalCommand:
    @pytest.mark.asyncio
    async def test_goal_no_args(self):
        console = Console(file=StringIO(), force_terminal=True)
        ctx = CommandContext(console=console, agent=MagicMock())
        cmd = GoalCommand()
        await cmd.run([], ctx)
        output = console.file.getvalue()
        assert "用法" in output

    def test_registered(self):
        from loongcli.tui.commands import create_default_registry
        reg = create_default_registry()
        assert reg.get("goal") is not None
