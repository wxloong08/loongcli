import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from io import StringIO

from rich.console import Console
from rich.text import Text

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
        # 步骤全标完 ≠ 计划完成：完成只认 plan complete（p3-1：收尾步骤标完成时模型还在等用户验收）
        assert completed is False
        assert "plan complete" in hint

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

    def test_next_step_hint_lists_criteria(self, tui, plan_store):
        plan = Plan(title="Test", steps=[
            PlanStep(index=0, description="录音落盘", criteria=["WAV 头一致", "界面不写盘"],
                     evidence=["test_wav ok", ""]),
        ])
        plan_store.save(plan)
        hint, _ = tui._get_plan_hint()
        assert "判据 1（已有证据）: WAV 头一致" in hint     # 对外编号从 1 起
        assert "判据 2: 界面不写盘" in hint

    def test_approval_panel_shows_criteria(self, tui):
        plan = Plan(title="T", steps=[PlanStep(index=0, description="录音落盘", criteria=["界面不写盘"])])
        tui._display_plan(plan)
        assert "界面不写盘" in tui.console.file.getvalue()

    def test_approval_panel_shows_approach_above_steps(self, tui):
        plan = Plan(title="T", approach="## 关键决定\n- 插入用剪贴板 + Ctrl+V，放弃逐字 SendInput：中文输入法下逐字会被拦",
                    steps=[PlanStep(index=0, description="录音落盘", criteria=["界面不写盘"])])
        tui._display_plan(plan)
        out = tui.console.file.getvalue()
        assert "放弃逐字 SendInput" in out and "录音落盘" in out
        assert out.index("放弃逐字 SendInput") < out.index("录音落盘")   # 先审为什么，再审怎么做

    def test_skipped_counts_as_done(self, tui, plan_store):
        plan = Plan(title="Test", steps=[
            PlanStep(index=0, description="A", status="completed"),
            PlanStep(index=1, description="B", status="skipped"),
        ])
        plan_store.save(plan)
        hint, completed = tui._get_plan_hint()
        assert completed is False
        assert "2/2" in hint and "所有步骤已标完成" in hint


class TestGoalPrompts:
    def test_planning_prompt(self, tui):
        prompt = tui._goal_planning_prompt("重构认证模块")
        assert "重构认证模块" in prompt
        assert "plan(create)" in prompt
        assert "只创建计划" in prompt
        # 先读点名的文件、不把后续阶段提前（dogfood exec-1/2 把 P3/P5 排进了 P1）
        assert "read_file" in prompt
        assert "后续阶段" in prompt
        assert "criteria" in prompt
        # p3fix：目标要等用户验收结果时，验收单列最后一步并在那里暂停，而不是只「列出清单」就关计划
        assert "用户回复验收通过" in prompt

    def test_planning_prompt_requires_design_in_approach(self, tui):
        # 第 23.2 项：方案四块、调研随任务加深、判据查内容不查格式
        prompt = tui._goal_planning_prompt("写架构稿")
        for word in ("approach", "背景与范围", "放弃了什么", "风险", "需要用户拍板", "说出理由",
                     "查内容不查格式", "章节数"):
            assert word in prompt, word

    def test_replan_prompt_keeps_approach_in_sync(self, tui):
        assert "approach" in tui._goal_replan_prompt("写架构稿", "插入改用 SendInput")

    def test_execution_prompt_requires_evidence(self, tui):
        prompt = tui._goal_execution_prompt("重构")
        assert "evidence" in prompt
        assert "skipped" in prompt

    def test_planning_tools_read_only(self):
        """规划阶段：能读文件、能跑只读 shell，不能写、不能派子代理。"""
        tools = TUI.GOAL_PLANNING_TOOLS
        assert {"plan", "read_file", "shell"} <= tools
        assert not tools & {"write_file", "edit_file", "delegate", "batch_delegate", "exit_plan_mode"}

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
            assert allowed_tools == set(TUI.GOAL_PLANNING_TOOLS)
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
        assert "目标模式 · 规划中" in Text.from_ansi(output).plain
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


class TestGoalQuestionsFirst:
    """第 23.3 项：设计类目标先问再规划——回合结束没建计划且模型在提问，让用户答；答案进下一规划回合。"""

    @staticmethod
    def _agent(texts):
        agent = MagicMock()
        agent.messages = [{"role": "assistant", "content": t} for t in texts]
        return agent

    def _scripted(self, tui, plan_store, agent, script):
        """script：每个规划回合要做的事——字符串 = 模型说这句话（不建计划）；'PLAN' = 建计划。"""
        prompts = []

        async def fake_handle(a, prompt, allowed_tools=None):
            prompts.append(prompt)
            step = script[len(prompts) - 1]
            if step == "PLAN":
                plan_store.save(Plan(title="P", approach="决定", steps=[PlanStep(index=0, description="S")]))
            else:
                agent.messages.append({"role": "assistant", "content": step})

        tui._handle_agent_response = fake_handle
        return prompts

    @pytest.mark.asyncio
    async def test_answer_goes_into_next_planning_turn(self, tui, plan_store):
        agent = self._agent([])
        prompts = self._scripted(tui, plan_store, agent, ["插入用剪贴板还是逐字输入？我推荐剪贴板。", "PLAN"])
        with patch.object(tui, "_choose", new_callable=AsyncMock, return_value=(2, "用剪贴板，但要恢复原内容")), \
                patch.object(tui, "_ask_plan_approval", new_callable=AsyncMock, return_value="n") as approve:
            await tui.run_goal(agent, "写架构稿")
        assert len(prompts) == 2
        assert "用剪贴板，但要恢复原内容" in prompts[1] and "approach" in prompts[1]
        approve.assert_awaited()                     # 第二回合建了计划 → 进入审批
        assert "重试" not in tui.console.file.getvalue()

    @pytest.mark.asyncio
    async def test_use_recommendation(self, tui, plan_store):
        agent = self._agent([])
        prompts = self._scripted(tui, plan_store, agent, ["用哪家云端识别？推荐百炼。", "PLAN"])
        with patch.object(tui, "_choose", new_callable=AsyncMock, return_value=(0, "")), \
                patch.object(tui, "_ask_plan_approval", new_callable=AsyncMock, return_value="n"):
            await tui.run_goal(agent, "写架构稿")
        assert "按自己的推荐" in prompts[1]

    @pytest.mark.asyncio
    async def test_cancel_at_questions(self, tui, plan_store):
        agent = self._agent([])
        prompts = self._scripted(tui, plan_store, agent, ["哪种插入方式？"])
        with patch.object(tui, "_choose", new_callable=AsyncMock, return_value=(1, "")):
            await tui.run_goal(agent, "写架构稿")
        assert len(prompts) == 1 and tui._goal_mode is False
        assert "规划已取消" in tui.console.file.getvalue()

    @pytest.mark.asyncio
    async def test_question_rounds_capped_then_forced(self, tui, plan_store):
        # 连问 3 轮仍不出计划：第 4 回合起不再弹问答框，提示里要求不许再问、直接出计划
        agent = self._agent([])
        script = ["问题一？", "问题二？", "问题三？", "问题四？", "PLAN"]
        prompts = self._scripted(tui, plan_store, agent, script)
        choose = AsyncMock(return_value=(2, "随你"))
        with patch.object(tui, "_choose", choose), \
                patch.object(tui, "_ask_plan_approval", new_callable=AsyncMock, return_value="n"):
            await tui.run_goal(agent, "写架构稿")
        assert choose.await_count == TUI.GOAL_QUESTION_ROUNDS
        assert "不要再提问" in prompts[3]
        assert "不要再提问" in prompts[4]

    @pytest.mark.asyncio
    async def test_non_question_text_still_retries(self, tui, plan_store):
        # 偶发没建成（说了话但没提问）走原来的重试，不弹问答框
        agent = self._agent([])
        prompts = self._scripted(tui, plan_store, agent, ["我先读需求文档。", "PLAN"])
        choose = AsyncMock()
        with patch.object(tui, "_choose", choose), \
                patch.object(tui, "_ask_plan_approval", new_callable=AsyncMock, return_value="n"):
            await tui.run_goal(agent, "写架构稿")
        choose.assert_not_awaited()
        assert "重试" in tui.console.file.getvalue()

    def test_planning_prompt_asks_before_planning_for_design_goals(self, tui):
        prompt = tui._goal_planning_prompt("写架构稿")
        assert "先问再规划" in prompt and "3-6 个问题" in prompt and "你的推荐" in prompt


class TestGoalApprovalPhase:
    def _setup_plan(self, plan_store):
        # created_at 设在未来 = 规划回合里新建的计划（run_goal 只认规划开始之后创建的）
        plan = Plan(title="Approve Me", created_at="9999-01-01T00:00:00+00:00", steps=[
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
                plan.status = "completed"   # 模型 plan complete 关闭计划
                plan_store.save(plan)

        tui._handle_agent_response = fake_handle

        approval_calls = [AsyncMock(return_value="")]
        with patch.object(tui, '_ask_plan_approval', side_effect=[""]):
            agent = MagicMock()
            agent.messages = [{"role": "assistant", "content": "done"}]
            with patch("loongcli.tui.app.detect_stop_intent", _mock_intent(StopIntent.COMPLETED)):
                await tui.run_goal(agent, "test goal")

        assert any(c["allowed_tools"] == set(TUI.GOAL_PLANNING_TOOLS) for c in handle_calls)
        assert any(c["allowed_tools"] is None for c in handle_calls)
        output = tui.console.file.getvalue()
        assert "目标模式 · 执行中" in Text.from_ansi(output).plain

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
                assert allowed_tools == set(TUI.GOAL_PLANNING_TOOLS)
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
        # created_at 设在未来 = 规划回合里新建的计划（run_goal 只认规划开始之后创建的）
        plan = Plan(title="Exec Test", created_at="9999-01-01T00:00:00+00:00", steps=[
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
                plan.status = "completed"   # 模型 plan complete 关闭计划
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
                    plan.status = "completed"
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


class TestGoalTracksOwnPlan:
    """goal 只认自己这份计划，不认「最近修改的计划」（exec-4：收尾时把遗留旧计划标放弃，目标完成被误判成等待输入）。"""

    def _stale_plan(self, plan_store):
        stale = Plan(title="上一轮遗留", steps=[PlanStep(index=0, description="遗留步骤")])
        stale.created_at = "2026-01-01T00:00:00+00:00"  # 早于本次 goal
        plan_store.save(stale)
        return stale

    @staticmethod
    def _save_as_latest_modified(plan_store, plan):
        """保存并把文件修改时间推后：测试里两次保存常落在同一时间刻度，不推后旧代码会碰巧选对。"""
        import os, time
        plan_store.save(plan)
        t = time.time() + 60
        os.utime(plan_store._path(plan.id), (t, t))

    @pytest.mark.asyncio
    async def test_completion_not_masked_by_later_modified_stale_plan(self, tui, plan_store):
        stale = self._stale_plan(plan_store)
        holder = {}

        async def fake_handle(agent, prompt, allowed_tools=None):
            if allowed_tools is not None:  # 规划：建本次计划
                holder["plan"] = Plan(title="本次", steps=[PlanStep(index=0, description="做 A")])
                plan_store.save(holder["plan"])
                return
            plan = holder["plan"]
            plan.steps[0].status = "completed"
            plan.status = "completed"
            plan_store.save(plan)
            stale.status = "abandoned"   # 最后一个动作：遗留计划标放弃 → 它成了最近修改的那份
            self._save_as_latest_modified(plan_store, stale)

        tui._handle_agent_response = fake_handle
        with patch.object(tui, '_ask_plan_approval', new_callable=AsyncMock, return_value=""):
            agent = MagicMock()
            agent.messages = [{"role": "assistant", "content": "需要你本人操作：录一段音"}]
            # 走到意图判断就会判成等待输入；正确行为是根本不走到这一步
            with patch("loongcli.tui.app.detect_stop_intent", _mock_intent(StopIntent.NEEDS_INPUT)):
                await tui.run_goal(agent, "goal")

        output = tui.console.file.getvalue()
        assert "计划已完成" in output
        assert "等待用户输入" not in output

    @pytest.mark.asyncio
    async def test_next_step_hint_comes_from_own_plan(self, tui, plan_store):
        stale = self._stale_plan(plan_store)
        holder, prompts = {}, []

        async def fake_handle(agent, prompt, allowed_tools=None):
            if allowed_tools is not None:
                holder["plan"] = Plan(title="本次", steps=[
                    PlanStep(index=0, description="做 A"), PlanStep(index=1, description="做 B"),
                ])
                plan_store.save(holder["plan"])
                return
            prompts.append(prompt)
            if len(prompts) == 1:
                holder["plan"].steps[0].status = "completed"
                plan_store.save(holder["plan"])
                self._save_as_latest_modified(plan_store, stale)   # 动一下遗留计划，刷新它的修改时间

        tui._handle_agent_response = fake_handle
        intents = iter([StopIntent.CONTINUE, StopIntent.COMPLETED])

        async def intent_seq(llm, text):
            return next(intents)

        with patch.object(tui, '_ask_plan_approval', new_callable=AsyncMock, return_value=""):
            agent = MagicMock()
            agent.messages = [{"role": "assistant", "content": "继续"}]
            with patch("loongcli.tui.app.detect_stop_intent", intent_seq):
                await tui.run_goal(agent, "goal")

        assert len(prompts) == 2
        assert "做 B" in prompts[1]
        assert "遗留步骤" not in prompts[1]

    def test_latest_plan_is_by_creation_time(self, tui, plan_store):
        stale = self._stale_plan(plan_store)
        new = Plan(title="新建", steps=[PlanStep(index=0, description="x")])
        plan_store.save(new)
        self._save_as_latest_modified(plan_store, stale)  # 旧计划最后被修改
        assert tui._get_latest_plan().id == new.id


class TestGoalPauseResume:
    """暂停与续跑（Claude Code / Codex 的 /goal 语义）：停下来不等于结束，用户回复后接着跑。"""

    def _rig(self, tui, plan_store, steps=("做 A",)):
        """规划回合建计划；执行回合把调用记下来，由 on_exec 决定做什么。"""
        rig = {"plan": None, "exec": [], "on_exec": None}

        async def fake_handle(agent, prompt, allowed_tools=None):
            if allowed_tools is not None and allowed_tools:   # 规划（非空工具集）
                rig["plan"] = Plan(title="本次", steps=[PlanStep(index=i, description=d) for i, d in enumerate(steps)])
                plan_store.save(rig["plan"])
                return
            rig["exec"].append({"prompt": prompt, "allowed_tools": allowed_tools})
            if rig["on_exec"]:
                rig["on_exec"](rig["plan"])

        tui._handle_agent_response = fake_handle
        return rig

    @staticmethod
    def _agent(tool_calls=1):
        agent = MagicMock()
        agent.messages = [{"role": "assistant", "content": "需要你按一下热键"}]
        agent.turn_tool_calls = tool_calls
        return agent

    @staticmethod
    def _complete(plan_store):
        def on_exec(plan):
            for s in plan.steps:
                s.status = "completed"
            plan.status = "completed"
            plan_store.save(plan)
        return on_exec

    async def _start_paused(self, tui, plan_store, rig, agent):
        with patch.object(tui, '_ask_plan_approval', new_callable=AsyncMock, return_value=""), \
             patch("loongcli.tui.app.detect_stop_intent", _mock_intent(StopIntent.NEEDS_INPUT)):
            await tui.run_goal(agent, "实现热键")

    @pytest.mark.asyncio
    async def test_needs_input_pauses_and_keeps_goal(self, tui, plan_store):
        rig = self._rig(tui, plan_store)
        agent = self._agent()
        await self._start_paused(tui, plan_store, rig, agent)
        assert tui._goal is not None and tui._goal.status == "paused"
        assert tui._goal.pause_reason == "等待用户输入"
        assert tui._goal_mode is False
        agent.set_goal.assert_called_once_with("实现热键", rig["plan"].id)
        agent.clear_goal.assert_not_called()     # 暂停期间目标仍常驻系统提示
        assert "目标已暂停：等待用户输入" in tui.console.file.getvalue()

    @pytest.mark.asyncio
    async def test_reply_resumes_to_completion(self, tui, plan_store):
        rig = self._rig(tui, plan_store)
        agent = self._agent()
        await self._start_paused(tui, plan_store, rig, agent)

        rig["on_exec"] = self._complete(plan_store)
        await tui.resume_goal(agent, "按好了，录音正常")
        resume_prompt = rig["exec"][-1]["prompt"]
        assert resume_prompt.startswith("按好了，录音正常")      # 用户原话在前
        assert "目标模式续跑" in resume_prompt and "实现热键" in resume_prompt
        assert tui._goal is None
        agent.clear_goal.assert_called_once()
        assert "目标完成" in tui.console.file.getvalue()

    @pytest.mark.asyncio
    async def test_stop_abandons_plan(self, tui, plan_store):
        rig = self._rig(tui, plan_store)
        agent = self._agent()
        await self._start_paused(tui, plan_store, rig, agent)
        tui.stop_goal(agent)
        assert tui._goal is None
        assert plan_store.load(rig["plan"].id).status == "abandoned"
        agent.clear_goal.assert_called_once()

    @pytest.mark.asyncio
    async def test_resume_when_plan_already_done_ends_goal(self, tui, plan_store):
        rig = self._rig(tui, plan_store)
        agent = self._agent()
        await self._start_paused(tui, plan_store, rig, agent)
        plan = rig["plan"]
        plan.status = "completed"
        plan_store.save(plan)
        await tui.resume_goal(agent, "顺便问个别的")
        assert tui._goal is None
        assert rig["exec"][-1]["prompt"] == "顺便问个别的"    # 按普通对话处理，不带续跑包装

    @pytest.mark.asyncio
    async def test_no_tool_calls_pauses_without_intent_call(self, tui, plan_store):
        rig = self._rig(tui, plan_store)
        agent = self._agent(tool_calls=0)

        async def must_not_call(llm, text):
            raise AssertionError("无进展时不应调用意图判断")

        with patch.object(tui, '_ask_plan_approval', new_callable=AsyncMock, return_value=""), \
             patch("loongcli.tui.app.detect_stop_intent", must_not_call):
            await tui.run_goal(agent, "g")
        assert tui._goal.status == "paused"
        assert tui._goal.pause_reason == "本轮没有调用任何工具"

    @pytest.mark.asyncio
    async def test_budget_exhausted_wraps_up_then_pauses(self, tui, plan_store):
        tui.MAX_GOAL_ITERATIONS = 2
        rig = self._rig(tui, plan_store)
        agent = self._agent()
        with patch.object(tui, '_ask_plan_approval', new_callable=AsyncMock, return_value=""), \
             patch("loongcli.tui.app.detect_stop_intent", _mock_intent(StopIntent.CONTINUE)):
            await tui.run_goal(agent, "g")
            assert [c["allowed_tools"] for c in rig["exec"]] == [None, None, set()]   # 收尾回合不给工具
            assert "迭代预算已用完" in rig["exec"][-1]["prompt"]
            assert tui._goal.status == "paused" and tui._goal.pause_reason == "迭代上限"
            assert plan_store.load(rig["plan"].id).status == "active"                  # 触顶 ≠ 完成
            # 续跑重给一轮预算
            await tui.resume_goal(agent, "继续")
        assert len(rig["exec"]) == 6
        assert tui._goal.iterations == 4

    @pytest.mark.asyncio
    async def test_claimed_complete_with_incomplete_plan_pauses(self, tui, plan_store):
        rig = self._rig(tui, plan_store)
        agent = self._agent()
        with patch.object(tui, '_ask_plan_approval', new_callable=AsyncMock, return_value=""), \
             patch("loongcli.tui.app.detect_stop_intent", _mock_intent(StopIntent.COMPLETED)):
            await tui.run_goal(agent, "g")
        assert tui._goal.status == "paused"
        assert "计划还有步骤未完成" in tui._goal.pause_reason
        assert "目标完成" not in tui.console.file.getvalue()

    @pytest.mark.asyncio
    async def test_all_steps_done_but_waiting_for_user_pauses(self, tui, plan_store):
        """p3-1 回归：模型把收尾步骤标完成、计划 8/8，但回复里还在请用户回验收结果、没 plan complete。
        旧逻辑判「目标完成」，用户的回复成了脱离目标的普通对话；应暂停，回复后续跑。"""
        rig = self._rig(tui, plan_store)
        agent = self._agent()

        def steps_done_not_closed(plan):
            for s in plan.steps:
                s.status = "completed"
            plan_store.save(plan)          # 不 plan complete

        rig["on_exec"] = steps_done_not_closed
        await self._start_paused(tui, plan_store, rig, agent)   # 意图判为等待输入
        assert tui._goal is not None and tui._goal.status == "paused"
        assert "目标完成" not in tui.console.file.getvalue()

        rig["on_exec"] = self._complete(plan_store)            # 用户回复后，模型审计并关闭计划
        await tui.resume_goal(agent, "第 10 条正确，A、B 都成立")
        assert rig["exec"][-1]["prompt"].startswith("第 10 条正确")
        assert tui._goal is None
        assert "目标完成" in tui.console.file.getvalue()

    @pytest.mark.asyncio
    async def test_claimed_complete_with_steps_done_continues_to_close(self, tui, plan_store):
        """步骤都标完、模型自称完成但没关计划：不暂停，再续一轮让它审计后 plan complete。"""
        rig = self._rig(tui, plan_store)
        agent = self._agent()
        calls = {"n": 0}

        def on_exec(plan):
            calls["n"] += 1
            for s in plan.steps:
                s.status = "completed"
            if calls["n"] == 2:
                plan.status = "completed"
            plan_store.save(plan)

        rig["on_exec"] = on_exec
        with patch.object(tui, '_ask_plan_approval', new_callable=AsyncMock, return_value=""), \
             patch("loongcli.tui.app.detect_stop_intent", _mock_intent(StopIntent.COMPLETED)):
            await tui.run_goal(agent, "g")
        assert calls["n"] == 2
        assert "plan complete" in rig["exec"][1]["prompt"]
        assert tui._goal is None

    @pytest.mark.asyncio
    async def test_new_goal_replaces_paused_one(self, tui, plan_store):
        rig = self._rig(tui, plan_store)
        agent = self._agent()
        await self._start_paused(tui, plan_store, rig, agent)
        old_id = rig["plan"].id
        await self._start_paused(tui, plan_store, rig, agent)
        assert plan_store.load(old_id).status == "abandoned"
        assert tui._goal.plan_id == rig["plan"].id != old_id

    @pytest.mark.asyncio
    async def test_goal_command_status_and_stop_do_not_resume(self, tui, plan_store):
        """暂停期间 /goal（看状态）与 /goal stop 是斜杠命令，不触发续跑。"""
        rig = self._rig(tui, plan_store)
        agent = self._agent()
        await self._start_paused(tui, plan_store, rig, agent)
        n_exec = len(rig["exec"])
        ctx = CommandContext(console=tui.console, agent=agent, tui=tui)
        await GoalCommand().run([], ctx)
        assert "暂停（等待用户输入）" in tui.console.file.getvalue()
        await GoalCommand().run(["stop"], ctx)
        assert tui._goal is None
        assert len(rig["exec"]) == n_exec

    @pytest.mark.asyncio
    async def test_esc_interrupt_pauses_instead_of_continuing(self, tui, plan_store):
        """执行回合被 Esc 中断：目标暂停，不再判断意图、不自动续跑（此前循环照常往下走）。"""
        rig = self._rig(tui, plan_store)
        inner = tui._handle_agent_response

        async def handle(agent, prompt, allowed_tools=None):
            await inner(agent, prompt, allowed_tools)
            return allowed_tools is None            # 执行回合 = 被用户中断
        tui._handle_agent_response = handle
        intent = AsyncMock(return_value=StopIntent.CONTINUE)
        with patch.object(tui, '_ask_plan_approval', new_callable=AsyncMock, return_value=""), \
             patch("loongcli.tui.app.detect_stop_intent", intent):
            await tui.run_goal(self._agent(), "实现热键")
        assert len(rig["exec"]) == 1
        intent.assert_not_called()
        assert tui._goal.status == "paused" and tui._goal.pause_reason == "用户中断"

    @pytest.mark.asyncio
    async def test_goal_stop_requested_while_running_ends_goal(self, tui, plan_store):
        """运行中发 /goal stop（_queue_interjection 置标志）：当前回合返回后结束目标、计划标放弃。"""
        rig = self._rig(tui, plan_store, steps=("做 A", "做 B"))
        rig["on_exec"] = lambda plan: setattr(tui, "_goal_stop_requested", True)
        with patch.object(tui, '_ask_plan_approval', new_callable=AsyncMock, return_value=""), \
             patch("loongcli.tui.app.detect_stop_intent", _mock_intent(StopIntent.CONTINUE)):
            await tui.run_goal(self._agent(), "实现热键")
        assert len(rig["exec"]) == 1
        assert tui._goal is None and tui._goal_stop_requested is False
        assert plan_store.load(rig["plan"].id).status == "abandoned"


class TestGoalPlanningInterrupted:
    """规划阶段被叫停：不拿上一个目标已放弃的计划去审批（14e 真机：Esc 后执行了已放弃的计划）。"""

    def _stale_plan(self, plan_store):
        stale = Plan(title="旧计划", status="abandoned", steps=[PlanStep(index=0, description="旧")])
        plan_store.save(stale)
        return stale

    @pytest.mark.asyncio
    async def test_esc_in_planning_cancels_goal(self, tui, plan_store):
        self._stale_plan(plan_store)

        async def handle(agent, prompt, allowed_tools=None):
            return True                              # 规划回合被 Esc 中断
        tui._handle_agent_response = handle
        approval = AsyncMock(return_value="")
        with patch.object(tui, "_ask_plan_approval", approval):
            await tui.run_goal(MagicMock(), "新目标")
        approval.assert_not_called()
        assert tui._goal is None and tui._goal_planning is False
        assert "规划已中断" in tui.console.file.getvalue()

    @pytest.mark.asyncio
    async def test_stop_in_planning_abandons_new_plan(self, tui, plan_store):
        made = {}

        async def handle(agent, prompt, allowed_tools=None):
            made["plan"] = Plan(title="新计划", steps=[PlanStep(index=0, description="做")])
            plan_store.save(made["plan"])
            tui._goal_stop_requested = True          # 规划中发了 /goal stop
            return True
        tui._handle_agent_response = handle
        with patch.object(tui, "_ask_plan_approval", AsyncMock(return_value="")) as approval:
            await tui.run_goal(MagicMock(), "新目标")
        approval.assert_not_called()
        assert plan_store.load(made["plan"].id).status == "abandoned"
        assert tui._goal_stop_requested is False

    @pytest.mark.asyncio
    async def test_stale_abandoned_plan_never_offered(self, tui, plan_store):
        """规划没建出新计划：旧的已放弃计划不算数，按「未能创建」处理。"""
        self._stale_plan(plan_store)

        async def handle(agent, prompt, allowed_tools=None):
            return False
        tui._handle_agent_response = handle
        with patch.object(tui, "_ask_plan_approval", AsyncMock(return_value="")) as approval:
            await tui.run_goal(MagicMock(), "新目标")
        approval.assert_not_called()
        assert "未能创建执行计划" in tui.console.file.getvalue()

    @pytest.mark.asyncio
    async def test_esc_at_approval_cancels(self, tui, plan_store):
        async def handle(agent, prompt, allowed_tools=None):
            plan_store.save(Plan(title="新计划", steps=[PlanStep(index=0, description="做")]))
            return False
        tui._handle_agent_response = handle
        with patch.object(tui, "_ask_plan_approval", AsyncMock(return_value=None)):
            await tui.run_goal(MagicMock(), "新目标")
        out = tui.console.file.getvalue()
        assert "计划已取消" in out and "执行阶段" not in out


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
