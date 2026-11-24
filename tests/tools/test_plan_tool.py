import pytest
from loongcli.tools.plan_tool import PlanTool
from loongcli.tools.base import ToolRegistry
from loongcli.plan.store import PlanStore, Plan, PlanStep
import loongcli.plan.store as store_mod


def _s(*descs: str) -> list[dict]:
    """create 用的步骤：每步带一条判据。"""
    return [{"description": d, "criteria": [f"{d} 做完"]} for d in descs]


def _ev(text: str = "pytest 3 passed") -> list[dict]:
    return [{"criterion": 1, "text": text}]


@pytest.fixture
def plan_tool(tmp_path):
    store = PlanStore(base_dir=tmp_path / "plans")
    return PlanTool(store), store


class TestPlanTool:
    @pytest.mark.asyncio
    async def test_create(self, plan_tool):
        tool, store = plan_tool
        result = await tool.execute(
            operation="create", approach="背景：测试用方案",
            title="Auth refactor",
            steps=_s("Analyze", "Test", "Refactor"),
        )
        assert "已创建" in result
        plans = store.list_plans()
        assert len(plans) == 1
        assert plans[0].title == "Auth refactor"
        assert len(plans[0].steps) == 3

    @pytest.mark.asyncio
    async def test_create_requires_title(self, plan_tool):
        tool, _ = plan_tool
        result = await tool.execute(operation="create", approach="背景：测试用方案", steps=_s("A"))
        assert "错误" in result

    @pytest.mark.asyncio
    async def test_create_requires_steps(self, plan_tool):
        tool, _ = plan_tool
        result = await tool.execute(operation="create", approach="背景：测试用方案", title="Test")
        assert "错误" in result

    @pytest.mark.asyncio
    async def test_update_step(self, plan_tool):
        tool, store = plan_tool
        result = await tool.execute(
            operation="create", approach="背景：测试用方案", title="Test", steps=_s("A", "B"),
        )
        plan_id = store.list_plans()[0].id

        result = await tool.execute(
            operation="update_step", plan_id=plan_id,
            step=1, step_status="completed", step_output="done", evidence=_ev(),
        )
        assert "已更新" in result
        assert "1/2" in result

        plan = store.load(plan_id)
        assert plan.steps[0].status == "completed"
        assert plan.steps[0].output == "done"
        assert plan.steps[0].evidence == ["pytest 3 passed"]

    @pytest.mark.asyncio
    async def test_update_step_invalid_index(self, plan_tool):
        tool, store = plan_tool
        await tool.execute(operation="create", approach="背景：测试用方案", title="T", steps=_s("A"))
        plan_id = store.list_plans()[0].id
        result = await tool.execute(
            operation="update_step", plan_id=plan_id, step=6,
        )
        assert "超出范围" in result

    @pytest.mark.asyncio
    async def test_update_step_missing_plan(self, plan_tool):
        tool, _ = plan_tool
        result = await tool.execute(
            operation="update_step", plan_id="nope", step=1,
        )
        assert "未找到" in result

    @pytest.mark.asyncio
    async def test_get_specific(self, plan_tool):
        tool, store = plan_tool
        await tool.execute(operation="create", approach="背景：测试用方案", title="My Plan", steps=_s("A", "B"))
        plan_id = store.list_plans()[0].id
        result = await tool.execute(operation="get", plan_id=plan_id)
        assert "My Plan" in result

    @pytest.mark.asyncio
    async def test_get_active(self, plan_tool):
        tool, _ = plan_tool
        await tool.execute(operation="create", approach="背景：测试用方案", title="Plan A", steps=_s("X"))
        result = await tool.execute(operation="get")
        assert "Plan A" in result

    @pytest.mark.asyncio
    async def test_get_no_active(self, plan_tool):
        tool, _ = plan_tool
        result = await tool.execute(operation="get")
        assert "没有活跃计划" in result

    @pytest.mark.asyncio
    async def test_complete(self, plan_tool):
        tool, store = plan_tool
        await tool.execute(operation="create", approach="背景：测试用方案", title="T", steps=_s("A"))
        plan_id = store.list_plans()[0].id
        await tool.execute(operation="update_step", plan_id=plan_id, step=1,
                           step_status="completed", evidence=_ev())
        result = await tool.execute(operation="complete", plan_id=plan_id)
        assert "completed" in result
        assert store.load(plan_id).status == "completed"

    @pytest.mark.asyncio
    async def test_abandon(self, plan_tool):
        tool, store = plan_tool
        await tool.execute(operation="create", approach="背景：测试用方案", title="T", steps=_s("A"))
        plan_id = store.list_plans()[0].id
        result = await tool.execute(operation="abandon", plan_id=plan_id)
        assert "abandoned" in result

    @pytest.mark.asyncio
    async def test_list(self, plan_tool):
        tool, store = plan_tool
        await tool.execute(operation="create", approach="背景：测试用方案", title="Plan A", steps=_s("X"))
        # A 动工后再建 B——未动工的纯草稿会被覆盖（这是特性，另有专测）
        id_a = store.list_plans()[0].id
        await tool.execute(operation="update_step", plan_id=id_a, step=1, step_status="in_progress")
        await tool.execute(operation="create", approach="背景：测试用方案", title="Plan B", steps=_s("Y"))
        result = await tool.execute(operation="list")
        assert "Plan A" in result
        assert "Plan B" in result

    @pytest.mark.asyncio
    async def test_list_empty(self, plan_tool):
        tool, _ = plan_tool
        result = await tool.execute(operation="list")
        assert "没有计划" in result

    def test_tool_schema(self, plan_tool):
        tool, _ = plan_tool
        reg = ToolRegistry()
        reg.register(tool)
        schemas = reg.get_tool_schemas()
        assert len(schemas) == 1
        fn = schemas[0]["function"]
        assert fn["name"] == "plan"
        props = fn["parameters"]["properties"]
        assert "operation" in props
        assert "plan_id" in props
        assert "steps" in props


class TestProjectScoping:
    """Plans are scoped per project dir, like sessions — no cross-project leak."""

    def test_plans_isolated_by_project(self, tmp_path, monkeypatch):
        # redirect the projects root into tmp so we never touch ~/.loongcli
        monkeypatch.setattr(store_mod, "_projects_root", lambda: tmp_path / "projects")

        proj_a = tmp_path / "ws" / "alpha"
        proj_b = tmp_path / "ws" / "beta"
        proj_a.mkdir(parents=True)
        proj_b.mkdir(parents=True)

        store_a = PlanStore(project_dir=proj_a)
        store_b = PlanStore(project_dir=proj_b)

        assert store_a.base_dir != store_b.base_dir  # distinct on-disk dirs

        store_a.save(Plan(title="alpha-only", steps=[PlanStep(index=0, description="x")]))

        assert [p.title for p in store_a.list_plans()] == ["alpha-only"]
        assert store_b.list_plans() == []  # beta cannot see alpha's plans

    def test_default_dir_is_under_projects_root(self, tmp_path, monkeypatch):
        monkeypatch.setattr(store_mod, "_projects_root", lambda: tmp_path / "projects")
        proj = tmp_path / "ws" / "gamma"
        proj.mkdir(parents=True)
        store = PlanStore(project_dir=proj)
        # plans sit alongside sessions: <root>/<slug>/plans
        assert store.base_dir.parent == (tmp_path / "projects" / store.base_dir.parent.name)
        assert store.base_dir.name == "plans"


# ── 草稿覆盖：孤儿计划回归（真机：一轮连开三个 plan） ──

@pytest.mark.asyncio
async def test_create_replaces_untouched_draft(tmp_path):
    from loongcli.plan.store import PlanStore
    from loongcli.tools.plan_tool import PlanTool

    store = PlanStore(base_dir=tmp_path)
    tool = PlanTool(store)
    r1 = await tool.execute(operation="create", approach="背景：测试用方案", title="A", steps=_s("s1"))
    id_a = r1.split("计划已创建: ")[1].split("（")[0].split("\n")[0].strip()
    r2 = await tool.execute(operation="create", approach="背景：测试用方案", title="B", steps=_s("s1"))
    assert "已覆盖未动工草稿" in r2
    assert store.load(id_a) is None          # 孤儿草稿被清
    assert len(store.list_plans()) == 1


@pytest.mark.asyncio
async def test_create_keeps_started_plan(tmp_path):
    """动过工（任一步骤非 pending）的计划绝不覆盖。"""
    from loongcli.plan.store import PlanStore
    from loongcli.tools.plan_tool import PlanTool

    store = PlanStore(base_dir=tmp_path)
    tool = PlanTool(store)
    r1 = await tool.execute(operation="create", approach="背景：测试用方案", title="A", steps=_s("s1", "s2"))
    id_a = r1.split("计划已创建: ")[1].split("（")[0].split("\n")[0].strip()
    await tool.execute(operation="update_step", plan_id=id_a, step=1, step_status="in_progress")
    r2 = await tool.execute(operation="create", approach="背景：测试用方案", title="B", steps=_s("s1"))
    assert "已覆盖" not in r2
    assert store.load(id_a) is not None
    assert len(store.list_plans()) == 2


@pytest.mark.asyncio
async def test_create_keeps_active_plan(tmp_path):
    """已批准（agent._active_plan_id）的计划绝不覆盖，即使步骤全 pending。"""
    from unittest.mock import MagicMock
    from loongcli.plan.store import PlanStore
    from loongcli.tools.plan_tool import PlanTool

    store = PlanStore(base_dir=tmp_path)
    tool = PlanTool(store)
    r1 = await tool.execute(operation="create", approach="背景：测试用方案", title="A", steps=_s("s1"))
    id_a = r1.split("计划已创建: ")[1].split("（")[0].split("\n")[0].strip()

    agent = MagicMock()
    agent._active_plan_id = id_a
    tool.bind_agent(agent)

    r2 = await tool.execute(operation="create", approach="背景：测试用方案", title="B", steps=_s("s1"))
    assert "已覆盖" not in r2
    assert store.load(id_a) is not None


# ── 判据与证据闸（2026-11-15：此前 plan complete 无条件接受，完成 = 模型自称完成） ──

class TestCriteriaGate:
    async def _plan(self, tool, store, steps):
        await tool.execute(operation="create", approach="背景：测试用方案", title="T", steps=steps)
        return store.list_plans()[0].id

    @pytest.mark.asyncio
    async def test_create_rejects_plain_text_steps(self, plan_tool):
        tool, store = plan_tool
        result = await tool.execute(operation="create", approach="背景：测试用方案", title="T", steps=["只有描述"])
        assert "criteria" in result and "错误" in result
        assert store.list_plans() == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize("criteria", [[], ["  "], ["a", "b", "c", "d", "e", "f"]])
    async def test_create_requires_1_to_5_criteria(self, plan_tool, criteria):
        tool, store = plan_tool
        result = await tool.execute(operation="create", approach="背景：测试用方案", title="T",
                                    steps=[{"description": "d", "criteria": criteria}])
        assert "1-5 条 criteria" in result
        assert store.list_plans() == []

    @pytest.mark.asyncio
    async def test_invalid_create_keeps_previous_draft(self, plan_tool):
        """新计划不合格时不能先把旧草稿删了。"""
        tool, store = plan_tool
        await tool.execute(operation="create", approach="背景：测试用方案", title="A", steps=_s("s1"))
        await tool.execute(operation="create", approach="背景：测试用方案", title="B", steps=["无判据"])
        assert [p.title for p in store.list_plans()] == ["A"]

    @pytest.mark.asyncio
    async def test_complete_step_without_evidence_rejected(self, plan_tool):
        tool, store = plan_tool
        pid = await self._plan(tool, store, [
            {"description": "录音落盘", "criteria": ["WAV 头长度一致", "界面线程不写盘"]},
        ])
        result = await tool.execute(operation="update_step", plan_id=pid, step=1,
                                    step_status="completed", evidence=_ev("test_wav 通过"))
        assert "未标完成" in result
        assert "界面线程不写盘" in result          # 列出缺的那条
        assert "WAV 头长度一致" not in result       # 已交证据的不再列
        step = store.load(pid).steps[0]
        assert step.status == "pending"             # 状态没变
        assert step.evidence == ["test_wav 通过", ""]  # 已交证据保存了

    @pytest.mark.asyncio
    async def test_evidence_can_be_given_incrementally(self, plan_tool):
        tool, store = plan_tool
        pid = await self._plan(tool, store, [{"description": "d", "criteria": ["c0", "c1"]}])
        await tool.execute(operation="update_step", plan_id=pid, step=1,
                           evidence=[{"criterion": 1, "text": "e0"}])
        result = await tool.execute(operation="update_step", plan_id=pid, step=1,
                                    step_status="completed", evidence=[{"criterion": "2", "text": "e1"}])
        assert "已更新" in result
        assert store.load(pid).steps[0].evidence == ["e0", "e1"]

    @pytest.mark.asyncio
    async def test_evidence_index_out_of_range(self, plan_tool):
        tool, store = plan_tool
        pid = await self._plan(tool, store, _s("A"))
        result = await tool.execute(operation="update_step", plan_id=pid, step=1,
                                    evidence=[{"criterion": 4, "text": "x"}])
        assert "超出范围" in result

    @pytest.mark.asyncio
    async def test_numbers_are_one_based(self, plan_tool):
        """对外编号一律从 1 起：步骤 0、判据 0 都越界，报错要说清从 1 起（真机：模型时而说第 0 步、时而第 1 步，界面对不上）。"""
        tool, store = plan_tool
        pid = await self._plan(tool, store, _s("A", "B"))
        r = await tool.execute(operation="update_step", plan_id=pid, step=0, step_status="in_progress")
        assert "超出范围" in r and "1-2" in r
        r = await tool.execute(operation="update_step", plan_id=pid, step=1, evidence=[{"criterion": 0, "text": "x"}])
        assert "超出范围" in r and "编号 1-1" in r
        r = await tool.execute(operation="update_step", plan_id=pid, step=2, step_status="completed", evidence=_ev())
        assert r.startswith("步骤 2 已更新") and store.load(pid).steps[1].status == "completed"
        assert "**1.** A" in r and "判据 1 [○]" in r     # 回带的计划全貌也从 1 编号

    @pytest.mark.asyncio
    async def test_skip_requires_reason(self, plan_tool):
        tool, store = plan_tool
        pid = await self._plan(tool, store, _s("A"))
        result = await tool.execute(operation="update_step", plan_id=pid, step=1, step_status="skipped")
        assert "跳过理由" in result
        assert store.load(pid).steps[0].status == "pending"
        result = await tool.execute(operation="update_step", plan_id=pid, step=1,
                                    step_status="skipped", step_output="需要用户实机操作")
        assert "已更新" in result

    @pytest.mark.asyncio
    async def test_plan_complete_rejected_with_unfinished_steps(self, plan_tool):
        tool, store = plan_tool
        pid = await self._plan(tool, store, _s("A", "B"))
        await tool.execute(operation="update_step", plan_id=pid, step=1,
                           step_status="completed", evidence=_ev())
        result = await tool.execute(operation="complete", plan_id=pid)
        assert "不能标为完成" in result
        assert "步骤 2 未完成" in result            # 对外编号从 1 起：没做的是第 2 步（B）
        assert store.load(pid).status == "active"

    @pytest.mark.asyncio
    async def test_plan_complete_accepts_completed_and_skipped(self, plan_tool):
        tool, store = plan_tool
        pid = await self._plan(tool, store, _s("A", "B"))
        await tool.execute(operation="update_step", plan_id=pid, step=1,
                           step_status="completed", evidence=_ev())
        await tool.execute(operation="update_step", plan_id=pid, step=2,
                           step_status="skipped", step_output="范围外")
        result = await tool.execute(operation="complete", plan_id=pid)
        assert "completed" in result
        assert store.load(pid).status == "completed"

    @pytest.mark.asyncio
    async def test_abandon_has_no_gate(self, plan_tool):
        tool, store = plan_tool
        pid = await self._plan(tool, store, _s("A"))
        await tool.execute(operation="abandon", plan_id=pid)
        assert store.load(pid).status == "abandoned"

    @pytest.mark.asyncio
    async def test_get_shows_evidence_text(self, plan_tool):
        tool, store = plan_tool
        pid = await self._plan(tool, store, _s("A"))
        await tool.execute(operation="update_step", plan_id=pid, step=1, evidence=_ev("tests/x.py:12"))
        assert "tests/x.py:12" in await tool.execute(operation="get", plan_id=pid)

    @pytest.mark.asyncio
    async def test_legacy_plan_without_criteria_still_works(self, plan_tool):
        """旧版计划文件（没有 criteria/evidence 字段）照常读取、标完成、关闭。"""
        import json
        tool, store = plan_tool
        legacy = {
            "id": "legacy0001", "title": "旧计划", "status": "active",
            "created_at": "2026-11-01T00:00:00+00:00", "updated_at": "2026-11-01T00:00:00+00:00",
            "steps": [{"index": 0, "description": "老步骤", "status": "pending", "output": ""}],
        }
        (store.base_dir / "legacy0001.json").write_text(json.dumps(legacy, ensure_ascii=False), encoding="utf-8")
        assert "老步骤" in await tool.execute(operation="get", plan_id="legacy0001")
        r = await tool.execute(operation="update_step", plan_id="legacy0001", step=1, step_status="completed")
        assert "已更新" in r
        assert "completed" in await tool.execute(operation="complete", plan_id="legacy0001")


# ── 第 23.1 项：方案字段 ──

import json as _json  # noqa: E402

_APPROACH = "## 背景与范围\n只做 Windows 第一版。\n## 关键决定\n- 插入用剪贴板：放弃逐字 SendInput（中文输入法拦截）"


@pytest.mark.asyncio
async def test_create_requires_approach(plan_tool):
    tool, store = plan_tool
    result = await tool.execute(operation="create", title="T", steps=_s("A"))
    assert "需要 approach" in result
    assert store.list_plans() == []          # 没建成，不留半个计划


@pytest.mark.asyncio
async def test_approach_persisted_and_in_get(plan_tool):
    tool, store = plan_tool
    created = await tool.execute(operation="create", title="T", steps=_s("A"), approach=_APPROACH)
    assert "方案已保存" in created and "放弃逐字" not in created   # 创建结果不回显全文
    plan = store.list_plans()[0]
    assert plan.approach == _APPROACH
    got = await tool.execute(operation="get", plan_id=plan.id)
    assert got.index("放弃逐字 SendInput") < got.index("A")      # 方案在步骤前、全文


def test_old_plan_file_without_approach_loads(tmp_path):
    store = PlanStore(base_dir=tmp_path / "plans")
    plan = Plan(title="旧", steps=[PlanStep(index=0, description="A", criteria=["c"])])
    store.save(plan)
    path = next((tmp_path / "plans").glob("*.json"))
    data = _json.loads(path.read_text(encoding="utf-8"))
    data.pop("approach")
    path.write_text(_json.dumps(data, ensure_ascii=False), encoding="utf-8")
    assert store.load(plan.id).approach == ""


def test_compact_puts_brief_approach_last_and_keeps_steps_within_budget(tmp_path):
    store = PlanStore(base_dir=tmp_path / "plans")
    steps = [PlanStep(index=i, description=f"步骤{i}" + "x" * 90, criteria=["c"]) for i in range(8)]
    store.save(Plan(title="T", approach="开头的决定" + "长" * 5000, steps=steps))
    text = store.format_for_prompt(max_chars=2000)
    assert "步骤7" in text                    # 方案再长也不挤掉步骤
    assert text.index("步骤7") < text.index("开头的决定")
    assert len(text) <= 2000 + 40
