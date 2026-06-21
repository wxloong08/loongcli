import pytest
from loongcli.tools.plan_tool import PlanTool
from loongcli.tools.base import ToolRegistry
from loongcli.plan.store import PlanStore, Plan, PlanStep
import loongcli.plan.store as store_mod


@pytest.fixture
def plan_tool(tmp_path):
    store = PlanStore(base_dir=tmp_path / "plans")
    return PlanTool(store), store


class TestPlanTool:
    @pytest.mark.asyncio
    async def test_create(self, plan_tool):
        tool, store = plan_tool
        result = await tool.execute(
            operation="create",
            title="Auth refactor",
            steps=["Analyze", "Test", "Refactor"],
        )
        assert "已创建" in result
        plans = store.list_plans()
        assert len(plans) == 1
        assert plans[0].title == "Auth refactor"
        assert len(plans[0].steps) == 3

    @pytest.mark.asyncio
    async def test_create_requires_title(self, plan_tool):
        tool, _ = plan_tool
        result = await tool.execute(operation="create", steps=["A"])
        assert "错误" in result

    @pytest.mark.asyncio
    async def test_create_requires_steps(self, plan_tool):
        tool, _ = plan_tool
        result = await tool.execute(operation="create", title="Test")
        assert "错误" in result

    @pytest.mark.asyncio
    async def test_update_step(self, plan_tool):
        tool, store = plan_tool
        result = await tool.execute(
            operation="create", title="Test", steps=["A", "B"],
        )
        plan_id = store.list_plans()[0].id

        result = await tool.execute(
            operation="update_step", plan_id=plan_id,
            step_index=0, step_status="completed", step_output="done",
        )
        assert "已更新" in result
        assert "1/2" in result

        plan = store.load(plan_id)
        assert plan.steps[0].status == "completed"
        assert plan.steps[0].output == "done"

    @pytest.mark.asyncio
    async def test_update_step_invalid_index(self, plan_tool):
        tool, store = plan_tool
        await tool.execute(operation="create", title="T", steps=["A"])
        plan_id = store.list_plans()[0].id
        result = await tool.execute(
            operation="update_step", plan_id=plan_id, step_index=5,
        )
        assert "超出范围" in result

    @pytest.mark.asyncio
    async def test_update_step_missing_plan(self, plan_tool):
        tool, _ = plan_tool
        result = await tool.execute(
            operation="update_step", plan_id="nope", step_index=0,
        )
        assert "未找到" in result

    @pytest.mark.asyncio
    async def test_get_specific(self, plan_tool):
        tool, store = plan_tool
        await tool.execute(operation="create", title="My Plan", steps=["A", "B"])
        plan_id = store.list_plans()[0].id
        result = await tool.execute(operation="get", plan_id=plan_id)
        assert "My Plan" in result

    @pytest.mark.asyncio
    async def test_get_active(self, plan_tool):
        tool, _ = plan_tool
        await tool.execute(operation="create", title="Plan A", steps=["X"])
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
        await tool.execute(operation="create", title="T", steps=["A"])
        plan_id = store.list_plans()[0].id
        result = await tool.execute(operation="complete", plan_id=plan_id)
        assert "completed" in result
        assert store.load(plan_id).status == "completed"

    @pytest.mark.asyncio
    async def test_abandon(self, plan_tool):
        tool, store = plan_tool
        await tool.execute(operation="create", title="T", steps=["A"])
        plan_id = store.list_plans()[0].id
        result = await tool.execute(operation="abandon", plan_id=plan_id)
        assert "abandoned" in result

    @pytest.mark.asyncio
    async def test_list(self, plan_tool):
        tool, _ = plan_tool
        await tool.execute(operation="create", title="Plan A", steps=["X"])
        await tool.execute(operation="create", title="Plan B", steps=["Y"])
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
