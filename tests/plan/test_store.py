import pytest
from loongcli.plan.store import PlanStore, Plan, PlanStep


@pytest.fixture
def store(tmp_path):
    return PlanStore(base_dir=tmp_path / "plans")


def test_create_and_load(store):
    plan = Plan(title="Refactor auth", steps=[
        PlanStep(index=0, description="Analyze current code"),
        PlanStep(index=1, description="Write tests"),
        PlanStep(index=2, description="Refactor"),
    ])
    store.save(plan)
    loaded = store.load(plan.id)
    assert loaded is not None
    assert loaded.title == "Refactor auth"
    assert len(loaded.steps) == 3
    assert loaded.steps[0].description == "Analyze current code"


def test_load_nonexistent(store):
    assert store.load("nope") is None


def test_delete(store):
    plan = Plan(title="Test")
    store.save(plan)
    assert store.delete(plan.id) is True
    assert store.load(plan.id) is None


def test_delete_nonexistent(store):
    assert store.delete("nope") is False


def test_list_plans(store):
    p1 = Plan(title="Plan A")
    p2 = Plan(title="Plan B", status="completed")
    store.save(p1)
    store.save(p2)
    all_plans = store.list_plans()
    assert len(all_plans) == 2
    active = store.list_plans(status="active")
    assert len(active) == 1
    assert active[0].title == "Plan A"


def test_get_active_plans(store):
    p1 = Plan(title="Active", status="active")
    p2 = Plan(title="Done", status="completed")
    store.save(p1)
    store.save(p2)
    active = store.get_active_plans()
    assert len(active) == 1
    assert active[0].title == "Active"


def test_progress(store):
    plan = Plan(title="Test", steps=[
        PlanStep(index=0, description="A", status="completed"),
        PlanStep(index=1, description="B", status="in_progress"),
        PlanStep(index=2, description="C", status="pending"),
        PlanStep(index=3, description="D", status="skipped"),
    ])
    done, total = plan.progress()
    assert done == 2
    assert total == 4


def test_format_summary():
    plan = Plan(title="Auth refactor", steps=[
        PlanStep(index=0, description="Analyze", status="completed", output="found 3 files"),
        PlanStep(index=1, description="Test", status="in_progress"),
        PlanStep(index=2, description="Refactor", status="pending"),
    ])
    summary = plan.format_summary()
    assert "Auth refactor" in summary
    assert "1/3" in summary
    assert "✓" in summary
    assert "◉" in summary
    assert "○" in summary
    assert "found 3 files" in summary


def test_format_for_prompt_empty(store):
    assert store.format_for_prompt() == ""


def test_format_for_prompt_with_active(store):
    plan = Plan(title="Test plan", steps=[
        PlanStep(index=0, description="Step A"),
    ])
    store.save(plan)
    text = store.format_for_prompt()
    assert "Test plan" in text
    assert "Step A" in text


def test_format_for_prompt_truncation(store):
    for i in range(20):
        p = Plan(title=f"Plan {i}", steps=[
            PlanStep(index=j, description=f"Step {j} " + "x" * 50)
            for j in range(10)
        ])
        store.save(p)
    text = store.format_for_prompt(max_chars=500)
    assert "还有" in text
    assert len(text) < 700


def test_persistence(tmp_path):
    base = tmp_path / "plans"
    s1 = PlanStore(base_dir=base)
    plan = Plan(title="Persist test", steps=[
        PlanStep(index=0, description="Do it"),
    ])
    s1.save(plan)

    s2 = PlanStore(base_dir=base)
    loaded = s2.load(plan.id)
    assert loaded is not None
    assert loaded.title == "Persist test"


def test_step_status_update(store):
    plan = Plan(title="Update test", steps=[
        PlanStep(index=0, description="First"),
        PlanStep(index=1, description="Second"),
    ])
    store.save(plan)

    loaded = store.load(plan.id)
    loaded.steps[0].status = "completed"
    loaded.steps[0].output = "done"
    store.save(loaded)

    reloaded = store.load(plan.id)
    assert reloaded.steps[0].status == "completed"
    assert reloaded.steps[0].output == "done"
    assert reloaded.steps[1].status == "pending"


def test_format_summary_is_markdown_list():
    """步骤必须是 markdown 列表项——单换行会被 Markdown 渲染折叠成一段（真机：审批面板糊成一行）。"""
    plan = Plan(title="T", steps=[
        PlanStep(index=0, description="改 a.py 的 f()，判据：pytest tests/a 全绿"),
        PlanStep(index=1, description="改 b.py", status="completed", output="+3 行"),
    ])
    lines = plan.format_summary().splitlines()
    assert lines[1] == ""                      # 标题后空行（markdown 段落分隔）
    assert lines[2].startswith("- ○ ")         # 列表项
    assert lines[3].startswith("- ✓ ")
    assert any("产出：+3 行" in l for l in lines)
