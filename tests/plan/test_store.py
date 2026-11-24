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


def test_format_summary_marks_criteria_evidence():
    plan = Plan(title="T", steps=[
        PlanStep(index=0, description="录音", criteria=["WAV 头一致", "界面不写盘"], evidence=["test_wav ok", ""]),
    ])
    s = plan.format_summary()
    assert "判据 1 [✓] WAV 头一致" in s            # 对外编号从 1 起
    assert "判据 2 [○] 界面不写盘" in s
    assert "test_wav ok" not in s                    # 默认不回显证据原文
    assert "证据：test_wav ok" in plan.format_summary(show_evidence=True)


def test_evidence_normalized_to_criteria_length():
    """手改过的文件证据条数和判据不一致：补齐/截断，按下标取不越界。"""
    assert PlanStep(index=0, description="d", criteria=["a", "b"]).evidence == ["", ""]
    assert PlanStep(index=0, description="d", criteria=["a"], evidence=["x", "y"]).evidence == ["x"]


def _exec4_scale_plan() -> Plan:
    """exec-4 规模：8 步、每步 5 条长判据，完整摘要约 6500 字符。"""
    steps = [
        PlanStep(index=i, description=f"第 {i} 步：" + "实现并验证某个模块的具体改动内容" * 4,
                 status="completed" if i < 3 else "pending",
                 criteria=[f"判据{i}-{j}：" + "运行某条命令并看到某个具体输出" * 3 for j in range(5)],
                 evidence=[f"证据{i}-{j}" for j in range(5)] if i < 3 else [])
        for i in range(8)
    ]
    return Plan(title="Windows 语音输入 v1：P1+P2", steps=steps)


def test_format_for_prompt_never_drops_large_plan(store):
    """回归：判据让计划摘要超出 2000 字符预算后，曾被整份丢弃、只剩「... (还有 1 个活跃计划)」。"""
    plan = _exec4_scale_plan()
    assert len(plan.format_summary()) > 2000      # 完整版确实超预算
    store.save(plan)
    text = store.format_for_prompt(max_chars=2000)
    assert "Windows 语音输入 v1" in text
    assert "当前步骤 4 的判据" in text            # 当前步骤的判据在
    assert "判据3-0" in text
    assert "还有 1 个活跃计划" not in text
    assert len(text) <= 2000 + 40


def test_format_compact_only_expands_current_step():
    plan = _exec4_scale_plan()
    compact = plan.format_compact()
    assert "判据3-0" in compact
    assert "判据4-0" not in compact               # 后面步骤的判据不展开
    assert "判据0-0" not in compact               # 已完成步骤的判据不展开
    assert f"id {plan.id}" in compact


def test_format_for_prompt_truncates_oversized_first_plan(store):
    plan = Plan(title="超长", steps=[PlanStep(index=0, description="x" * 90, criteria=["y" * 3000])])
    store.save(plan)
    text = store.format_for_prompt(max_chars=500)
    assert "超长" in text
    assert "已截断" in text


def test_closure_problems():
    plan = Plan(title="T", steps=[
        PlanStep(index=0, description="a", status="completed", criteria=["c"], evidence=["e"]),
        PlanStep(index=1, description="b", status="completed", criteria=["c0", "c1"], evidence=["e", ""]),
        PlanStep(index=2, description="c", status="skipped"),
        PlanStep(index=3, description="d", status="in_progress"),
    ])
    problems = plan.closure_problems()
    assert len(problems) == 3
    assert "步骤 2 判据 2 没有证据" in problems[0]    # 对外编号从 1 起
    assert "步骤 3 标了跳过但没写理由" in problems[1]
    assert "步骤 4 未完成" in problems[2]
