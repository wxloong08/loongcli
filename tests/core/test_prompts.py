from unittest.mock import MagicMock
from loongcli.core.prompts import get_system_prompt
from loongcli.core.git_context import GitContext

# 环境段会注入 git 最近提交记录，提交标题可能恰好含被断言的字符串（如某次提交标题
# 带「记忆索引」四字）。凡断言「某字符串不在提示词里」或按位置比较的测试，都传一个
# 确定的非仓库 git_ctx 隔离掉 git log，否则测试会因无关的提交历史随机变红。
_CLEAN_GIT = GitContext(is_repo=False)


def test_prompt_contains_identity():
    prompt = get_system_prompt(model="deepseek-v4-flash")
    assert "loongcli" in prompt


def test_prompt_contains_all_sections():
    prompt = get_system_prompt(model="deepseek-v4-flash")
    assert "# 系统规则" in prompt
    assert "# 执行任务" in prompt
    assert "# 谨慎操作" in prompt
    assert "# 使用工具" in prompt
    assert "# 计划（Plan）使用" in prompt
    assert "浓缩指令集" in prompt  # skill 段不混为普通规则
    assert "# 外部工具（MCP）" in prompt
    assert "# 交流风格" in prompt
    assert "# 环境" in prompt


def test_actions_section_covers_live_desktop_side_effects():
    # understand-1 事故：「核实环境」时脚本合成全局 Ctrl 按键、被中断后 Ctrl 卡住。
    # 高风险清单原只有仓库/共享环境类，缺「作用于用户正在用的电脑和外设」这一类
    prompt = get_system_prompt(model="deepseek-v4-flash", git_ctx=_CLEAN_GIT)
    section = prompt.split("# 谨慎操作", 1)[1].split("\n# ", 1)[0]
    for word in ("模拟按键", "全局键盘", "麦克风", "注册表", "写在脚本里一样算", "只读查询"):
        assert word in section, word


def test_prompt_contains_tool_descriptions():
    prompt = get_system_prompt(model="deepseek-v4-flash")
    for tool in ["read_file", "write_file", "edit_file", "shell",
                  "recall", "memorize", "delegate", "batch_delegate", "task_status"]:
        assert tool in prompt
    # grep/glob/send_message 不再注册（2026-11-19）：提示里不能再指引模型去调它们，搜索改走 shell
    assert "- glob:" not in prompt and "- grep:" not in prompt and "send_message" not in prompt
    assert "git grep" in prompt


def test_prompt_contains_model_name():
    prompt = get_system_prompt(model="deepseek-v4-pro")
    assert "deepseek-v4-pro" in prompt


def test_prompt_includes_memory():
    memory = MagicMock(spec=["get_index"])
    memory.get_index.return_value = "- [user-role](user-role.md) — Python developer"
    prompt = get_system_prompt(model="deepseek-v4-flash", memory=memory)
    assert "user-role" in prompt
    assert "记忆索引" in prompt


def test_prompt_includes_mcp():
    mcp = MagicMock()
    mcp.get_tool_descriptions.return_value = "MCP: searxng tools available"
    prompt = get_system_prompt(model="deepseek-v4-flash", mcp=mcp)
    assert "searxng" in prompt


def test_prompt_includes_skills_listing():
    skills = MagicMock(spec=["build_listing"])
    skills.build_listing.return_value = "- debug: 调试工作流\n- tdd: 测试驱动开发"
    prompt = get_system_prompt(model="deepseek-v4-flash", skills=skills)
    assert "# 可用技能" in prompt
    assert "- debug: 调试工作流" in prompt
    # 对模型隐藏 disable-model-invocation 的技能——清单从工具 description 挪来时语义不变；
    # verbose=True 带一句话描述（自主触发的语义匹配靠它）
    skills.build_listing.assert_called_once_with(include_model_disabled=False, verbose=True)


def test_prompt_no_skills_section_when_empty():
    skills = MagicMock(spec=["build_listing"])
    skills.build_listing.return_value = ""
    prompt = get_system_prompt(model="deepseek-v4-flash", skills=skills, git_ctx=_CLEAN_GIT)
    assert "# 可用技能" not in prompt


def test_prompt_skills_section_position():
    # 按易变性排序：MCP（手写极少变）→ 技能清单（手动增删才变）→ 记忆索引（自动抽取会变）
    memory = MagicMock(spec=["get_index"])
    memory.get_index.return_value = "- [user-role](user-role.md) — Python developer"
    mcp = MagicMock()
    mcp.get_tool_descriptions.return_value = "MCP: searxng tools available"
    skills = MagicMock(spec=["build_listing"])
    skills.build_listing.return_value = "- debug: 调试工作流\n- tdd: 测试驱动开发"
    prompt = get_system_prompt(model="deepseek-v4-flash", memory=memory, mcp=mcp, skills=skills,
                               git_ctx=_CLEAN_GIT)
    assert prompt.index("searxng") < prompt.index("# 可用技能") < prompt.index("记忆索引")


def test_prompt_no_memory_when_empty():
    memory = MagicMock(spec=["get_index"])
    memory.get_index.return_value = ""
    prompt = get_system_prompt(model="deepseek-v4-flash", memory=memory, git_ctx=_CLEAN_GIT)
    assert "记忆系统" in prompt
    assert "记忆索引" not in prompt


def test_prompt_environment_has_cwd():
    prompt = get_system_prompt(model="deepseek-v4-flash")
    assert "工作目录" in prompt


def test_prompt_skill_discipline():
    prompt = get_system_prompt(model="deepseek-v4-flash")
    assert "浓缩指令集" in prompt  # 不再用 CC 标签而是自然段
    assert "第一件事" in prompt  # 读取 skill 是硬顺序——调其他工具之前
    assert "SKILL.md" in prompt


def test_prompt_mcp_usage():
    # loongcli 核心提示应保持 MCP-无关：只给通用工具纪律，不硬编码具体 MCP 工具名/参数
    # （searxng 等工具的具体用法由各 MCP 自带描述提供）。直接查该节，避免误命中
    # 动态注入的 git 提交信息等无关内容。
    from loongcli.core.prompts import _mcp_usage_section
    section = _mcp_usage_section()
    assert "# 外部工具（MCP）" in section
    assert "搜索止损" in section
    assert "searxng" not in section   # 不耦合私有 MCP
    assert "language=" not in section


def test_prompt_identity_general_purpose():
    prompt = get_system_prompt(model="deepseek-v4-flash")
    assert "通用 AI Agent" in prompt
    assert "信息检索" in prompt


def test_prompt_identity_model_agnostic():
    # 身份不应硬编码具体 provider——loongcli 支持多 provider（动态模型名在环境节）。
    from loongcli.core.prompts import _identity_section
    assert "DeepSeek" not in _identity_section()
    assert "大语言模型" in _identity_section()


def test_prompt_critical_thinking_principles():
    # 通用推理原则：拉波波特法则（反稻草人）+ 同一思路失败两次止损（反 flailing）。
    prompt = get_system_prompt(model="deepseek-v4-flash")
    assert "拉波波特" in prompt
    assert "失败两次就停" in prompt


def test_prompt_compact_recovery_guidance():
    prompt = get_system_prompt(model="deepseek-v4-flash")
    assert "技能（skill）" in prompt
    assert "重新加载" in prompt


def test_prompt_content_accuracy_clause():
    """Must instruct against fabricating/inflating facts & numbers in output."""
    prompt = get_system_prompt(model="deepseek-v4-flash")
    assert "以来源为准" in prompt
    assert "不臆造" in prompt


def test_prompt_synthesis_discipline_clause():
    """主代理合成纪律：负面信号禁丢、矛盾须裁决——百得思维事故（合成层丢弃不利证据）。"""
    prompt = get_system_prompt(model="deepseek-v4-flash")
    assert "负面信号" in prompt
    assert "禁止静默丢弃" in prompt
    assert "正面与负面同权保留" in prompt
    assert "免责声明" in prompt


def test_prompt_plan_usage():
    prompt = get_system_prompt(model="deepseek-v4-flash")
    assert "计划（Plan）使用" in prompt
    assert "步骤粒度" in prompt
    assert "plan(complete)" in prompt
    assert "step_output" in prompt


# ── 会话级快照族：前缀缓存的承重不变量（域 5 拷问产出）────────────


class TestSessionPinnedSnapshots:
    """git_ctx / plan_text / memory_index 三个源固定后，系统提示词字节恒定。

    _rebuild_system_prompt 在计划执行期每轮触发且重拼整个提示词。这三个源
    任一每轮现采，messages[0] 就每轮都变，前缀缓存连同其后的全部对话历史
    一起作废。测试用「每次调用都返回不同值」的假源，验证固定住之后渲染仍
    字节一致——若哪天有人把某个源改回现采，这条会红。
    """

    def _drifting_memory(self):
        calls = {"n": 0}
        memory = MagicMock()

        def drift(max_bytes=25_000):
            calls["n"] += 1
            return f"- mem-{calls['n']} — 描述"

        memory.get_index = drift
        return memory, calls

    def _drifting_plan_store(self):
        calls = {"n": 0}
        store = MagicMock()

        def drift(*a, **kw):
            calls["n"] += 1
            return f"**Plan** ({calls['n']}/3 完成)"

        store.format_for_prompt = drift
        return store, calls

    def test_pinned_sources_keep_prompt_byte_stable(self, monkeypatch):
        from loongcli.core.git_context import GitContext

        git_calls: list[int] = []

        def drifting_git(cwd=None):
            git_calls.append(1)
            return GitContext(is_repo=True, branch=f"b{len(git_calls)}", main_branch="main")

        monkeypatch.setattr("loongcli.core.prompts.collect_git_context", drifting_git)
        memory, mem_calls = self._drifting_memory()
        store, plan_calls = self._drifting_plan_store()

        pinned = dict(
            git_ctx=GitContext(is_repo=True, branch="pinned", main_branch="main"),
            plan_text="**Plan** (0/3 完成)",
            memory_index="- mem-pinned — 描述",
        )
        first = get_system_prompt(model="m", memory=memory, plan_store=store, **pinned)
        second = get_system_prompt(model="m", memory=memory, plan_store=store, **pinned)

        assert first == second, "三个源都固定后，两次渲染必须字节一致"
        assert git_calls == [] and mem_calls["n"] == 0 and plan_calls["n"] == 0, (
            "传了快照就不该再去现采任何一个源"
        )

    def test_unpinned_sources_still_drift(self, monkeypatch):
        """不传则保持旧行为（现采）——供测试与一次性构建使用。"""
        from loongcli.core.git_context import GitContext

        monkeypatch.setattr(
            "loongcli.core.prompts.collect_git_context",
            lambda cwd=None: GitContext(is_repo=False),
        )
        memory, mem_calls = self._drifting_memory()
        a = get_system_prompt(model="m", memory=memory)
        b = get_system_prompt(model="m", memory=memory)
        assert a != b
        assert mem_calls["n"] == 2

    def test_snapshot_labels_present(self):
        """固定住就必须标注是快照，并指明拿最新状态的通道。"""
        memory = MagicMock()
        memory.get_index = lambda max_bytes=25_000: "- m — d"
        store = MagicMock()
        store.format_for_prompt = lambda *a, **kw: "**Plan** (0/1 完成)"
        prompt = get_system_prompt(
            model="m", memory=memory, plan_store=store,
            plan_text="**Plan** (0/1 完成)", memory_index="- m — d",
        )
        assert "会话开始时的快照" in prompt and "plan get" in prompt
        assert "会话开始时的快照" in prompt and "memorize 的返回值" in prompt
