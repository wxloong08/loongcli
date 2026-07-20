from unittest.mock import MagicMock
from loongcli.core.prompts import get_system_prompt


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


def test_prompt_contains_tool_descriptions():
    prompt = get_system_prompt(model="deepseek-v4-flash")
    for tool in ["read_file", "write_file", "edit_file", "shell", "glob", "grep",
                  "recall", "memorize", "delegate", "batch_delegate",
                  "task_status", "send_message"]:
        assert tool in prompt


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
    skills.build_listing.return_value = "debug, tdd"
    prompt = get_system_prompt(model="deepseek-v4-flash", skills=skills)
    assert "# 可用技能" in prompt
    assert "debug, tdd" in prompt
    # 对模型隐藏 disable-model-invocation 的技能——清单从工具 description 挪来时语义不变
    skills.build_listing.assert_called_once_with(include_model_disabled=False)


def test_prompt_no_skills_section_when_empty():
    skills = MagicMock(spec=["build_listing"])
    skills.build_listing.return_value = ""
    prompt = get_system_prompt(model="deepseek-v4-flash", skills=skills)
    assert "# 可用技能" not in prompt


def test_prompt_skills_section_position():
    # 按易变性排序：MCP（手写极少变）→ 技能清单（手动增删才变）→ 记忆索引（自动抽取会变）
    memory = MagicMock(spec=["get_index"])
    memory.get_index.return_value = "- [user-role](user-role.md) — Python developer"
    mcp = MagicMock()
    mcp.get_tool_descriptions.return_value = "MCP: searxng tools available"
    skills = MagicMock(spec=["build_listing"])
    skills.build_listing.return_value = "debug, tdd"
    prompt = get_system_prompt(model="deepseek-v4-flash", memory=memory, mcp=mcp, skills=skills)
    assert prompt.index("searxng") < prompt.index("# 可用技能") < prompt.index("记忆索引")


def test_prompt_no_memory_when_empty():
    memory = MagicMock(spec=["get_index"])
    memory.get_index.return_value = ""
    prompt = get_system_prompt(model="deepseek-v4-flash", memory=memory)
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
