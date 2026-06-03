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
    assert "# 技能执行规则" in prompt
    assert "# MCP 工具使用" in prompt
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
    assert "技能执行规则" in prompt
    assert "严格遵循" in prompt
    assert "references/" in prompt


def test_prompt_mcp_usage():
    prompt = get_system_prompt(model="deepseek-v4-flash")
    assert "MCP 工具使用" in prompt
    assert "language" in prompt
    assert "zh" in prompt


def test_prompt_identity_general_purpose():
    prompt = get_system_prompt(model="deepseek-v4-flash")
    assert "通用 AI Agent" in prompt
    assert "信息检索" in prompt


def test_prompt_compact_recovery_guidance():
    prompt = get_system_prompt(model="deepseek-v4-flash")
    assert "技能（skill）" in prompt
    assert "重新加载" in prompt


def test_prompt_plan_usage():
    prompt = get_system_prompt(model="deepseek-v4-flash")
    assert "计划（Plan）使用" in prompt
    assert "步骤粒度" in prompt
    assert "plan(complete)" in prompt
    assert "step_output" in prompt
