import pytest
from pathlib import Path

from loongcli.skills.registry import SkillRegistry
from loongcli.tools.skill import SkillTool, SKILL_TOOL_DESC, NO_SKILLS_DESC


VALID_SKILL = """\
---
name: debug
description: 系统化调试工作流
---

## 步骤
1. 复现问题
2. 修复
"""

USER_ONLY_SKILL = """\
---
name: deploy
description: 部署工作流
disable-model-invocation: true
---

部署步骤。
"""


def _make_registry(tmp_path, skills: dict[str, str]) -> SkillRegistry:
    skills_dir = tmp_path / ".loongcli" / "skills"
    for name, content in skills.items():
        d = skills_dir / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "SKILL.md").write_text(content, encoding="utf-8")
    return SkillRegistry(project_dir=tmp_path, personal_dir=tmp_path / "empty")


class TestSkillTool:
    def test_description_static_without_listing(self, tmp_path):
        # 技能清单不进 description（进系统提示易变段，防打断前缀缓存）——description 是静态指引
        reg = _make_registry(tmp_path, {"debug": VALID_SKILL})
        tool = SkillTool(reg)
        assert tool.description == SKILL_TOOL_DESC
        assert "debug" not in tool.description

    def test_description_no_skills(self, tmp_path):
        reg = SkillRegistry(project_dir=tmp_path, personal_dir=tmp_path / "empty")
        tool = SkillTool(reg)
        assert tool.description == NO_SKILLS_DESC

    def test_description_only_model_disabled_counts_as_none(self, tmp_path):
        # 全部技能都 disable-model-invocation 时，对模型而言等于没有技能
        reg = _make_registry(tmp_path, {"deploy": USER_ONLY_SKILL})
        tool = SkillTool(reg)
        assert tool.description == NO_SKILLS_DESC

    async def test_execute_found(self, tmp_path):
        reg = _make_registry(tmp_path, {"debug": VALID_SKILL})
        tool = SkillTool(reg)
        result = await tool.execute(name="debug")
        assert "复现问题" in result
        assert "技能: debug" in result

    async def test_execute_not_found(self, tmp_path):
        reg = _make_registry(tmp_path, {"debug": VALID_SKILL})
        tool = SkillTool(reg)
        result = await tool.execute(name="nonexistent")
        assert "未找到" in result
        assert "debug" in result

    async def test_execute_not_found_empty(self, tmp_path):
        reg = SkillRegistry(project_dir=tmp_path, personal_dir=tmp_path / "empty")
        tool = SkillTool(reg)
        result = await tool.execute(name="anything")
        assert "未找到" in result
        assert "没有可用技能" in result

    def test_tool_schema(self, tmp_path):
        reg = _make_registry(tmp_path, {"debug": VALID_SKILL})
        tool = SkillTool(reg)
        assert tool.name == "skill"
        assert "name" in tool.parameters["properties"]
        assert "name" in tool.parameters["required"]
