import pytest
from pathlib import Path

from loongcli.skills.registry import SkillRegistry
from loongcli.tools.skill import SkillTool, SKILL_TOOL_PREFIX, NO_SKILLS_DESC


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
    def test_description_with_skills(self, tmp_path):
        reg = _make_registry(tmp_path, {"debug": VALID_SKILL})
        tool = SkillTool(reg)
        assert SKILL_TOOL_PREFIX in tool.description
        assert "debug" in tool.description

    def test_description_no_skills(self, tmp_path):
        reg = SkillRegistry(project_dir=tmp_path, personal_dir=tmp_path / "empty")
        tool = SkillTool(reg)
        assert tool.description == NO_SKILLS_DESC

    def test_description_excludes_model_disabled(self, tmp_path):
        reg = _make_registry(tmp_path, {
            "debug": VALID_SKILL,
            "deploy": USER_ONLY_SKILL,
        })
        tool = SkillTool(reg)
        assert "debug" in tool.description
        assert "deploy" not in tool.description

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
