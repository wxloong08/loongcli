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

    async def test_big_skill_under_cap_returned_intact(self, tmp_path):
        # 技能是指令：>8K 的技能（现存 8 个）必须完整送达，不许被入口截成 2000 预览
        # ——入口豁免（SELF_BUDGETED_TOOLS 含 skill）配合本工具 50K 自管上限
        from loongcli.tools.skill import SKILL_CONTENT_CAP

        body = "## 步骤\n" + "- 关键规则一行\n" * 3000  # ~27K，介于 8K 与 50K 之间
        big_skill = f"---\nname: big\ndescription: 大技能\n---\n\n{body}"
        reg = _make_registry(tmp_path, {"big": big_skill})
        tool = SkillTool(reg)
        result = await tool.execute(name="big")
        assert body.rstrip("\n") in result  # 原文完整在场
        assert "已截断" not in result
        assert len(big_skill) > 8000  # 前提自检：确实超过入口单条上限

    async def test_oversize_skill_capped_with_file_pointer(self, tmp_path):
        # 超过 50K 的极端技能：截断 + 指向源文件分页读取，绝不建议重调 skill（同结果死循环）
        from loongcli.tools.skill import SKILL_CONTENT_CAP

        body = "规则行\n" * 15000  # ~60K
        huge_skill = f"---\nname: huge\ndescription: 超大技能\n---\n\n{body}"
        reg = _make_registry(tmp_path, {"huge": huge_skill})
        tool = SkillTool(reg)
        result = await tool.execute(name="huge")
        assert len(result) < SKILL_CONTENT_CAP + 500
        assert "技能原文超长已截断" in result
        assert "SKILL.md" in result  # 恢复指针指向源文件
        assert "read_file" in result
        assert "重新调用工具" not in result
