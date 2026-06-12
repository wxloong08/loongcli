"""压缩后活跃技能原文重注入（契约类内容结构性保活）。"""
from __future__ import annotations

from loongcli.core.attachments import (
    ATTACHMENT_MARKER,
    MAX_SKILL_REINJECT_CHARS,
    build_attachments,
    skill_section,
)


class FakeRegistry:
    def __init__(self, skills: dict[str, str]):
        self._skills = skills

    def load_content(self, name: str) -> str | None:
        return self._skills.get(name)


class TestSkillSection:
    def test_reinjects_skill_content(self):
        reg = FakeRegistry({"jobhunter": "第一步：打开招聘网站\n第二步：筛选职位"})
        text = skill_section(reg, "jobhunter")
        assert "jobhunter" in text
        assert "第一步：打开招聘网站" in text
        assert "原文已重新挂载" in text

    def test_no_registry_or_skill(self):
        assert skill_section(None, "jobhunter") == ""
        assert skill_section(FakeRegistry({}), None) == ""
        assert skill_section(FakeRegistry({}), "missing") == ""

    def test_truncates_huge_skill(self):
        reg = FakeRegistry({"big": "x" * (MAX_SKILL_REINJECT_CHARS + 1000)})
        text = skill_section(reg, "big")
        assert "已截断" in text
        assert len(text) < MAX_SKILL_REINJECT_CHARS + 300


class TestBuildAttachmentsWithSkill:
    def test_skill_included_in_attachments(self):
        reg = FakeRegistry({"jobhunter": "技能工作流细节"})
        result = build_attachments([], skill_registry=reg, active_skill="jobhunter")
        assert len(result) == 2
        assert ATTACHMENT_MARKER in result[0]["content"]
        assert "技能工作流细节" in result[0]["content"]

    def test_no_skill_no_section(self):
        assert build_attachments([]) == []
