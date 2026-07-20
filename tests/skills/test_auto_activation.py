import pytest
from pathlib import Path

from loongcli.skills.registry import (
    SkillRegistry, SkillMeta, MAX_AUTO_SKILLS, MAX_SKILL_CHARS,
    MAX_INPUT_FOR_TRIGGERS, _parse_skill_file,
)


BRAINSTORM_SKILL = """\
---
name: brainstorm
description: 头脑风暴工作流
triggers:
  - 我有个想法
  - brainstorm
  - 头脑风暴
---

## 步骤
1. 发散思维
2. 收敛评估
3. 优先级排序
"""

DEBUG_SKILL = """\
---
name: debug
description: 系统化调试工作流
triggers:
  - bug
  - 调试
  - debug
---

## 调试步骤
1. 复现问题
2. 定位根因
3. 修复并验证
"""

NO_TRIGGERS_SKILL = """\
---
name: plain
description: 没有 triggers 的技能
---

普通内容。
"""

EMPTY_TRIGGERS_SKILL = """\
---
name: empty-triggers
description: triggers 为空列表
triggers: []
---

内容。
"""


def _make_skill(base: Path, name: str, content: str):
    skill_dir = base / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")


class TestTriggersParser:
    def test_parse_triggers(self, tmp_path):
        f = tmp_path / "SKILL.md"
        f.write_text(BRAINSTORM_SKILL, encoding="utf-8")
        meta = _parse_skill_file(f)
        assert meta is not None
        assert meta.triggers == ["我有个想法", "brainstorm", "头脑风暴"]

    def test_no_triggers_field(self, tmp_path):
        f = tmp_path / "SKILL.md"
        f.write_text(NO_TRIGGERS_SKILL, encoding="utf-8")
        meta = _parse_skill_file(f)
        assert meta is not None
        assert meta.triggers == []

    def test_empty_triggers(self, tmp_path):
        f = tmp_path / "SKILL.md"
        f.write_text(EMPTY_TRIGGERS_SKILL, encoding="utf-8")
        meta = _parse_skill_file(f)
        assert meta is not None
        assert meta.triggers == []

    def test_triggers_lowercased(self, tmp_path):
        skill = """\
---
name: test-case
description: test
triggers:
  - BugFix
  - DEBUG
---
content
"""
        f = tmp_path / "SKILL.md"
        f.write_text(skill, encoding="utf-8")
        meta = _parse_skill_file(f)
        assert meta.triggers == ["bugfix", "debug"]


class TestMatch:
    def _build_registry(self, tmp_path, skills: dict[str, str]) -> SkillRegistry:
        skills_dir = tmp_path / ".loongcli" / "skills"
        for name, content in skills.items():
            _make_skill(skills_dir, name, content)
        return SkillRegistry(project_dir=tmp_path, personal_dir=tmp_path / "empty")

    def test_match_single(self, tmp_path):
        reg = self._build_registry(tmp_path, {
            "brainstorm": BRAINSTORM_SKILL,
            "debug": DEBUG_SKILL,
        })
        matched = reg.match("我有个想法，做一个新功能")
        assert len(matched) == 1
        assert matched[0].name == "brainstorm"

    def test_match_case_insensitive(self, tmp_path):
        reg = self._build_registry(tmp_path, {"debug": DEBUG_SKILL})
        matched = reg.match("遇到一个 BUG，帮我看看")
        assert len(matched) == 1
        assert matched[0].name == "debug"

    def test_no_match(self, tmp_path):
        reg = self._build_registry(tmp_path, {"debug": DEBUG_SKILL})
        matched = reg.match("帮我写个函数")
        assert matched == []

    def test_no_triggers_not_matched(self, tmp_path):
        reg = self._build_registry(tmp_path, {"plain": NO_TRIGGERS_SKILL})
        matched = reg.match("plain text here")
        assert matched == []

    def test_max_auto_skills_limit(self, tmp_path):
        skills = {}
        for i in range(5):
            skills[f"skill-{i}"] = f"""\
---
name: skill-{i}
description: Skill {i}
triggers:
  - common-keyword
---
Content {i}
"""
        reg = self._build_registry(tmp_path, skills)
        matched = reg.match("common-keyword test")
        assert len(matched) == MAX_AUTO_SKILLS

    def test_match_chinese_triggers(self, tmp_path):
        reg = self._build_registry(tmp_path, {"brainstorm": BRAINSTORM_SKILL})
        matched = reg.match("来一场头脑风暴吧")
        assert len(matched) == 1

    def test_long_input_skips_triggers(self, tmp_path):
        """Long user input (e.g. interview answers) should not trigger skills."""
        reg = self._build_registry(tmp_path, {"debug": DEBUG_SKILL})
        long_input = "这是一段很长的面试回答内容，" * 20 + "其中偶然提到了 bug 这个词"
        assert len(long_input) > MAX_INPUT_FOR_TRIGGERS
        matched = reg.match(long_input)
        assert matched == []

    def test_short_input_still_triggers(self, tmp_path):
        """Short input should still trigger as before."""
        reg = self._build_registry(tmp_path, {"debug": DEBUG_SKILL})
        matched = reg.match("遇到一个 bug")
        assert len(matched) == 1


class TestEnrichPrompt:
    def _build_registry(self, tmp_path, skills: dict[str, str]) -> SkillRegistry:
        skills_dir = tmp_path / ".loongcli" / "skills"
        for name, content in skills.items():
            _make_skill(skills_dir, name, content)
        return SkillRegistry(project_dir=tmp_path, personal_dir=tmp_path / "empty")

    def test_no_match_returns_original(self, tmp_path):
        reg = self._build_registry(tmp_path, {"debug": DEBUG_SKILL})
        original = "帮我写个函数"
        assert reg.enrich_prompt(original) == original

    def test_match_appends_content(self, tmp_path):
        reg = self._build_registry(tmp_path, {"debug": DEBUG_SKILL})
        enriched = reg.enrich_prompt("遇到一个 bug")
        assert enriched.startswith("遇到一个 bug")
        assert "[自动加载 skill: debug]" in enriched
        assert "复现问题" in enriched

    def test_content_truncated(self, tmp_path):
        long_content = "x" * (MAX_SKILL_CHARS + 1000)
        skill = f"""\
---
name: long-skill
description: 超长技能
triggers:
  - long
---

{long_content}
"""
        reg = self._build_registry(tmp_path, {"long-skill": skill})
        enriched = reg.enrich_prompt("long test")
        assert "截断" in enriched
        body_start = enriched.index("[自动加载 skill: long-skill]")
        body = enriched[body_start:]
        assert len(body) < MAX_SKILL_CHARS + 200

    def test_frontmatter_stripped(self, tmp_path):
        reg = self._build_registry(tmp_path, {"brainstorm": BRAINSTORM_SKILL})
        enriched = reg.enrich_prompt("我有个想法")
        assert "name:" not in enriched.split("---", 2)[-1]
        assert "triggers:" not in enriched.split("[自动加载")[1]


class TestAgentIntegration:
    @pytest.mark.asyncio
    async def test_skill_injected_into_messages(self, tmp_path):
        from unittest.mock import AsyncMock, MagicMock
        from loongcli.core.agent import AgentLoop, AgentServices
        from loongcli.core.events import AgentDone
        from loongcli.security.permissions import PermissionChecker

        skills_dir = tmp_path / ".loongcli" / "skills"
        _make_skill(skills_dir, "debug", DEBUG_SKILL)
        reg = SkillRegistry(project_dir=tmp_path, personal_dir=tmp_path / "empty")

        llm = MagicMock()

        async def fake_stream(*args, **kwargs):
            yield {"type": "text", "content": "OK"}
            yield {"type": "done", "content": "OK", "tool_calls": [],
                   "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
                   "prompt_cache_hit_tokens": 0, "prompt_cache_miss_tokens": 0,
                   "reasoning_tokens": 0}

        from loongcli.core.stream_collector import CollectedResponse
        from loongcli.core.events import TextDelta

        mock_response = CollectedResponse()
        mock_response.content = "OK"
        mock_response.prompt_tokens = 10
        mock_response.completion_tokens = 5
        mock_response.total_tokens = 15

        class FakeCollector:
            def __init__(self):
                self.response = mock_response

            async def collect(self, stream):
                yield TextDelta(text="OK")

        import loongcli.core.agent as agent_mod
        original_collector = agent_mod.StreamCollector

        agent = AgentLoop(
            llm=llm,
            tool_registry=MagicMock(get_tool_schemas=MagicMock(return_value=None)),
            permission_checker=PermissionChecker(),
            services=AgentServices(skill_registry=reg),
        )

        agent_mod.StreamCollector = FakeCollector
        try:
            events = []
            async for event in agent.run_stream("遇到一个 bug"):
                events.append(event)

            user_msg = agent.messages[0]
            assert user_msg["role"] == "user"
            assert "[自动加载 skill: debug]" in user_msg["content"]
            assert "复现问题" in user_msg["content"]
        finally:
            agent_mod.StreamCollector = original_collector

    @pytest.mark.asyncio
    async def test_no_skill_registry_passthrough(self, tmp_path):
        from unittest.mock import MagicMock
        from loongcli.core.agent import AgentLoop, AgentServices
        from loongcli.core.stream_collector import CollectedResponse
        from loongcli.core.events import TextDelta
        from loongcli.security.permissions import PermissionChecker
        import loongcli.core.agent as agent_mod

        mock_response = CollectedResponse()
        mock_response.content = "OK"
        mock_response.prompt_tokens = 10
        mock_response.completion_tokens = 5
        mock_response.total_tokens = 15

        class FakeCollector:
            def __init__(self):
                self.response = mock_response
            async def collect(self, stream):
                yield TextDelta(text="OK")

        agent = AgentLoop(
            llm=MagicMock(),
            tool_registry=MagicMock(get_tool_schemas=MagicMock(return_value=None)),
            permission_checker=PermissionChecker(),
        )

        original = agent_mod.StreamCollector
        agent_mod.StreamCollector = FakeCollector
        try:
            async for _ in agent.run_stream("hello world"):
                pass
            assert agent.messages[0]["content"] == "hello world"
        finally:
            agent_mod.StreamCollector = original
