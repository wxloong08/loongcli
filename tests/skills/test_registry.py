import pytest
from pathlib import Path

from loongcli.skills.registry import (
    SkillRegistry, SkillMeta,
    _parse_frontmatter, _extract_body, _parse_skill_file,
)


VALID_SKILL = """\
---
name: debug
description: 系统化调试工作流。遇到 bug 时使用。
---

## 步骤
1. 复现问题
2. 定位根因
3. 修复并验证
"""

SKILL_WITH_OPTIONS = """\
---
name: code-review
description: 代码审查工作流
disable-model-invocation: true
allowed-tools: read_file grep shell
---

审查代码质量和安全性。
"""

MINIMAL_SKILL = """\
---
name: hello
description: Say hello
---

Hello world!
"""

MISSING_NAME = """\
---
description: no name here
---

content
"""

INVALID_NAME = """\
---
name: Hello-World
description: uppercase not allowed
---

content
"""


class TestParseFrontmatter:
    def test_valid(self):
        fm = _parse_frontmatter(VALID_SKILL)
        assert fm["name"] == "debug"
        assert "调试" in fm["description"]

    def test_with_options(self):
        fm = _parse_frontmatter(SKILL_WITH_OPTIONS)
        assert fm["name"] == "code-review"
        assert fm["disable-model-invocation"] is True
        assert fm["allowed-tools"] == "read_file grep shell"

    def test_no_frontmatter(self):
        fm = _parse_frontmatter("just plain text")
        assert fm == {}

    def test_empty(self):
        fm = _parse_frontmatter("")
        assert fm == {}


class TestExtractBody:
    def test_with_frontmatter(self):
        body = _extract_body(VALID_SKILL)
        assert "## 步骤" in body
        assert "name:" not in body

    def test_no_frontmatter(self):
        text = "just content"
        assert _extract_body(text) == text


class TestParseSkillFile:
    def test_valid_skill(self, tmp_path):
        f = tmp_path / "SKILL.md"
        f.write_text(VALID_SKILL, encoding="utf-8")
        meta = _parse_skill_file(f)
        assert meta is not None
        assert meta.name == "debug"
        assert "调试" in meta.description
        assert meta.path == f
        assert meta.disable_model_invocation is False
        assert meta.allowed_tools == []

    def test_with_options(self, tmp_path):
        f = tmp_path / "SKILL.md"
        f.write_text(SKILL_WITH_OPTIONS, encoding="utf-8")
        meta = _parse_skill_file(f)
        assert meta is not None
        assert meta.disable_model_invocation is True
        assert meta.allowed_tools == ["read_file", "grep", "shell"]

    def test_missing_name(self, tmp_path):
        f = tmp_path / "SKILL.md"
        f.write_text(MISSING_NAME, encoding="utf-8")
        assert _parse_skill_file(f) is None

    def test_invalid_name(self, tmp_path):
        f = tmp_path / "SKILL.md"
        f.write_text(INVALID_NAME, encoding="utf-8")
        assert _parse_skill_file(f) is None

    def test_nonexistent_file(self, tmp_path):
        f = tmp_path / "missing.md"
        assert _parse_skill_file(f) is None


class TestSkillRegistry:
    def _setup_skills(self, base: Path, skills: dict[str, str]):
        for name, content in skills.items():
            if "/" in name:
                skill_dir = base / name.split("/")[0]
                skill_dir.mkdir(parents=True, exist_ok=True)
                (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")
            else:
                (base).mkdir(parents=True, exist_ok=True)
                (base / name).write_text(content, encoding="utf-8")

    def test_scan_directory_format(self, tmp_path):
        skills_dir = tmp_path / ".loongcli" / "skills"
        self._setup_skills(skills_dir, {"debug/": VALID_SKILL})
        reg = SkillRegistry(project_dir=tmp_path, personal_dir=tmp_path / "empty")
        assert reg.get("debug") is not None
        assert reg.get("debug").name == "debug"

    def test_loose_md_files_ignored(self, tmp_path):
        """Only SKILL.md inside directories should be loaded, not loose .md files."""
        skills_dir = tmp_path / ".loongcli" / "skills"
        skills_dir.mkdir(parents=True)
        (skills_dir / "hello.md").write_text(MINIMAL_SKILL, encoding="utf-8")
        (skills_dir / "CLAUDE.md").write_text("# Not a skill", encoding="utf-8")
        reg = SkillRegistry(project_dir=tmp_path, personal_dir=tmp_path / "empty")
        assert reg.get("hello") is None
        assert len(reg.list_skills()) == 0

    def test_scan_personal_dir(self, tmp_path):
        personal = tmp_path / "personal"
        self._setup_skills(personal, {"debug/": VALID_SKILL})
        reg = SkillRegistry(project_dir=None, personal_dir=personal)
        assert reg.get("debug") is not None

    def test_project_overrides_personal(self, tmp_path):
        personal = tmp_path / "personal"
        project_skills = tmp_path / "project" / ".loongcli" / "skills"

        self._setup_skills(personal, {"debug/": VALID_SKILL})

        project_debug = """\
---
name: debug
description: 项目级调试工作流
---

项目自定义步骤。
"""
        self._setup_skills(project_skills, {"debug/": project_debug})

        reg = SkillRegistry(project_dir=tmp_path / "project", personal_dir=personal)
        meta = reg.get("debug")
        assert meta is not None
        assert "项目级" in meta.description

    def test_list_skills(self, tmp_path):
        skills_dir = tmp_path / ".loongcli" / "skills"
        self._setup_skills(skills_dir, {"debug/": VALID_SKILL})
        hello_dir = skills_dir / "hello"
        hello_dir.mkdir(parents=True, exist_ok=True)
        (hello_dir / "SKILL.md").write_text(MINIMAL_SKILL, encoding="utf-8")

        reg = SkillRegistry(project_dir=tmp_path, personal_dir=tmp_path / "empty")
        skills = reg.list_skills()
        assert len(skills) == 2
        names = [s.name for s in skills]
        assert "debug" in names
        assert "hello" in names

    def test_list_skills_exclude_model_disabled(self, tmp_path):
        skills_dir = tmp_path / ".loongcli" / "skills"
        self._setup_skills(skills_dir, {"debug/": VALID_SKILL})
        self._setup_skills(skills_dir, {"code-review/": SKILL_WITH_OPTIONS})

        reg = SkillRegistry(project_dir=tmp_path, personal_dir=tmp_path / "empty")
        all_skills = reg.list_skills(include_model_disabled=True)
        model_skills = reg.list_skills(include_model_disabled=False)
        assert len(all_skills) == 2
        assert len(model_skills) == 1
        assert model_skills[0].name == "debug"

    def test_load_content(self, tmp_path):
        skills_dir = tmp_path / ".loongcli" / "skills"
        self._setup_skills(skills_dir, {"debug/": VALID_SKILL})

        reg = SkillRegistry(project_dir=tmp_path, personal_dir=tmp_path / "empty")
        content = reg.load_content("debug")
        assert content is not None
        assert "复现问题" in content
        assert "name:" not in content

    def test_load_content_missing(self, tmp_path):
        reg = SkillRegistry(project_dir=tmp_path, personal_dir=tmp_path / "empty")
        assert reg.load_content("nonexistent") is None

    def test_build_listing(self, tmp_path):
        skills_dir = tmp_path / ".loongcli" / "skills"
        self._setup_skills(skills_dir, {"debug/": VALID_SKILL})

        reg = SkillRegistry(project_dir=tmp_path, personal_dir=tmp_path / "empty")
        listing = reg.build_listing()
        assert "debug" in listing

        verbose = reg.build_listing(verbose=True)
        assert "debug:" in verbose
        assert "调试" in verbose

    def test_build_listing_verbose_truncates_description(self, tmp_path):
        # 超长/多行描述：只取首行、截 80 字符加省略号——清单是一行一条的索引，
        # 单个技能的长描述不许吃掉整段前缀预算
        long_desc = "长" * 120
        skill_md = f"---\nname: chatty\ndescription: {long_desc}\n---\n\n正文\n"
        skills_dir = tmp_path / ".loongcli" / "skills"
        self._setup_skills(skills_dir, {"chatty/": skill_md})

        reg = SkillRegistry(project_dir=tmp_path, personal_dir=tmp_path / "empty")
        verbose = reg.build_listing(verbose=True)
        line = next(l for l in verbose.splitlines() if l.startswith("- chatty:"))
        assert "长" * 80 + "…" in line
        assert "长" * 81 not in line

    def test_nested_claude_skills(self, tmp_path):
        project = tmp_path / "myproject" / ".claude" / "skills" / "myskill"
        project.mkdir(parents=True)
        (project / "SKILL.md").write_text(VALID_SKILL, encoding="utf-8")
        reg = SkillRegistry(project_dir=None, personal_dir=tmp_path / "empty", extra_dirs=[tmp_path])
        assert reg.get("debug") is not None
        assert reg.get("debug").path == project / "SKILL.md"

    def test_nested_loongcli_skills(self, tmp_path):
        project = tmp_path / "myproject" / ".loongcli" / "skills" / "myskill"
        project.mkdir(parents=True)
        (project / "SKILL.md").write_text(MINIMAL_SKILL, encoding="utf-8")
        reg = SkillRegistry(project_dir=None, personal_dir=tmp_path / "empty", extra_dirs=[tmp_path])
        assert reg.get("hello") is not None

    def test_nested_not_override_direct(self, tmp_path):
        direct = tmp_path / "debug"
        direct.mkdir(parents=True)
        (direct / "SKILL.md").write_text(VALID_SKILL, encoding="utf-8")

        nested = tmp_path / "proj" / ".claude" / "skills" / "debug"
        nested.mkdir(parents=True)
        custom = VALID_SKILL.replace("系统化调试工作流", "嵌套版本")
        (nested / "SKILL.md").write_text(custom, encoding="utf-8")

        reg = SkillRegistry(project_dir=None, personal_dir=tmp_path / "empty", extra_dirs=[tmp_path])
        meta = reg.get("debug")
        assert "系统化" in meta.description

    def test_non_skill_md_silently_skipped(self, tmp_path):
        """Loose .md files in skill dirs are completely ignored."""
        skills_dir = tmp_path / ".loongcli" / "skills"
        skills_dir.mkdir(parents=True)
        (skills_dir / "CLAUDE.md").write_text("# Not a skill\nJust notes.", encoding="utf-8")
        (skills_dir / "README.md").write_text("# Readme", encoding="utf-8")
        (skills_dir / "hello.md").write_text(MINIMAL_SKILL, encoding="utf-8")
        reg = SkillRegistry(project_dir=tmp_path, personal_dir=tmp_path / "empty")
        assert len(reg.list_skills()) == 0

    def test_empty_registry(self, tmp_path):
        reg = SkillRegistry(project_dir=tmp_path, personal_dir=tmp_path / "empty")
        assert reg.list_skills() == []
        assert reg.build_listing() == ""
        assert reg.get("anything") is None
