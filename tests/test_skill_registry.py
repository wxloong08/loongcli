"""skill 注册器的扫描边界与命名空间回退单测。

设计取舍（2026-07-16）：全局技能是「精选」不是「扫到就算」——注册器只认
skill_dirs 里的直接子技能 + 项目约定目录（.claude/skills、.loongcli/skills），
刻意不递归任意 collection/skills/。技能集合（如 superpowers）要挂用户级，
在 skill_dirs 里显式指向其 skills/ 子目录（一次配置=一次背书）。
命名空间回退：外部技能内容里的 superpowers:writing-plans 引用剥前缀解析。
"""
import pytest
from pathlib import Path

from loongcli.skills.registry import SkillRegistry
from loongcli.tools.skill import SkillTool


def _mk_skill(base: Path, name: str, desc: str = "测试技能"):
    d = base / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {desc}\n---\n\n# {name} 的指令内容\n",
        encoding="utf-8",
    )


def _registry(tmp_path: Path, *skills_dirs: Path) -> SkillRegistry:
    # personal_dir 指向不存在的目录，隔离真实 ~/.loongcli/skills
    return SkillRegistry(
        personal_dir=tmp_path / "no-personal", extra_dirs=list(skills_dirs),
    )


class TestScanBoundary:
    def test_direct_skill_found(self, tmp_path):
        base = tmp_path / "skills"
        _mk_skill(base, "direct-one")
        assert _registry(tmp_path, base).get("direct-one")

    def test_collection_not_auto_scanned(self, tmp_path):
        # 集合里的嵌套技能不自动全局加载——避免 D:\skills 变成"扫到就算"的大仓库
        base = tmp_path / "skills"
        _mk_skill(base / "some-collection", "nested-skill")
        assert _registry(tmp_path, base).get("nested-skill") is None

    def test_collection_loads_when_pointed_explicitly(self, tmp_path):
        # 显式把集合的目录列进 skill_dirs → 其直接子技能加载（superpowers 的挂载方式）
        base = tmp_path / "skills"
        coll = base / "superpowers" / "skills"
        _mk_skill(coll, "writing-plans")
        _mk_skill(coll, "brainstorming")
        reg = _registry(tmp_path, base, coll)
        assert reg.get("writing-plans") and reg.get("brainstorming")
        # 但没显式指向时，从 base 扫不到它们
        assert _registry(tmp_path, base).get("writing-plans") is None

    def test_project_convention_dirs_scanned(self, tmp_path):
        # 项目约定目录仍认：collection/.loongcli/skills 与 .claude/skills
        base = tmp_path / "skills"
        _mk_skill(base / "proj" / ".loongcli" / "skills", "proj-skill")
        assert _registry(tmp_path, base).get("proj-skill")

    def test_dot_config_dirs_not_swept(self, tmp_path):
        # 别的工具的点配置目录（.claude/skills 位于 skill_dir 根）不被当技能源
        base = tmp_path / "skills"
        _mk_skill(base / ".claude" / "skills", "other-tool-skill")
        assert _registry(tmp_path, base).get("other-tool-skill") is None

    @pytest.mark.skipif(
        __import__("platform").system() != "Windows", reason="junction 是 Windows 概念"
    )
    def test_dangling_junction_does_not_crash(self, tmp_path):
        # 真机事故：悬空 junction 抛 WinError 1920，一个坏条目放倒整个注册器
        import _winapi
        import shutil

        base = tmp_path / "skills"
        _mk_skill(base, "healthy-one")
        target = tmp_path / "victim"
        target.mkdir()
        _winapi.CreateJunction(str(target), str(base / "dangling"))
        shutil.rmtree(target)  # 现在 base/dangling 是悬空 junction

        reg = _registry(tmp_path, base)  # 不应抛异常
        assert reg.get("healthy-one")


class TestNamespaceFallback:
    async def test_namespaced_name_resolves(self, tmp_path):
        base = tmp_path / "skills"
        _mk_skill(base, "writing-plans")
        tool = SkillTool(_registry(tmp_path, base))
        result = await tool.execute("superpowers:writing-plans")
        assert "writing-plans 的指令内容" in result

    async def test_plain_name_still_works(self, tmp_path):
        base = tmp_path / "skills"
        _mk_skill(base, "jobhunter-like")
        tool = SkillTool(_registry(tmp_path, base))
        result = await tool.execute("jobhunter-like")
        assert "jobhunter-like 的指令内容" in result

    async def test_unknown_name_reports_available(self, tmp_path):
        base = tmp_path / "skills"
        _mk_skill(base, "only-one")
        tool = SkillTool(_registry(tmp_path, base))
        result = await tool.execute("superpowers:nonexistent")
        assert "未找到技能" in result
