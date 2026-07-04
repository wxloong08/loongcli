import pytest
from pathlib import Path
from loongcli.tools.glob_tool import GlobTool


@pytest.fixture
def tool():
    return GlobTool()


@pytest.fixture
def project_dir(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("main", encoding="utf-8")
    (tmp_path / "src" / "utils.py").write_text("utils", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_main.py").write_text("test", encoding="utf-8")
    (tmp_path / "README.md").write_text("readme", encoding="utf-8")
    return str(tmp_path)


@pytest.mark.asyncio
async def test_glob_py_files(tool, project_dir):
    result = await tool.execute(pattern="**/*.py", path=project_dir)
    assert "main.py" in result
    assert "utils.py" in result
    assert "test_main.py" in result
    assert "README.md" not in result


@pytest.mark.asyncio
async def test_glob_specific_dir(tool, project_dir):
    result = await tool.execute(pattern="src/*.py", path=project_dir)
    assert "main.py" in result
    assert "test_main.py" not in result


@pytest.mark.asyncio
async def test_glob_no_matches(tool, project_dir):
    result = await tool.execute(pattern="**/*.rs", path=project_dir)
    assert "未找到" in result


@pytest.mark.asyncio
async def test_glob_default_cwd(tool):
    result = await tool.execute(pattern="*.toml")
    assert "pyproject.toml" in result


def test_tool_schema(tool):
    assert tool.name == "glob"
    assert "pattern" in tool.parameters["properties"]


@pytest.mark.asyncio
async def test_glob_scan_limit(tool, tmp_path, monkeypatch):
    """Scan limit stops early on huge directories."""
    import loongcli.tools.glob_tool as mod

    monkeypatch.setattr(mod, "_MAX_SCAN", 5)
    for i in range(50):
        (tmp_path / f"file_{i}.txt").write_text("", encoding="utf-8")
    result = await tool.execute(pattern="*.txt", path=str(tmp_path))
    assert "扫描已截断" in result


# ── zero-result diagnostics tests ──


@pytest.mark.asyncio
async def test_zero_result_shows_dir_contents(tool, tmp_path):
    """On zero results, show directory contents as hints."""
    (tmp_path / "src").mkdir()
    (tmp_path / "README.md").write_text("", encoding="utf-8")
    result = await tool.execute(pattern="**/*.rs", path=str(tmp_path))
    assert "未找到" in result
    assert "src" in result  # directory listing should mention it


@pytest.mark.asyncio
async def test_backslash_suggests_forward_slash(tool, tmp_path):
    """Backslash in pattern → diagnostic hint."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("", encoding="utf-8")
    result = await tool.execute(pattern="src\\*.rs", path=str(tmp_path))
    assert "未找到" in result
    assert "反斜杠" in result or "backslash" in result.lower()


@pytest.mark.asyncio
async def test_similar_dir_suggestion(tool, tmp_path):
    """Typo in directory name → suggest similar existing dir."""
    (tmp_path / "components").mkdir()
    (tmp_path / "components" / "Button.tsx").write_text("", encoding="utf-8")
    result = await tool.execute(pattern="componets/*", path=str(tmp_path))  # typo
    assert "未找到" in result
    assert "components" in result  # should suggest the correct name


@pytest.mark.asyncio
async def test_trailing_slash_warns_directory_match(tool, tmp_path):
    """Pattern ending with / suggests adding *."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("", encoding="utf-8")
    result = await tool.execute(pattern="src/", path=str(tmp_path))
    assert "未找到" in result
    assert "/" in result  # diagnostic mentions the trailing slash issue


@pytest.mark.asyncio
async def test_empty_directory_shown(tool, tmp_path):
    """Empty directory → report it as empty."""
    (tmp_path / "just_this.txt").write_text("", encoding="utf-8")
    result = await tool.execute(pattern="**/*.rs", path=str(tmp_path))
    assert "未找到" in result
    # Should show the one file that exists
    assert "just_this.txt" in result


@pytest.mark.asyncio
async def test_glob_skips_unstatable_entry(tmp_path, monkeypatch):
    """单个条目 is_file 抛 OSError → 只计跳过，其余文件正常列出（不再整个扫描报错）。"""
    (tmp_path / "a.py").write_text("x", encoding="utf-8")
    (tmp_path / "badlink.py").write_text("x", encoding="utf-8")

    real_is_file = Path.is_file

    def fake_is_file(self, *a, **kw):
        if self.name == "badlink.py":
            raise OSError(1920, "系统无法访问此文件")
        return real_is_file(self, *a, **kw)

    monkeypatch.setattr(Path, "is_file", fake_is_file)

    result = await GlobTool().execute(pattern="*.py", path=str(tmp_path))
    assert "a.py" in result
    assert "跳过 1 个" in result
    assert not result.startswith("错误")


@pytest.mark.asyncio
async def test_glob_budget_not_burned_by_skip_dirs(tmp_path, monkeypatch):
    """同 grep：.venv 条目不得消耗 _MAX_SCAN 预算。"""
    import loongcli.tools.glob_tool as mod

    venv = tmp_path / ".venv" / "lib"
    venv.mkdir(parents=True)
    for i in range(30):
        (venv / f"junk{i:02d}.py").write_text("x", encoding="utf-8")
    (tmp_path / "zz_target.py").write_text("x", encoding="utf-8")

    monkeypatch.setattr(mod, "_MAX_SCAN", 5)

    result = await mod.GlobTool().execute(pattern="**/*.py", path=str(tmp_path))
    assert "zz_target.py" in result
    assert "扫描已截断" not in result
